import time
import numpy as np
import torch
from runner.shared.base_runner import Runner
from utils.arbitrator import Arbitrator

def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    """
    Runner class for AutoPipe.
    - Uses a shared policy (CTDE).
    - Integrates the Arbitrator for decision making.
    """

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        # 为每个并行的环境实例创建一个仲裁器
        self.arbitrators = [Arbitrator(self.all_args, env.partition, env.model_layers) for env in self.envs.envs]

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # 采样动作并让仲裁器做决策
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Arbitrator做出最终决策
                # actions shape: (n_rollout_threads, num_agents, 3)
                action_decisions = [self.arbitrators[i].decide(actions[i]) for i in range(self.n_rollout_threads)]

                # 与环境交互
                obs, rewards, dones, infos = self.envs.step(action_decisions)

                # 更新仲裁器内部的状态 (partition)
                for i in range(self.n_rollout_threads):
                    self.arbitrators[i].partition = self.envs.envs[i].partition

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions, # 存储的是原始的3维连续动作
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # 将数据存入缓冲区
                self.insert(data)

            # 计算回报并更新网络
            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Algo {} Exp {} updates {}/{}, total steps {}/{}, FPS {}.\n".format(
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        obs = self.envs.reset()
        share_obs = obs.reshape(self.n_rollout_threads, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = obs.reshape(self.n_rollout_threads, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )
