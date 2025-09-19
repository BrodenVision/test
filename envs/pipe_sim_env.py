import numpy as np
import gymnasium as gym
from gym import spaces
from collections import deque
import torch

# 一个简化的模型定义，实际应用中可以从真实模型中profile得到
# (计算成本, 激活大小)
DEFAULT_MODEL_LAYERS = [
    (10, 32), (12, 64), (11, 64), (15, 128), (14, 128), (18, 256), (17, 256),
    (20, 512), (22, 512), (25, 512), (24, 1024), (28, 1024), (30, 512), (26, 256)
]


class PipeSimEnv(gym.Env):
    """
    流水线并行训练的高保真仿真环境 (PipeSimEnv)
    """

    def __init__(self, args):
        self.args = args
        self.num_gpus = args.num_agents  # 将 num_agents 视为 num_gpus
        self.model_layers = DEFAULT_MODEL_LAYERS
        self.num_layers = len(self.model_layers)

        # 资源动态变化轨迹
        # 这里用一个简单的周期函数模拟，实际可从文件中加载
        self.time_step = 0
        self.compute_power = np.ones(self.num_gpus)
        self.bandwidth = np.ones(self.num_gpus - 1)

        # 状态表示
        # 每个GPU的观察: [GPU索引, 持有层数, 总计算负载, 发送带宽, 接收带宽]
        self.observation_space = []
        self.share_observation_space = []
        self.obs_dim = 5
        share_obs_dim = 0
        for i in range(self.num_gpus):
            self.observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32))
            share_obs_dim += self.obs_dim
        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
                                        for _ in range(self.num_gpus)]

        # 动作空间: 3维连续动作 [向左迁移意愿, 向右迁移意愿, 触发阈值]
        # 意愿和阈值范围都设为 [0, 1]
        self.action_space = [spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32) for _ in range(self.num_gpus)]

        # 初始化模型层划分
        self.partition = self._get_uniform_partition()
        self.current_throughput = self._calculate_throughput(self.partition)

        # 用于计算奖励的吞吐量历史记录
        self.throughput_history = deque(maxlen=10)
        self.throughput_history.append(self.current_throughput)

    def reset(self):
        self.time_step = 0
        self.partition = self._get_uniform_partition()
        self.current_throughput = self._calculate_throughput(self.partition)
        self.throughput_history.clear()
        self.throughput_history.append(self.current_throughput)
        return self._get_obs()

    def step(self, action_decision):
        """
        执行一步环境演化
        :param action_decision: 来自仲裁器的最终决策，例如 (from_gpu, layer_index_to_move)
                                to_gpu 是隐式的 (from_gpu+1 or from_gpu-1)
                                如果为None，则不发生迁移
        """
        self.time_step += 1
        migration_cost = 0

        if action_decision:
            from_gpu, to_gpu, layer_idx_global = action_decision

            # 执行迁移
            original_layer = self.model_layers[layer_idx_global]
            self.partition[from_gpu].remove(layer_idx_global)
            self.partition[to_gpu].append(layer_idx_global)
            self.partition[to_gpu].sort()  # 保持有序

            # 计算迁移成本 (简化模型：与激活大小成正比)
            migration_cost = original_layer[1] * self.args.migration_cost_factor

        # 更新资源状态
        self._update_resources()

        # 计算新划分下的吞吐量
        new_throughput = self._calculate_throughput(self.partition)
        self.current_throughput = new_throughput
        self.throughput_history.append(new_throughput)

        # 计算奖励
        reward = self._get_reward(migration_cost)

        # 获取新观察
        obs = self._get_obs()

        # done 信号（可以设置为一个 episode 结束后为 True）
        done = self.time_step >= self.args.episode_length
        dones = np.full(self.num_gpus, done)

        # info - 必须是每个 agent 一个 dict
        infos = [{} for _ in range(self.num_gpus)]

        # 奖励 - 必须是 (num_gpus, 1) 的形状以匹配 buffer
        rewards = [[reward] for _ in range(self.num_gpus)]

        return obs, rewards, dones, infos

    def _get_obs(self):
        """
        为每个智能体生成观察
        """
        all_obs = []
        for i in range(self.num_gpus):
            num_layers_held = len(self.partition[i])
            compute_load = sum(self.model_layers[l_idx][0] for l_idx in self.partition[i])

            send_bw = self.bandwidth[i] if i < self.num_gpus - 1 else 0
            recv_bw = self.bandwidth[i - 1] if i > 0 else 0

            obs = np.array([
                float(i) / self.num_gpus,
                float(num_layers_held) / self.num_layers,
                float(compute_load) / sum(l[0] for l in self.model_layers),
                send_bw,
                recv_bw
            ])
            all_obs.append(obs)
        return np.array(all_obs)

    def _get_reward(self, migration_cost):
        """
        定义奖励函数：吞吐量提升 - 迁移成本
        """
        # 使用历史平均吞吐量作为基线，使奖励更平滑
        avg_throughput = np.mean(self.throughput_history)
        throughput_improvement = self.current_throughput - avg_throughput

        reward = self.args.reward_throughput_factor * throughput_improvement - migration_cost
        return reward

    def _calculate_throughput(self, partition):
        """
        根据当前划分和资源状态计算流水线吞吐量
        吞吐量由最慢的阶段（瓶颈）决定
        """
        stage_times = np.zeros(self.num_gpus)
        for i in range(self.num_gpus):
            # 计算时间
            compute_time = sum(self.model_layers[l_idx][0] for l_idx in partition[i]) / self.compute_power[i]

            # 通信时间
            comm_time = 0
            if i < self.num_gpus - 1 and len(partition[i]) > 0:
                # 找到当前GPU上的最后一层，以确定需要传输的激活大小
                last_layer_idx = max(partition[i])
                activation_size = self.model_layers[last_layer_idx][1]
                comm_time = activation_size / self.bandwidth[i]

            stage_times[i] = compute_time + comm_time

        bottleneck_time = np.max(stage_times) if len(stage_times) > 0 else 1.0

        # 避免除以零
        if bottleneck_time <= 1e-6:
            return 1e6

        return 1.0 / bottleneck_time

    def _update_resources(self):
        """
        模拟资源动态变化
        """
        # 简单的正弦波模拟算力波动
        self.compute_power = 0.8 + 0.2 * np.sin(2 * np.pi * self.time_step / 50 + np.arange(self.num_gpus) * np.pi / 4)
        # 简单的正弦波模拟带宽波动
        self.bandwidth = 0.7 + 0.3 * np.cos(2 * np.pi * self.time_step / 80 + np.arange(self.num_gpus - 1) * np.pi / 3)

    def _get_uniform_partition(self):
        """
        生成一个均匀的模型层划分
        """
        partition = [[] for _ in range(self.num_gpus)]
        layers_per_gpu = self.num_layers // self.num_gpus
        remainder = self.num_layers % self.num_gpus

        current_layer = 0
        for i in range(self.num_gpus):
            num_layers_on_this_gpu = layers_per_gpu + (1 if i < remainder else 0)
            for _ in range(num_layers_on_this_gpu):
                partition[i].append(current_layer)
                current_layer += 1
        return partition

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def close(self):
        pass

