import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import matplotlib.pyplot as plt
import io
from PIL import Image

from utils.profiler import get_model_profile
from utils.trace import ResourceTrace
from utils.partitioner import DynamicProgrammingPartitioner


class PipeSimEnv(gym.Env):
    """
    流水线并行训练的高保真仿真环境 (PipeSimEnv)
    - 使用 ARPipe 论文中的精确时间模型
    """
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, args):
        self.args = args
        self.num_gpus = args.num_agents
        self.num_micro_batches = args.num_micro_batches

        self.model_profile = get_model_profile(args.model_name)
        self.num_layers = len(self.model_profile['layers'])

        self.trace_generator = ResourceTrace(self.num_gpus, args.episode_length, args.trace_path)
        self.compute_trace, self.bandwidth_trace = self.trace_generator.get_trace(args.trace_type)
        self.compute_power = self.compute_trace[0]
        self.bandwidth = self.bandwidth_trace[0]

        # 状态表示
        self.observation_space = []
        self.share_observation_space = []
        self.obs_dim = 5
        share_obs_dim = self.obs_dim * self.num_gpus
        for i in range(self.num_gpus):
            self.observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32))
        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
                                        for _ in range(self.num_gpus)]

        # 动作空间
        self.action_space = [spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32) for _ in range(self.num_gpus)]

        # 初始化
        self.time_step = 0
        self.partition = self._get_initial_partition()
        self.current_throughput = self._calculate_throughput_arpip(self.partition)
        self.rewards_history = deque(maxlen=args.episode_length)

    def _get_initial_partition(self):
        """根据策略获取初始分区"""
        if self.args.initial_partition_strategy == "dp":
            print("--- 使用动态规划 (DP) 生成初始分区 ---")
            partitioner = DynamicProgrammingPartitioner(
                self.model_profile, self.num_gpus, self.compute_power, self.bandwidth
            )
            return partitioner.partition()
        else:
            print("--- 使用均匀 (Uniform) 方式生成初始分区 ---")
            return self._get_uniform_partition()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.time_step = 0
        self.compute_trace, self.bandwidth_trace = self.trace_generator.get_trace(self.args.trace_type)
        self.compute_power = self.compute_trace[0]
        self.bandwidth = self.bandwidth_trace[0]

        self.partition = self._get_initial_partition()

        self.current_throughput = self._calculate_throughput_arpip(self.partition)
        self.rewards_history.clear()

        obs = self._get_obs()
        info = {}  # Gymnasium 需要返回 info 字典
        return obs, info

    def step(self, action_decision):
        migration_cost = 0
        if action_decision:
            from_gpu, to_gpu, layer_idx_global = action_decision
            if layer_idx_global in self.partition[from_gpu] and abs(from_gpu - to_gpu) == 1:
                layer_profile = self.model_profile['layers'][layer_idx_global]
                self.partition[from_gpu].remove(layer_idx_global)
                self.partition[to_gpu].append(layer_idx_global)
                self.partition[to_gpu].sort()
                migration_cost_time = (layer_profile['params_mb'] / self.bandwidth[min(from_gpu, to_gpu)]) * 1000
                migration_cost = self.args.migration_cost_factor * migration_cost_time

        self.time_step += 1
        terminated = self.time_step >= self.args.episode_length
        truncated = False  # 我们不使用 truncated

        if not terminated:
            self.compute_power = self.compute_trace[self.time_step]
            self.bandwidth = self.bandwidth_trace[self.time_step]

        new_throughput = self._calculate_throughput_arpip(self.partition)
        reward = self._get_reward(self.current_throughput, new_throughput, migration_cost)
        self.current_throughput = new_throughput
        self.rewards_history.append(reward)

        obs = self._get_obs()
        infos = [{} for _ in range(self.num_gpus)]
        rewards = [[reward] for _ in range(self.num_gpus)]

        return obs, rewards, np.array([terminated] * self.num_gpus), np.array([truncated] * self.num_gpus), infos

    def _get_stage_times_per_microbatch(self, partition):
        """计算每个阶段处理单个微批次所需的前向、后向和通信时间 (ms)"""
        stage_fwd_times = np.zeros(self.num_gpus)
        stage_bwd_times = np.zeros(self.num_gpus)
        stage_comm_times = np.zeros(self.num_gpus - 1)

        for i in range(self.num_gpus):
            if not partition[i]: continue

            # 计算时间 (除以微批次数量)
            fwd_compute = sum(self.model_profile['layers'][l_idx]['forward_ms'] for l_idx in partition[i])
            bwd_compute = sum(self.model_profile['layers'][l_idx]['backward_ms'] for l_idx in partition[i])
            stage_fwd_times[i] = (fwd_compute / self.num_micro_batches) / self.compute_power[i]
            stage_bwd_times[i] = (bwd_compute / self.num_micro_batches) / self.compute_power[i]

            # 通信时间 (除以微批次数量)
            if i < self.num_gpus - 1:
                last_layer_idx = max(partition[i])
                activation_size = self.model_profile['layers'][last_layer_idx]['activations_mb']
                stage_comm_times[i] = ((activation_size / self.num_micro_batches) / self.bandwidth[i]) * 1000

        return stage_fwd_times, stage_bwd_times, stage_comm_times

    def _calculate_throughput_arpip(self, partition):
        """
        根据 ARPipe 论文中的公式 (5), (9), (10) 计算吞吐量。
        这个模型更精确地描述了流水线的填充、稳态和排空阶段。
        """
        if self.num_micro_batches < 1:
            return 0.0

        tf, tb, te = self._get_stage_times_per_microbatch(partition)

        # 瓶颈由最慢的阶段决定 (计算+通信)
        stage_work_times = tf + tb
        bottleneck_stage_time = np.max(stage_work_times) if len(stage_work_times) > 0 else 1.0

        # ARPipe 公式 (9): 前向阶段总时间
        total_forward_time = np.sum(tf) + np.sum(te) + (self.num_micro_batches - 1) * np.max(tf)

        # ARPipe 公式 (10): 后向阶段总时间
        total_backward_time = np.sum(tb) + np.sum(te) + (self.num_micro_batches - 1) * np.max(tb)

        # ARPipe 公式 (5) 的精神：总时间是前向和后向阶段之和
        # 注意：论文中的公式(5)本身似乎有误，因为它将所有项加了两次。
        # 一个更合理的解释是总时间 T = Tf + Tb。
        # 我们使用一个更标准的流水线总时间公式：
        # 总时间 = (阶段数 - 1 + 微批次数) * 瓶颈时间 + (填充和排空的其他开销)
        # 这里我们直接使用 ARPipe 的 Tf 和 Tb 作为更精确的估计
        total_time_ms = total_forward_time + total_backward_time

        if total_time_ms <= 1e-6:
            return 1e6  # 避免除以零

        # 吞吐量 = (总批次大小) / (总时间 秒)
        # 注意这里的批次大小是1，因为我们计算的是单位批次的时间
        throughput = 1.0 / (total_time_ms / 1000.0)

        return throughput

    def _get_obs(self):
        all_obs = []
        total_fwd_cost = sum(l['forward_ms'] for l in self.model_profile['layers'])

        for i in range(self.num_gpus):
            num_layers_held = len(self.partition[i])
            compute_load = sum(self.model_profile['layers'][l_idx]['forward_ms'] for l_idx in self.partition[i])

            norm_compute_power = self.compute_power[i]
            norm_send_bw = (self.bandwidth[i] / 1000) if i < self.num_gpus - 1 else 0
            norm_recv_bw = (self.bandwidth[i - 1] / 1000) if i > 0 else 0

            obs = np.array([
                float(i) / self.num_gpus,
                float(num_layers_held) / self.num_layers if self.num_layers > 0 else 0,
                float(compute_load) / total_fwd_cost if total_fwd_cost > 0 else 0,
                norm_send_bw,
                norm_recv_bw
            ])
            all_obs.append(obs)

        return np.array(all_obs)

    def _get_reward(self, old_throughput, new_throughput, migration_cost):
        batch_size = self.args.n_rollout_threads  # 假设一个 rollout thread 对应一个批次
        if old_throughput < 1e-6 or new_throughput < 1e-6:
            time_saving_ms = 0
        else:
            time_saving_ms = ((1.0 / old_throughput) - (1.0 / new_throughput)) * batch_size * 1000

        reward = self.args.reward_throughput_factor * time_saving_ms - migration_cost
        return reward

    def _get_uniform_partition(self):
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
        plt.close('all')

    def render(self, mode='rgb_array'):
        # 渲染代码保持不变
        if mode == 'rgb_array':
            fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
            fig.patch.set_facecolor('#f0f0f0')

            gpu_colors = plt.get_cmap('viridis', self.num_gpus)
            all_fwd_costs = [l['forward_ms'] for l in self.model_profile['layers']]
            max_layer_cost = max(all_fwd_costs) if all_fwd_costs else 1

            tf, tb, _ = self._get_stage_times_per_microbatch(self.partition)
            stage_total = (tf + tb) * self.num_micro_batches
            bottleneck_time = np.max(stage_total) if len(stage_total) > 0 else 0

            for i in range(self.num_gpus):
                is_bottleneck = np.isclose(stage_total[i], bottleneck_time) and bottleneck_time > 0
                edgecolor = 'red' if is_bottleneck else 'black'
                linewidth = 3.0 if is_bottleneck else 1.0

                ax.add_patch(plt.Rectangle((i * 1.5, 0), 1, 6, facecolor=gpu_colors(i), alpha=0.1, edgecolor=edgecolor,
                                           linewidth=linewidth, zorder=0))
                ax.text(i * 1.5 + 0.5, 6.2, f'GPU {i}', ha='center', fontsize=12, weight='bold')
                ax.text(i * 1.5 + 0.5, -0.4, f'阶段耗时: {stage_total[i]:.1f}ms', ha='center', fontsize=9,
                        color='red' if is_bottleneck else 'black')

                if self.partition[i]:
                    y_pos = 5.5
                    for layer_idx in sorted(self.partition[i]):
                        layer_cost = self.model_profile['layers'][layer_idx]['forward_ms']
                        alpha_val = 0.4 + 0.6 * (layer_cost / max_layer_cost)
                        ax.add_patch(
                            plt.Rectangle((i * 1.5 + 0.1, y_pos), 0.8, 0.4, facecolor=gpu_colors(i), alpha=alpha_val,
                                          zorder=1))
                        ax.text(i * 1.5 + 0.5, y_pos + 0.2, f'L{layer_idx}', ha='center', va='center', color='white',
                                fontsize=8, weight='bold')
                        y_pos -= 0.5

            for i in range(self.num_gpus - 1):
                bw = self.bandwidth[i]
                linewidth = 1 + (bw / 2000) * 8
                linecolor = '#888888'
                ax.plot([i * 1.5 + 1, (i + 1) * 1.5], [3, 3], color=linecolor, linewidth=linewidth, linestyle='-',
                        zorder=2)
                ax.text((i * 1.5 + 1.25), 3.2, f'{bw:.0f} MB/s', ha='center', fontsize=8,
                        bbox=dict(facecolor='#f0f0f0', edgecolor='none', pad=1))

            for i in range(self.num_gpus):
                power = self.compute_power[i]
                ax.text(i * 1.5 + 0.5, -1.0, f'算力: {power:.2f}x', ha='center', fontsize=9)

            title_str = f'Pipe 状态面板 (模型: {self.args.model_name}, 轨迹: {self.args.trace_type}) - 时间步: {self.time_step}'
            reward_val = self.rewards_history[-1] if self.rewards_history else 0
            info_str = f'吞吐量: {self.current_throughput:.2f} samples/sec | 上一步奖励: {reward_val:.2f}'
            ax.set_title(title_str, fontsize=16)
            fig.text(0.5, 0.01, info_str, ha='center', fontsize=12)

            ax.set_xlim(-0.5, self.num_gpus * 1.5 - 0.5)
            ax.set_ylim(-1.5, 7)
            ax.axis('off')
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            img = Image.open(buf)
            img_arr = np.array(img)
            buf.close()
            plt.close(fig)

            return img_arr[:, :, :3]
        else:
            raise NotImplementedError
