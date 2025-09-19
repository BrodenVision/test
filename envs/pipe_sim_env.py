import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import matplotlib.pyplot as plt
import io
from PIL import Image
import sys
import os

# --- 路径修复 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.profiler import get_model_profile
from utils.trace import ResourceTrace
from utils.partitioner import DynamicProgrammingPartitioner


class PipeSimEnv(gym.Env):
    """
    流水线并行训练的高保真仿真环境 (PipeSimEnv)
    - 新增支持同步(sync)和异步(async)两种时间模型
    - 丰富了观察空间以提供更多决策依据
    """
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, args):
        self.args = args
        self.num_gpus = args.num_agents
        self.num_micro_batches = args.num_micro_batches
        self.time_model = args.pipeline_time_model

        self.model_profile = get_model_profile(args.model_name)
        self.num_layers = len(self.model_profile['layers'])

        self.trace_generator = ResourceTrace(self.num_gpus, args.episode_length, args.trace_path)
        self.compute_trace, self.bandwidth_trace = self.trace_generator.get_trace(args.trace_type)

        # --- 1. 更丰富的观察空间 ---
        # [self_compute_load, self_comm_load, left_neighbor_load, right_neighbor_load, self_compute_power, left_bw, right_bw]
        self.obs_dim = 7
        share_obs_dim = self.obs_dim * self.num_gpus
        self.observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32) for _
                                  in range(self.num_gpus)]
        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
                                        for _ in range(self.num_gpus)]

        # 动作空间保持不变
        self.action_space = [spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32) for _ in range(self.num_gpus)]

        # 初始化状态变量
        self.time_step = 0
        self.compute_power = self.compute_trace[0]
        self.bandwidth = self.bandwidth_trace[0]
        self.partition = self._get_initial_partition()
        self.current_batch_time = self._get_batch_time(self.partition)
        self.rewards_history = deque(maxlen=args.episode_length)

    def _get_initial_partition(self):
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
        self.current_batch_time = self._get_batch_time(self.partition)
        self.rewards_history.clear()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action_decision):
        migration_overhead_ms = 0
        if action_decision:
            from_gpu, to_gpu, layer_idx_global = action_decision
            if layer_idx_global in self.partition[from_gpu] and abs(from_gpu - to_gpu) == 1:
                layer_profile = self.model_profile['layers'][layer_idx_global]
                self.partition[from_gpu].remove(layer_idx_global)
                self.partition[to_gpu].append(layer_idx_global)
                self.partition[to_gpu].sort()
                # 确保带宽索引不出界
                bw_index = min(from_gpu, to_gpu)
                if bw_index < len(self.bandwidth):
                    transfer_time_ms = (layer_profile['params_mb'] / self.bandwidth[bw_index]) * 1000
                    migration_overhead_ms = self.args.migration_cost_factor * transfer_time_ms

        self.time_step += 1
        terminated = self.time_step >= self.args.episode_length
        truncated = False
        if not terminated:
            self.compute_power = self.compute_trace[self.time_step]
            self.bandwidth = self.bandwidth_trace[self.time_step]

        new_batch_time = self._get_batch_time(self.partition)
        reward = self._get_reward(self.current_batch_time, new_batch_time, migration_overhead_ms)
        self.current_batch_time = new_batch_time
        self.rewards_history.append(reward)
        obs = self._get_obs()
        infos = [{"throughput": 1000.0 / self.current_batch_time if self.current_batch_time > 0 else 0} for _ in
                 range(self.num_gpus)]
        rewards = [[reward] for _ in range(self.num_gpus)]

        return obs, rewards, np.array([terminated] * self.num_gpus), np.array([truncated] * self.num_gpus), infos

    def _get_stage_costs(self, partition):
        """计算每个阶段的计算和通信成本（整个批次）。"""
        stage_compute_costs = np.zeros(self.num_gpus)
        stage_comm_costs = np.zeros(self.num_gpus)

        for i in range(self.num_gpus):
            if not partition[i]: continue

            fwd_compute = sum(self.model_profile['layers'][l_idx]['forward_ms'] for l_idx in partition[i])
            bwd_compute = sum(self.model_profile['layers'][l_idx]['backward_ms'] for l_idx in partition[i])
            stage_compute_costs[i] = (fwd_compute + bwd_compute) / self.compute_power[i]

            if i < self.num_gpus - 1:
                last_layer_idx = max(partition[i])
                activation_size = self.model_profile['layers'][last_layer_idx]['activations_mb']
                stage_comm_costs[i] = ((activation_size * 2) / self.bandwidth[i]) * 1000

        return stage_compute_costs, stage_comm_costs

    def _get_batch_time(self, partition):
        """根据配置的时间模型计算批处理时间。"""
        if self.time_model == 'async':
            return self._get_async_batch_time(partition)
        elif self.time_model == 'sync':
            return self._get_sync_batch_time(partition)
        else:
            raise ValueError(f"未知的时间模型: {self.time_model}")

    def _get_async_batch_time(self, partition):
        """根据 PipeDream/DynPipe 的异步1F1B模型计算时间。"""
        if not any(p for p in partition if p): return float('inf')

        stage_compute_costs, stage_comm_costs = self._get_stage_costs(partition)
        stage_total_times = stage_compute_costs + stage_comm_costs

        bottleneck_per_batch = np.max(stage_total_times)

        return bottleneck_per_batch

    def _get_sync_batch_time(self, partition):
        """根据 ARPipe 论文的同步模型计算时间。"""
        if self.num_micro_batches < 1 or not any(p for p in partition if p): return float('inf')

        stage_compute_costs, stage_comm_costs = self._get_stage_costs(partition)

        stage_total_micro_time = (stage_compute_costs + stage_comm_costs) / self.num_micro_batches

        bottleneck_micro_time = np.max(stage_total_micro_time) if len(
            stage_total_micro_time[stage_total_micro_time > 0]) > 0 else 0

        total_time_ms = np.sum(stage_total_micro_time) * (self.num_gpus - 1) + (
                    self.num_micro_batches - 1) * bottleneck_micro_time

        return total_time_ms if total_time_ms > 0 else float('inf')

    def _get_obs(self):
        """生成更丰富的观察。"""
        all_obs = []
        stage_compute_costs, stage_comm_costs = self._get_stage_costs(self.partition)

        total_compute_base = sum(l['forward_ms'] + l['backward_ms'] for l in self.model_profile['layers'])
        total_comm_base = sum(l['activations_mb'] for l in self.model_profile['layers'])

        for i in range(self.num_gpus):
            my_compute_load = stage_compute_costs[i]
            my_comm_load = stage_comm_costs[i]

            left_neighbor_load = stage_compute_costs[i - 1] if i > 0 else 0
            right_neighbor_load = stage_compute_costs[i + 1] if i < self.num_gpus - 1 else 0

            my_compute_power = self.compute_power[i]
            left_bw = self.bandwidth[i - 1] if i > 0 else 0
            right_bw = self.bandwidth[i] if i < self.num_gpus - 1 else 0

            obs = np.array([
                my_compute_load / total_compute_base if total_compute_base > 0 else 0,
                my_comm_load / total_comm_base if total_comm_base > 0 else 0,
                left_neighbor_load / total_compute_base if total_compute_base > 0 else 0,
                right_neighbor_load / total_compute_base if total_compute_base > 0 else 0,
                my_compute_power,
                left_bw / 1000.0,
                right_bw / 1000.0,
            ])
            all_obs.append(obs)

        return np.array(all_obs)

    def _get_reward(self, old_batch_time_ms, new_batch_time_ms, migration_overhead_ms):
        time_saving_ms = old_batch_time_ms - new_batch_time_ms
        reward = self.args.reward_throughput_factor * time_saving_ms
        reward -= migration_overhead_ms
        num_active_stages = len([p for p in self.partition if p])
        staleness_penalty = self.args.staleness_penalty_factor * (num_active_stages ** 2)
        reward -= staleness_penalty
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
        if mode == 'rgb_array':
            fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
            fig.patch.set_facecolor('#f0f0f0')

            gpu_colors = plt.get_cmap('viridis', self.num_gpus)

            compute_costs, comm_costs = self._get_stage_costs(self.partition)
            stage_total = compute_costs + comm_costs
            bottleneck_time = np.max(stage_total) if len(stage_total) > 0 and np.any(stage_total) else 0

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
                        ax.add_patch(plt.Rectangle((i * 1.5 + 0.1, y_pos), 0.8, 0.4, facecolor=gpu_colors(i), alpha=0.6,
                                                   zorder=1))
                        ax.text(i * 1.5 + 0.5, y_pos + 0.2, f'L{layer_idx}', ha='center', va='center', color='white',
                                fontsize=8, weight='bold')
                        y_pos -= 0.5

            title_str = f'Pipe 状态面板 (模型: {self.args.model_name}, 时间模型: {self.time_model}) - 时间步: {self.time_step}'
            reward_val = self.rewards_history[-1] if self.rewards_history else 0
            throughput = 1000.0 / self.current_batch_time if self.current_batch_time > 0 and self.current_batch_time != float(
                'inf') else 0
            info_str = f'总批次时间: {self.current_batch_time:.2f} ms | 吞吐量: {throughput:.2f} samples/sec | 上一步奖励: {reward_val:.2f}'
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

