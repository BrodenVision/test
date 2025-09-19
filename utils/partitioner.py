import numpy as np


class DynamicProgrammingPartitioner:
    """
    根据 PipeDream 论文中的思想，使用动态规划来找到最优的模型静态划分。
    目标：最小化流水线中最慢阶段的处理时间。
    """

    def __init__(self, model_profile, num_gpus, compute_powers, bandwidths):
        """
        初始化分区器。

        Args:
            model_profile (dict): 包含模型各层性能指标的字典。
            num_gpus (int): GPU 的数量。
            compute_powers (list): 各 GPU 的初始算力系数。
            bandwidths (list): 各链路的初始带宽 (MB/s)。
        """
        self.model_profile = model_profile
        self.num_layers = len(model_profile['layers'])
        self.num_gpus = num_gpus
        self.compute_powers = compute_powers
        self.bandwidths = bandwidths

        # 预计算每一层的成本
        self.layer_costs = self._precompute_layer_costs()

        # DP 表和回溯表
        # dp[k][j] 表示将前 j+1 层划分到 k+1 个 GPU 上的最小瓶颈时间
        self.dp_table = np.full((num_gpus, self.num_layers), np.inf)
        # backtrack_table[k][j] 记录了在最优解中，第 k+1 个 GPU 开始的层索引
        self.backtrack_table = np.full((num_gpus, self.num_layers), -1, dtype=int)

    def _precompute_layer_costs(self):
        """预计算每层的计算和通信成本。"""
        costs = []
        for layer in self.model_profile['layers']:
            # 注意：这里的成本是未除以资源前的基础成本
            compute_cost = layer['forward_ms'] + layer['backward_ms']
            comm_cost = (layer['activations_mb']) * 1000  # 转换为 MB*ms, 带宽单位是 MB/s
            costs.append({'compute': compute_cost, 'comm': comm_cost})
        return costs

    def _get_stage_cost(self, start_layer, end_layer, gpu_id):
        """计算将 [start_layer, end_layer] 放在单个 GPU 上的总时间。"""
        if start_layer > end_layer:
            return 0

        # 计算成本
        total_compute = sum(self.layer_costs[i]['compute'] for i in range(start_layer, end_layer + 1))
        # 通信成本由该阶段的最后一层决定
        total_comm = self.layer_costs[end_layer]['comm']

        # 应用当前 GPU 的资源状态
        time_on_gpu = total_compute / self.compute_powers[gpu_id]
        if gpu_id < self.num_gpus - 1:  # 最后一个GPU没有出向通信
            time_on_gpu += total_comm / self.bandwidths[gpu_id]

        return time_on_gpu

    def partition(self):
        """执行动态规划算法来找到最优分区。"""
        # --- 初始化：将前 j 层都放在 GPU 0 上 ---
        for j in range(self.num_layers):
            self.dp_table[0, j] = self._get_stage_cost(0, j, 0)
            self.backtrack_table[0, j] = 0

        # --- 填充 DP 表 ---
        # k: GPU 数量 (从第 2 个 GPU 开始)
        for k in range(1, self.num_gpus):
            # j: 层的数量
            for j in range(self.num_layers):
                # i: 切分点
                for i in range(j + 1):
                    # 将层 [i, j] 分配给第 k 个 GPU
                    cost_of_last_stage = self._get_stage_cost(i, j, k)

                    # 之前 k-1 个 GPU 处理层 [0, i-1] 的最优成本
                    cost_of_previous_stages = self.dp_table[k - 1, i - 1] if i > 0 else 0

                    # 当前划分的瓶颈时间
                    current_bottleneck = max(cost_of_last_stage, cost_of_previous_stages)

                    if current_bottleneck < self.dp_table[k, j]:
                        self.dp_table[k, j] = current_bottleneck
                        self.backtrack_table[k, j] = i

        # --- 回溯找到最优划分方案 ---
        partition = [[] for _ in range(self.num_gpus)]
        current_layer_idx = self.num_layers - 1

        for k in range(self.num_gpus - 1, -1, -1):
            start_layer_idx = self.backtrack_table[k, current_layer_idx]
            if start_layer_idx == -1:  # 如果没有找到划分，说明GPU过多，将空列表分配给前面的GPU
                continue

            layers_for_this_gpu = list(range(start_layer_idx, current_layer_idx + 1))
            partition[k] = layers_for_this_gpu
            current_layer_idx = start_layer_idx - 1

        return partition
