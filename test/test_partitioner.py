import sys
import os
import numpy as np

# --- 路径修复：确保可以从项目根目录导入模块 ---
# 这段代码会将项目的根目录添加到Python的搜索路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# --- 路径修复结束 ---

from utils.partitioner import DynamicProgrammingPartitioner
from utils.profiler import get_model_profile


def print_partition_results(partition, partitioner):
    """
    以美观、易读的方式打印分区结果和成本分析。
    """
    print("\n--- 最优划分方案已找到 ---")
    if not any(partition):
        print("未能生成有效划分。可能是GPU数量超过了层数。")
        return

    stage_costs = []
    for i, layers in enumerate(partition):
        if not layers:
            print(f"GPU {i}: (空闲)")
            stage_costs.append(0)
            continue

        print(f"GPU {i}: Layers {layers}")
        start_layer = min(layers)
        end_layer = max(layers)
        cost = partitioner._get_stage_cost(start_layer, end_layer, i)
        stage_costs.append(cost)

    bottleneck_time = max(stage_costs) if stage_costs else 0

    print("\n--- 阶段成本分析 ---")
    for i, cost in enumerate(stage_costs):
        is_bottleneck = " (瓶颈)" if np.isclose(cost, bottleneck_time) else ""
        print(f"阶段 {i} 成本: {cost:.2f} ms{is_bottleneck}")

    print(f"\n=> 最终流水线瓶颈时间: {bottleneck_time:.2f} ms")


def main():
    """
    主测试函数
    """
    # --- 1. 定义测试场景 ---
    # 您可以在这里修改这些参数来测试不同的场景
    MODEL_NAME = "VGG16"
    NUM_GPUS = 4

    # 假设一个静态、同构的环境
    # 算力为基准值 1.0
    COMPUTE_POWERS = [1.0] * NUM_GPUS
    # 链路带宽为 10 Gbps (约 1250 MB/s)
    BANDWIDTHS = [1250.0] * (NUM_GPUS - 1)

    print("--- 开始测试动态规划分区器 ---")
    print(f"模型: {MODEL_NAME}")
    print(f"GPU数量: {NUM_GPUS}")
    print("初始资源:")
    print(f"  - 算力系数: {COMPUTE_POWERS}")
    print(f"  - 链路带宽 (MB/s): {BANDWIDTHS}")

    # --- 2. 加载模型性能指标 ---
    try:
        model_profile = get_model_profile(MODEL_NAME)
        print(f"成功加载模型 profile，共 {len(model_profile['layers'])} 层。")
    except ValueError as e:
        print(f"错误: {e}")
        return

    # --- 3. 初始化并运行分区器 ---
    partitioner = DynamicProgrammingPartitioner(
        model_profile=model_profile,
        num_gpus=NUM_GPUS,
        compute_powers=COMPUTE_POWERS,
        bandwidths=BANDWIDTHS
    )

    optimal_partition = partitioner.partition()

    # --- 4. 打印结果 ---
    print_partition_results(optimal_partition, partitioner)


if __name__ == "__main__":
    main()
