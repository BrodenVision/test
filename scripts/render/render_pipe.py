import sys
import os
import numpy as np
from pathlib import Path
import torch
import imageio

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import get_config
from envs.env_wrappers import DummyVecEnv
from envs.pipe_sim_env import PipeSimEnv


def make_render_env(all_args):
    """创建用于渲染的单个环境实例。"""

    def get_env_fn(rank):
        def init_env():
            env = PipeSimEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(0)])


def parse_args(args, parser):
    """解析渲染脚本特有的参数。"""
    parser.add_argument("--num_gpus", type=int, default=4, help="number of GPUs for pipeline parallelism")
    all_args = parser.parse_known_args(args)[0]
    all_args.num_agents = all_args.num_gpus
    all_args.env_name = "AutoPipe"
    all_args.scenario_name = f"{all_args.num_gpus}GPUs_Render"
    all_args.share_policy = True
    # 强制线程数为1，因为渲染只需要一个环境
    all_args.n_rollout_threads = 1

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.model_dir is None:
        print("错误：请使用 --model_dir 参数指定要加载的已训练模型的路径。")
        print("例如：--model_dir results/AutoPipe/4GPUs/mappo/4gpu_explore_v1/models")
        return

    # 强制在CPU上运行渲染
    device = torch.device("cpu")
    torch.set_num_threads(1)

    # 设置运行目录
    run_dir = Path(all_args.model_dir).parent

    # 初始化环境
    envs = make_render_env(all_args)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # 对于CTDE架构，我们总是使用 shared runner
    from runner.shared.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.render()

    # 关闭环境
    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
