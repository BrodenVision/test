import sys
import os
import numpy as np
from pathlib import Path
import torch

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from config import get_config
from envs.env_wrappers import DummyVecEnv
from envs.pipe_sim_env import PipeSimEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = PipeSimEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = PipeSimEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    # AutoPipe 特有参数
    parser.add_argument("--num_gpus", type=int, default=4, help="number of GPUs for pipeline parallelism")
    parser.add_argument("--num_micro_batches", type=int, default=16, help="number of micro-batches in pipeline")
    parser.add_argument("--model_name", type=str, default="gpt2-small", help="name of the model to profile")
    parser.add_argument("--trace_type", type=str, default="sine", help="type of resource trace (sine, random, real)")
    parser.add_argument("--trace_path", type=str, default=None,
                        help="path to real trace data (if trace_type is 'real')")
    parser.add_argument("--initial_partition_strategy", type=str, default="uniform",
                        choices=["uniform", "dp"], help="strategy for initial partition")
    parser.add_argument("--reward_throughput_factor", type=float, default=1.0,
                        help="factor for throughput reward component")
    parser.add_argument("--migration_cost_factor", type=float, default=0.1,
                        help="factor for migration cost penalty")
    parser.add_argument("--staleness_penalty_factor", type=float, default=0.0,
                        help="factor for staleness penalty (set to 0 to disable)")

    # 由于环境现在是 AutoPipe，我们将 num_agents 与 num_gpus 关联
    # 覆盖一些默认参数以适应新环境
    all_args = parser.parse_known_args(args)[0]
    all_args.num_agents = all_args.num_gpus
    all_args.env_name = "AutoPipe"
    all_args.scenario_name = f"{all_args.num_gpus}GPUs"
    # AutoPipe 使用连续动作空间
    all_args.share_policy = True  # CTDE 必须共享策略 (对于Critic)

    # 设置连续动作空间
    all_args.use_continuous_action = True

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name not in ["rmappo", "mappo"]:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 路径设置
    run_dir = (
            Path("./results")
            / all_args.env_name
            / all_args.scenario_name
            / all_args.algorithm_name
            / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    # 对于 AutoPipe 的 CTDE 架构，我们总是使用 shared runner
    from runner.shared.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()


if __name__ == "__main__":
    # 移除 `train.py` 自身，只传递超参数
    main(sys.argv[1:])