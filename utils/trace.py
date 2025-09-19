import numpy as np
import pandas as pd


class ResourceTrace:
    """
    生成或加载资源动态变化的轨迹。
    """

    def __init__(self, num_gpus, episode_length, csv_path=None):
        self.num_gpus = num_gpus
        self.episode_length = episode_length
        self.csv_path = csv_path

    def get_trace(self, trace_type="sin_wave"):
        """
        获取指定类型的资源轨迹。

        Args:
            trace_type (str): 轨迹类型, 支持 "static", "sin_wave", "random_walk", "csv".

        Returns:
            tuple: (compute_trace, bandwidth_trace)
                   compute_trace: shape (episode_length, num_gpus)
                   bandwidth_trace: shape (episode_length, num_gpus - 1)
        """
        print(f"--- 正在生成资源轨迹: {trace_type} ---")
        if trace_type == "static":
            return self._get_static_trace()
        elif trace_type == "sin_wave":
            return self._get_sin_wave_trace()
        elif trace_type == "random_walk":
            return self._get_random_walk_trace()
        elif trace_type == "csv":
            if self.csv_path is None:
                raise ValueError("使用 'csv' 轨迹类型时，必须提供 trace_path。")
            return self._get_csv_trace()
        else:
            raise ValueError(f"未知的轨迹类型: {trace_type}")

    def _get_static_trace(self):
        compute_trace = np.ones((self.episode_length, self.num_gpus))
        bandwidth_trace = np.full((self.episode_length, self.num_gpus - 1), 1250.0)  # 10 Gbps
        return compute_trace, bandwidth_trace

    def _get_sin_wave_trace(self):
        time_steps = np.arange(self.episode_length)
        compute_trace = np.zeros((self.episode_length, self.num_gpus))
        bandwidth_trace = np.zeros((self.episode_length, self.num_gpus - 1))

        for i in range(self.num_gpus):
            compute_trace[:, i] = 0.8 + 0.2 * np.sin(2 * np.pi * time_steps / 50 + i * np.pi / 4)

        for i in range(self.num_gpus - 1):
            bandwidth_trace[:, i] = 1250 * (0.7 + 0.3 * np.cos(2 * np.pi * time_steps / 80 + i * np.pi / 3))

        return compute_trace, bandwidth_trace

    def _get_random_walk_trace(self):
        compute_trace = np.ones((self.episode_length, self.num_gpus))
        bandwidth_trace = np.full((self.episode_length, self.num_gpus - 1), 1250.0)

        for t in range(1, self.episode_length):
            compute_trace[t, :] = compute_trace[t - 1, :] + np.random.normal(0, 0.05, self.num_gpus)
            bandwidth_trace[t, :] = bandwidth_trace[t - 1, :] + np.random.normal(0, 50, self.num_gpus - 1)

        # 限制在合理范围内
        compute_trace = np.clip(compute_trace, 0.5, 1.5)
        bandwidth_trace = np.clip(bandwidth_trace, 500, 2500)

        return compute_trace, bandwidth_trace

    def _get_csv_trace(self):
        """从CSV文件加载轨迹。"""
        try:
            print(f"--- 从文件加载资源轨迹: {self.csv_path} ---")
            df = pd.read_csv(self.csv_path)

            # 验证列是否存在
            compute_cols = [f'gpu_{i}_compute' for i in range(self.num_gpus)]
            bw_cols = [f'link_{i}_bw' for i in range(self.num_gpus - 1)]

            if not all(col in df.columns for col in compute_cols) or not all(col in df.columns for col in bw_cols):
                raise ValueError("CSV文件缺少必要的列 (例如, gpu_0_compute, link_0_bw)")

            # 截取或循环以匹配 episode_length
            num_rows = len(df)
            if num_rows < self.episode_length:
                repeats = int(np.ceil(self.episode_length / num_rows))
                df = pd.concat([df] * repeats, ignore_index=True)

            df = df.iloc[:self.episode_length]

            compute_trace = df[compute_cols].values
            bandwidth_trace = df[bw_cols].values

            return compute_trace, bandwidth_trace

        except FileNotFoundError:
            raise FileNotFoundError(f"轨迹文件未找到: {self.csv_path}")
        except Exception as e:
            raise RuntimeError(f"加载或解析轨迹文件时出错: {e}")

