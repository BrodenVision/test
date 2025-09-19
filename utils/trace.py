import numpy as np
import os


class ResourceTrace:
    """
    管理和生成资源动态变化轨迹。
    """

    def __init__(self, num_gpus, episode_length, trace_path=None):
        self.num_gpus = num_gpus
        self.episode_length = episode_length
        self.trace_path = trace_path

    def get_trace(self, trace_type="sin_wave"):
        """
        根据指定的类型生成轨迹。

        Args:
            trace_type (str): 轨迹类型, "static", "sin_wave", "random_walk", 或 "csv"。

        Returns:
            tuple: (compute_trace, bandwidth_trace)
        """
        if trace_type == "static":
            return self._get_static_trace()
        elif trace_type == "sin_wave":
            return self._get_sin_wave_trace()
        elif trace_type == "random_walk":
            return self._get_random_walk_trace()
        elif trace_type == "csv":
            return self._load_from_csv()
        else:
            raise ValueError(f"未知的轨迹类型: {trace_type}")

    def _get_static_trace(self):
        compute_trace = np.ones((self.episode_length, self.num_gpus))
        bandwidth_trace = np.ones((self.episode_length, self.num_gpus - 1)) * 1250  # 1250 MB/s (10G)
        return compute_trace, bandwidth_trace

    def _get_sin_wave_trace(self):
        """生成周期性变化的正弦波轨迹。"""
        compute_trace = np.zeros((self.episode_length, self.num_gpus))
        bandwidth_trace = np.zeros((self.episode_length, self.num_gpus - 1))

        time_steps = np.arange(self.episode_length)

        for i in range(self.num_gpus):
            compute_trace[:, i] = 1.0 + 0.5 * np.sin(2 * np.pi * time_steps / 50 + i * np.pi / 4)

        for i in range(self.num_gpus - 1):
            bandwidth_trace[:, i] = 1250 + 750 * np.cos(2 * np.pi * time_steps / 80 + i * np.pi / 3)

        return np.clip(compute_trace, 0.2, 2.0), np.clip(bandwidth_trace, 200, 5000)

    def _get_random_walk_trace(self):
        """生成随机游走轨迹，模拟更不可预测的变化。"""
        compute_trace = np.ones((self.episode_length, self.num_gpus))
        bandwidth_trace = np.ones((self.episode_length, self.num_gpus - 1)) * 1250

        for t in range(1, self.episode_length):
            compute_noise = np.random.normal(0, 0.05, self.num_gpus)
            compute_trace[t] = np.clip(compute_trace[t - 1] + compute_noise, 0.2, 2.0)

            bw_noise = np.random.normal(0, 50, self.num_gpus - 1)
            bandwidth_trace[t] = np.clip(bandwidth_trace[t - 1] + bw_noise, 200, 5000)

        return compute_trace, bandwidth_trace

    def _load_from_csv(self):
        """从 CSV 文件加载轨迹。"""
        if self.trace_path is None or not os.path.exists(self.trace_path):
            raise FileNotFoundError(f"轨迹文件未找到或未指定路径: {self.trace_path}")

        print(f"--- 从文件加载资源轨迹: {self.trace_path} ---")

        try:
            data = np.loadtxt(self.trace_path, delimiter=',', skiprows=1)
        except Exception as e:
            raise IOError(f"无法读取或解析轨迹文件 {self.trace_path}: {e}")

        # 校验列数
        expected_cols = self.num_gpus + (self.num_gpus - 1)
        if data.shape[1] != expected_cols:
            raise ValueError(f"轨迹文件列数错误。期望 {expected_cols} 列，但文件中有 {data.shape[1]} 列。")

        compute_trace_full = data[:, :self.num_gpus]
        bandwidth_trace_full = data[:, self.num_gpus:]

        # 如果轨迹文件比 episode_length 短，则循环使用
        num_rows = data.shape[0]
        if num_rows < self.episode_length:
            repeats = int(np.ceil(self.episode_length / num_rows))
            compute_trace = np.tile(compute_trace_full, (repeats, 1))[:self.episode_length]
            bandwidth_trace = np.tile(bandwidth_trace_full, (repeats, 1))[:self.episode_length]
        else:
            compute_trace = compute_trace_full[:self.episode_length]
            bandwidth_trace = bandwidth_trace_full[:self.episode_length]

        return compute_trace, bandwidth_trace
