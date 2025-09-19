import numpy as np

# 数据来源: 基于公开论文和常见的 profiler 工具估算的典型值
# 单位:
# - forward_ms: 前向传播毫秒数
# - backward_ms: 反向传播毫秒数
# - activations_mb: 激活值大小 (MB)
# - params_mb: 参数大小 (MB)

# VGG-16 模型在典型 GPU 上的性能指标
VGG16_PROFILE = {
    'layers': [
        # Block 1
        {'name': 'conv1_1', 'forward_ms': 2.5, 'backward_ms': 3.8, 'activations_mb': 150.5, 'params_mb': 1.7},
        {'name': 'conv1_2', 'forward_ms': 4.1, 'backward_ms': 6.2, 'activations_mb': 150.5, 'params_mb': 36.9},
        {'name': 'pool1', 'forward_ms': 0.8, 'backward_ms': 1.1, 'activations_mb': 75.2, 'params_mb': 0},
        # Block 2
        {'name': 'conv2_1', 'forward_ms': 3.9, 'backward_ms': 5.9, 'activations_mb': 75.2, 'params_mb': 73.7},
        {'name': 'conv2_2', 'forward_ms': 7.8, 'backward_ms': 11.5, 'activations_mb': 75.2, 'params_mb': 147.5},
        {'name': 'pool2', 'forward_ms': 0.6, 'backward_ms': 0.9, 'activations_mb': 37.6, 'params_mb': 0},
        # Block 3
        {'name': 'conv3_1', 'forward_ms': 7.5, 'backward_ms': 11.2, 'activations_mb': 37.6, 'params_mb': 294.9},
        {'name': 'conv3_2', 'forward_ms': 14.9, 'backward_ms': 22.0, 'activations_mb': 37.6, 'params_mb': 589.8},
        {'name': 'conv3_3', 'forward_ms': 14.9, 'backward_ms': 22.0, 'activations_mb': 37.6, 'params_mb': 589.8},
        {'name': 'pool3', 'forward_ms': 0.4, 'backward_ms': 0.6, 'activations_mb': 18.8, 'params_mb': 0},
        # Block 4 & 5 (类似)
        {'name': 'conv4_1', 'forward_ms': 14.5, 'backward_ms': 21.5, 'activations_mb': 18.8, 'params_mb': 1179.6},
        {'name': 'conv4_2', 'forward_ms': 14.5, 'backward_ms': 21.5, 'activations_mb': 18.8, 'params_mb': 1179.6},
        {'name': 'conv4_3', 'forward_ms': 14.5, 'backward_ms': 21.5, 'activations_mb': 18.8, 'params_mb': 1179.6},
        {'name': 'pool4', 'forward_ms': 0.2, 'backward_ms': 0.4, 'activations_mb': 9.4, 'params_mb': 0},
        # FC Layers
        {'name': 'fc1', 'forward_ms': 25.0, 'backward_ms': 35.0, 'activations_mb': 4.0, 'params_mb': 102.7},
        {'name': 'fc2', 'forward_ms': 8.0, 'backward_ms': 12.0, 'activations_mb': 0.016, 'params_mb': 16.8},
        {'name': 'fc3', 'forward_ms': 4.0, 'backward_ms': 6.0, 'activations_mb': 0.004, 'params_mb': 4.1},
    ]
}

# ResNet-50 模型在典型 GPU 上的性能指标
RESNET50_PROFILE = {
    'layers': [
        {'name': 'conv1', 'forward_ms': 1.5, 'backward_ms': 2.2, 'activations_mb': 67.2, 'params_mb': 0.04},
        # ... (此处省略了ResNet50中间的所有残差块，以保持简洁)
        # 我们可以用一个聚合的块来代表它们
        {'name': 'res_block_1', 'forward_ms': 10.2, 'backward_ms': 15.3, 'activations_mb': 67.2, 'params_mb': 0.8},
        {'name': 'res_block_2', 'forward_ms': 15.8, 'backward_ms': 23.7, 'activations_mb': 33.6, 'params_mb': 3.1},
        {'name': 'res_block_3', 'forward_ms': 22.5, 'backward_ms': 33.8, 'activations_mb': 16.8, 'params_mb': 12.3},
        {'name': 'res_block_4', 'forward_ms': 30.1, 'backward_ms': 45.2, 'activations_mb': 8.4, 'params_mb': 49.2},
        {'name': 'avgpool', 'forward_ms': 0.5, 'backward_ms': 0.7, 'activations_mb': 0.008, 'params_mb': 0},
        {'name': 'fc', 'forward_ms': 2.1, 'backward_ms': 3.1, 'activations_mb': 0.004, 'params_mb': 8.2},
    ]
}


def get_model_profile(model_name="VGG16"):
    """
    获取指定模型的性能指标。
    """
    if model_name.upper() == "VGG16":
        return VGG16_PROFILE
    elif model_name.upper() == "RESNET50":
        return RESNET50_PROFILE
    else:
        raise ValueError(f"未知的模型名称: {model_name}")
