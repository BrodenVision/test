import numpy as np


class Arbitrator:
    """
    一个轻量级、基于规则的仲裁器模块。
    接收所有Actor的动作输出，并做出最终的层迁移决策。
    """

    def __init__(self, args, partition, model_layers):
        self.args = args
        self.num_gpus = args.num_agents
        self.partition = partition
        self.model_layers = model_layers

    def decide(self, actions):
        """
        根据所有智能体的意愿和阈值做出决策。
        :param actions: 来自所有Actor的动作输出, shape: (num_gpus, 3)
                        actions[i] = [向左迁移意愿, 向右迁移意愿, 触发阈值]
        :return: 一个迁移决策 (from_gpu, to_gpu, layer_idx_global) or None
        """

        potential_migrations = []

        for i in range(self.num_gpus):
            my_action = actions[i]
            my_will_left, my_will_right, my_threshold = my_action

            # 检查向右迁移 (i -> i+1)
            if i < self.num_gpus - 1:
                neighbor_action = actions[i + 1]
                neighbor_will_left = neighbor_action[0]  # 邻居向左迁移的意愿，即接收意愿

                # 意愿匹配: 我想给，邻居也想收
                if my_will_right > my_threshold and neighbor_will_left > neighbor_action[2]:
                    if len(self.partition[i]) > 1:  # 确保至少保留一层
                        # 预估收益: 简单地用双方意愿之和作为代理
                        estimated_gain = my_will_right + neighbor_will_left
                        layer_to_move = self.partition[i][-1]  # 总是移动最右边的一层
                        potential_migrations.append((estimated_gain, i, i + 1, layer_to_move))

            # 检查向左迁移 (i -> i-1)
            if i > 0:
                neighbor_action = actions[i - 1]
                neighbor_will_right = neighbor_action[1]  # 邻居向右迁移的意愿，即接收意愿

                if my_will_left > my_threshold and neighbor_will_right > neighbor_action[2]:
                    if len(self.partition[i]) > 1:
                        estimated_gain = my_will_left + neighbor_will_right
                        layer_to_move = self.partition[i][0]  # 总是移动最左边的一层
                        potential_migrations.append((estimated_gain, i, i - 1, layer_to_move))

        # 如果没有潜在的迁移，则不采取任何行动
        if not potential_migrations:
            return None

        # 选择预估收益最高的迁移
        potential_migrations.sort(key=lambda x: x[0], reverse=True)
        best_migration = potential_migrations[0]

        _, from_gpu, to_gpu, layer_idx = best_migration

        return from_gpu, to_gpu, layer_idx
