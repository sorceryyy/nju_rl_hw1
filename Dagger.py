import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from abc import abstractmethod

from model import CnnPolicy



class DaggerAgent:
	def __init__(self,):
		pass

	@abstractmethod
	def select_action(self, ob):
		pass


# here is an example of creating your own Agent
class MyAgent(DaggerAgent):
	def __init__(self, env, args):
		super(DaggerAgent, self).__init__()
		# init your model
		obs_shape = env.observation_space.shape  # 输入的图像大小 (通道数, 高度, 宽度)
		n_actions = env.action_space.n  # 动作数量
		self.model = CnnPolicy(input_shape=obs_shape, num_actions=n_actions)

		# 使用 Adam 作为优化器
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        
        # 使用交叉熵作为损失函数（适用于分类任务）
		self.loss_fn = nn.CrossEntropyLoss()

	def update(self, data_batch, label_batch, batch_size=32):
        # 将数据和标签转化为 PyTorch Tensor
		data_batch = torch.tensor(data_batch, dtype=torch.float32)
		label_batch = torch.tensor(label_batch, dtype=torch.long)

        # 创建一个 DataLoader 对象，用于 mini-batch 采样
		dataset = TensorDataset(data_batch, label_batch)
		data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 切换模型到训练模式
		self.model.train()

		total_loss = 0

        # 遍历数据加载器，分批训练模型
		for data, label in data_loader:
            # 清除梯度
			self.optimizer.zero_grad()

            # 前向传播：通过模型预测输出
			output = self.model(data)

            # 计算损失：预测值与标签之间的差异
			loss = self.loss_fn(output, label)

            # 反向传播：计算梯度
			loss.backward()

            # 优化：更新模型参数
			self.optimizer.step()

			total_loss += loss.item()

        # 返回总的平均损失值以便跟踪
		return total_loss / len(data_loader)

    # 根据模型的输出选择动作
	def select_action(self, data_batch):
        # 将数据转化为 PyTorch Tensor
		data_batch = torch.tensor(data_batch, dtype=torch.float32)
        
        # 切换到评估模式
		self.model.eval()

        # 关闭梯度计算
		with torch.no_grad():
            # 前向传播：通过模型预测动作
			output = self.model(data_batch)
			action = torch.argmax(output, dim=1).item()
        
		return action	# train your model with labeled data




