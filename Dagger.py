import numpy as np
import torch
import torch.nn as nn

from abc import abstractmethod
from tqdm import tqdm

from model import CnnPolicy

from RLA import logger, time_step_holder, exp_manager


class DaggerAgent:
	def __init__(self,):
		pass

	@abstractmethod
	def select_action(self, ob):
		pass


# here is an example of creating your own Agent
class MyAgent(DaggerAgent):
	def __init__(self, env, args, device):
		super(DaggerAgent, self).__init__()
		self.device = device
		# init your model
		obs_shape = env.observation_space.shape  # 输入的图像大小 (通道数, 高度, 宽度)
		n_actions = env.action_space.n  # 动作数量
		self.model = CnnPolicy(input_size=obs_shape, output_size=n_actions)

		# 使用 Adam 作为优化器
		
		self.model.to(device=device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        
        # 使用交叉熵作为损失函数（适用于分类任务）
		self.loss_fn = nn.CrossEntropyLoss()

		self.update_num = 0

	def train(self, data_loader):
		total_loss = 0
		device = self.device
		self.model.train()

		for data, label in tqdm(data_loader):
			data = data.to(device)
			label = label.to(device)
			output = self.model(data)
			loss = self.loss_fn(output, label)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			total_loss += loss.item()

			# for log
			self.update_num += 1
			logger.logkv("train_loss", loss.item())
			time_step_holder.set_time(self.update_num)
			logger.dumpkvs()

		return total_loss / len(data_loader)

	def evaluate(self, data_loader):
		total_loss = 0
		correct = 0
		total = 0
		device = self.device
		self.model.eval()

		with torch.no_grad():
			for data, label in tqdm(data_loader):
				data = data.to(device)
				label = label.to(device)
				output = self.model(data)
				loss = self.loss_fn(output, label)
				total_loss += loss.item()
				predicted = output.argmax(dim=1)  # Assuming multi-class classification
				correct += (predicted == label).sum().item()
				total += label.size(0)

		accuracy = correct / total
		return total_loss / len(data_loader), accuracy


	def predict(self, data_batch):
        # 将数据转化为 PyTorch Tensor
		data_batch = torch.tensor(data_batch, dtype=torch.float32).to(self.device)
        
        # 切换到评估模式
		self.model.eval()

        # 关闭梯度计算
		with torch.no_grad():
            # 前向传播：通过模型预测动作
			data_batch = data_batch.unsqueeze(0) if data_batch.dim() == 3 else data_batch
			output = self.model(data_batch)
			action = torch.argmax(output, dim=1).item()
        
		return action	# train your model with labeled data
	
	def get_checkpoint(self):
		related_variable ={
			"optimizer": self.optimizer,
			"update_num": self.update_num,
		}
		return self.model.state_dict(), related_variable




