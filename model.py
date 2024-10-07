import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnPolicy(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnPolicy, self).__init__()

        # 提取图像输入的通道数，宽度和高度
        channels, height, width = input_shape

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # 计算卷积层的输出维度
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 8, 4), 4, 2), 3, 1)
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 8, 4), 4, 2), 3, 1)
        linear_input_size = conv_w * conv_h * 64
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 将特征展平
        x = F.relu(self.fc1(x))
        return self.fc2(x)