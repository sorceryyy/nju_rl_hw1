import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init


import torch.nn as nn
import torch.nn.init as init
import numpy as np

# class CnnPolicy(nn.Module):
#     def __init__(self, input_size, output_size, use_noisy_net=False):
#         super(CnnPolicy, self).__init__()

#         # Select linear or NoisyLinear based on use_noisy_net
#         linear = NoisyLinear if use_noisy_net else nn.Linear

#         # Feature extraction: Adding Dropout and Pooling after the second and third CNN layers
#         self.feature = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
#             nn.ReLU(),
            
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             # nn.Dropout(0.2),  # Add Dropout after the second convolution layer
#             # nn.MaxPool2d(kernel_size=2, stride=2),  # Add Pooling after the second convolution layer
            
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
#             nn.ReLU(),
#             # nn.Dropout(0.3),  # Add Dropout after the third convolution layer
#             # nn.MaxPool2d(kernel_size=2, stride=2),  # Add Pooling after the third convolution layer

#             nn.Flatten(),  # Flatten the output before fully connected layers
            
#             nn.Linear(7 * 7 *64, 256),  # Fully connected layer
#             nn.ReLU(),
#             nn.Linear(256, 128),  # Reduced size of fully connected layers
#             nn.ReLU()
#         )

#         # Actor network: Simplified and reduced layer size
#         self.actor = nn.Sequential(
#             # nn.Linear(128, 128),  # Match with the simplified size
#             # nn.ReLU(),
#             nn.Linear(128, output_size)  # Final output layer
#         )       
        
#         # Feature extraction: simplified version
#         # self.feature = nn.Sequential(
#         #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
#         #     nn.ReLU(),
#         #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.2),
#         #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
#         #     nn.ReLU(),
#         #     nn.Flatten(),
#         #     linear(7 * 7 * 64, 256),  # Reduced size of fully connected layers
#         #     nn.ReLU(),
#         #     linear(256, 128),  # Reduced the output size from 448 to 128 for simplification
#         #     nn.ReLU()
#         # )

#         # # Actor network: simplified and reduced layer size
#         # self.actor = nn.Sequential(
#         #     linear(128, 128),  # Match with the simplified size
#         #     nn.ReLU(),
#         #     linear(128, output_size)
#         # )

#         # Initialize weights
#         self._initialize_weights()

#     def _initialize_weights(self):
#         """Apply orthogonal initialization to weights and zero initialization to biases."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 init.orthogonal_(m.weight, np.sqrt(2))
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#     def forward(self, state):
#         """Forward pass of the network."""
#         x = self.feature(state)
#         policy = self.actor(x)
#         return policy


class CnnPolicy(nn.Module):
    def __init__(self, input_size, output_size, use_noisy_net=False):
        super(CnnPolicy, self).__init__()

        if use_noisy_net:
            print('use NoisyNet')
            linear = NoisyLinear
        else:
            linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            Flatten(),
            linear(
                7 * 7 * 64,
                256),
            nn.ReLU(),
            linear(
                256,
                448),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            linear(448, 448),
            nn.ReLU(),
            linear(448, output_size)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
        
        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        return policy


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet"""

    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noisy_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        self.noise_std = sigma0 / math.sqrt(self.in_features)

        self.reset_parameters()
        self.register_noise()

    def register_noise(self):
        in_noise = torch.FloatTensor(self.in_features)
        out_noise = torch.FloatTensor(self.out_features)
        noise = torch.FloatTensor(self.out_features, self.in_features)
        self.register_buffer('in_noise', in_noise)
        self.register_buffer('out_noise', out_noise)
        self.register_buffer('noise', noise)

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(
            self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Note: noise will be updated if x is not volatile
        """
        normal_y = nn.functional.linear(x, self.weight, self.bias)
        if self.training:
            # update the noise once per update
            self.sample_noise()

        noisy_weight = self.noisy_weight * self.noise
        noisy_bias = self.noisy_bias * self.out_noise
        noisy_y = nn.functional.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


class CnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, use_noisy_net=False):
        super(CnnActorCriticNetwork, self).__init__()

        if use_noisy_net:
            print('use NoisyNet')
            linear = NoisyLinear
        else:
            linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            Flatten(),
            linear(
                7 * 7 * 64,
                256),
            nn.ReLU(),
            linear(
                256,
                448),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            linear(448, 448),
            nn.ReLU(),
            linear(448, output_size)
        )

        self.extra_layer = nn.Sequential(
            linear(448, 448),
            nn.ReLU()
        )

        self.critic_ext = linear(448, 1)
        self.critic_int = linear(448, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        init.orthogonal_(self.critic_ext.weight, 0.01)
        self.critic_ext.bias.data.zero_()

        init.orthogonal_(self.critic_int.weight, 0.01)
        self.critic_int.bias.data.zero_()

        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        for i in range(len(self.extra_layer)):
            if type(self.extra_layer[i]) == nn.Linear:
                init.orthogonal_(self.extra_layer[i].weight, 0.1)
                self.extra_layer[i].bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value_ext = self.critic_ext(self.extra_layer(x) + x)
        value_int = self.critic_int(self.extra_layer(x) + x)
        return policy, value_ext, value_int


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        feature_output = 7 * 7 * 64
        self.predictor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.target = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature
