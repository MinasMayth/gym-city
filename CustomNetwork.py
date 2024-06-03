
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Tuple
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)

class CustomNetwork(nn.Module):
    def __init__(self, map_w, map_h, feature_dim: int, action_space: spaces.Space):
        super().__init__()

        # Convolutional layers
        #self.conv1 = nn.Conv2d(in_channels=feature_dim, out_channels=32, kernel_size=5)
        #self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        #self.relu = nn.ReLU()

        # 1x1 Convolution for action distribution
        self.action_net = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=action_space.n, kernel_size=1),
            nn.Flatten(),
            nn.Linear(in_features=action_space.n * 18 * 18, out_features=576))

        # Dimensions after convolution layers
        self.conv_output_dim = 10368 # Update this according to your input size after conv layers

        # Dense layers for value prediction
        self.fc1 = nn.Linear(self.conv_output_dim, 256)  # Assuming input feature size after conv layers is 6400
        self.tanh = nn.Tanh()
        self.feature_dim = feature_dim

        # Latent dimensions for policy and value networks
        self.latent_dim_pi = action_space.n
        self.latent_dim_vf = 256

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        #x = self.relu(self.conv1(features))
        #x = self.relu(self.conv2(x))
        action_distribution = self.action_net(features)
        # action_distribution = th.flatten(action_distribution)

        # Flattening for the dense layer
        x = th.flatten(features, start_dim=1)
        value = self.tanh(self.fc1(x))

        #value = self.fc2(value)
        return action_distribution, value


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        self.action_space = action_space
        self.observation_space = observation_space


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.observation_space.shape[1], self.observation_space.shape[2],
                                           self.features_dim, self.action_space)
        self.latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.latent_dim_vf = self.mlp_extractor.latent_dim_vf
