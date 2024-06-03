import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Tuple
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def get_conv_output_shape(h_w, conv_layer):
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation

    h = ((h_w[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
    w = ((h_w[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1
    return h, w


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        print(observation_space.shape)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        with th.no_grad():
            n_flatten = self._get_conv_output_size(observation_space.shape) / 2
            self.n_flatten = int(n_flatten)
    def _get_conv_output_size(self, shape):
        o = shape[1:]  # Only height and width
        for layer in self.cnn:
            if isinstance(layer, nn.Conv2d):
                o = get_conv_output_shape(o, layer)
        return o[0] * o[1] * 64  # Number of channels in the last conv layer

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)


class CustomNetwork(nn.Module):
    def __init__(self, map_w, map_h, conv_output_dim, feature_dim: int, action_space: spaces.Space):
        super().__init__()

        # 1x1 Convolution for action distribution
        self.action_net = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=action_space.n, kernel_size=1),
            nn.Flatten(),
            nn.Linear(in_features=1 * action_space.n * (map_w - 6) * (map_h - 6), out_features=action_space.n))

        # print(1*action_space.n*(map_w-6)*(map_h-6))
        # quit()
        # Dimensions after convolution layers
        self.conv_output_dim = conv_output_dim  # Update this according to your input size after conv layers
        # Dense layers for value prediction
        self.fc1 = nn.Linear(self.conv_output_dim, 256)  # Assuming input feature size after conv layers is 6400
        self.tanh = nn.Tanh()
        self.feature_dim = feature_dim

        # Latent dimensions for policy and value networks
        self.latent_dim_pi = action_space.n
        self.latent_dim_vf = 256

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        # x = self.relu(self.conv1(features))
        # x = self.relu(self.conv2(x))
        action_distribution = self.action_net(features)
        # action_distribution = th.flatten(action_distribution)

        # Flattening for the dense layer
        x = th.flatten(features, start_dim=1)
        value = self.tanh(self.fc1(x))

        # value = self.fc2(value)
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
        self.cnn_extractor = CustomCNN(self.observation_space)
        self.mlp_extractor = CustomNetwork(self.observation_space.shape[1], self.observation_space.shape[2],
                                           self.cnn_extractor.n_flatten, self.features_dim, self.action_space)
        self.latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.latent_dim_vf = self.mlp_extractor.latent_dim_vf
