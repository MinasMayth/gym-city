import torch
import torch.nn as nn
import torch.optim as optim

import sys
if sys.version[0] == '3':
    from .kfac import KFACOptimizer

class A2C_ACKTR():
    """
    A2C_ACKTR (Advantage Actor-Critic with ACKTR) algorithm implementation.

    Parameters:
        - actor_critic (nn.Module): The neural network model representing the actor-critic architecture.
        - value_loss_coef (float): Coefficient for the value loss in the total loss calculation.
        - entropy_coef (float): Coefficient for the entropy loss in the total loss calculation.
        - lr (float): Learning rate for the optimizer. If ACKTR is enabled, this parameter is ignored.
        - eps (float): Epsilon term for the RMSprop optimizer. Ignored if ACKTR is enabled.
        - alpha (float): Alpha term for the RMSprop optimizer. Ignored if ACKTR is enabled.
        - max_grad_norm (float): Maximum gradient norm for gradient clipping during optimization.
        - acktr (bool): Flag to indicate whether to use the ACKTR optimizer. If True, KFACOptimizer is used.

    Attributes:
        - actor_critic (nn.Module): The actor-critic neural network model.
        - acktr (bool): Flag indicating whether ACKTR optimizer is used.
        - value_loss_coef (float): Coefficient for the value loss in the total loss calculation.
        - entropy_coef (float): Coefficient for the entropy loss in the total loss calculation.
        - max_grad_norm (float): Maximum gradient norm for gradient clipping during optimization.
        - optimizer (optim.Optimizer): The optimizer used for parameter updates, either RMSprop or KFACOptimizer.

    Methods:
        - update(rollouts): Performs a single update step using the provided rollouts.
    """
    def __init__(self, actor_critic, value_loss_coef, entropy_coef, lr=None, eps=None, alpha=None, max_grad_norm=None, acktr=False):
        self.actor_critic = actor_critic
        self.acktr = acktr
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        """
        Performs a single update step using the provided rollouts.

        Parameters:
            - rollouts: Rollout data containing observations, actions, rewards, etc.

        Returns:
            - Tuple of updated loss values (value_loss, action_loss, dist_entropy).
        """

        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()
