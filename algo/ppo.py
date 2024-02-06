import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PPO():
    """
    Proximal Policy Optimization (PPO) implementation for training actor-critic models.

    Parameters:
        - actor_critic (nn.Module): The neural network model representing the actor-critic architecture.
        - clip_param (float): Clipping parameter for PPO objective to prevent large policy updates.
        - ppo_epoch (int): Number of optimization epochs to update the policy.
        - num_mini_batch (int): Number of mini-batches to be sampled from the collected data for each update.
        - value_loss_coef (float): Coefficient for the value loss in the total loss calculation.
        - entropy_coef (float): Coefficient for the entropy loss in the total loss calculation.
        - lr (float): Learning rate for the Adam optimizer.
        - eps (float): Epsilon term for the Adam optimizer.
        - max_grad_norm (float): Maximum gradient norm for gradient clipping during optimization.
        - use_clipped_value_loss (bool): Flag to enable clipped value loss in the total loss calculation.

    Attributes:
        - actor_critic (nn.Module): The actor-critic neural network model.
        - clip_param (float): Clipping parameter for PPO objective.
        - ppo_epoch (int): Number of optimization epochs for updating the policy.
        - num_mini_batch (int): Number of mini-batches for sampling data in each update.
        - value_loss_coef (float): Coefficient for the value loss in the total loss calculation.
        - entropy_coef (float): Coefficient for the entropy loss in the total loss calculation.
        - max_grad_norm (float): Maximum gradient norm for gradient clipping during optimization.
        - use_clipped_value_loss (bool): Flag indicating whether to use clipped value loss.
        - optimizer: The Adam optimizer used for parameter updates.

    Methods:
        - update(rollouts): Performs a single update step using the provided rollouts.

    """
    def __init__(self, actor_critic, clip_param, ppo_epoch, num_mini_batch, value_loss_coef, entropy_coef, lr=None,
                 eps=None, max_grad_norm=None, use_clipped_value_loss=False):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        """
        Performs a single update step using the provided rollouts.

        Parameters:
            - rollouts: Rollout data containing observations, actions, rewards, etc.

        Returns:
            - Tuple of updated loss values (value_loss, action_loss, dist_entropy).
        """
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch,
                    masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = .5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(return_batch, values)

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
