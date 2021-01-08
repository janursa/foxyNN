import numpy as np
import torch
import torch.nn.functional as F
def AC_lossfunc(rewards,actions,gamma=None):
    """
    Calculates loss based on the actor and critic algorithm.
    """
    assert gamma != None
    R = 0
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value. The rewards in later time is more important than the initial ones 
    for r in rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)  
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())
    for (log_prob, value), R in zip(actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss 
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    return loss