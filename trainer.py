import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F
from itertools import count
import argparse


def env_train(train_config=None):
    train_config = train_config
    def inner_func(env_class):
        setattr(env_class,"train_config",train_config)
        # env_class.train_config = train_config
        class Wrapper(env_class):

            train_config = env_class.train_config or {}
            eps = np.finfo(np.float32).eps.item()
            train_config.update({'lr':train_config.get('lr',3e-2)})
            train_config.update({'gamma':train_config.get('gamma',0.99)})
            optimizers = {}
            def reset_trainer(self):
                for agent in self.schedule.agents:
                    agent.reset()

            def run(self):
                super().run()

                saved_rewards = {}
                saved_actions = {}
                policies = {}
                episode_reward = {}
                
                ## get rewards, saved actions, and policies from agents
                for agent in self.schedule.agents:
                    if agent.__name__ not in saved_rewards.keys():
                        saved_rewards.update({agent.__name__:[*agent.saved_rewards]})
                    else:
                        saved_rewards[agent.__name__].append(*agent.saved_rewards)

                    if agent.__name__ not in saved_actions.keys():
                        saved_actions.update({agent.__name__:agent.saved_actions})
                    else:
                        saved_actions[agent.__name__].append(*agent.saved_actions) # unwrapped saved actions

                    if agent.__name__ not in policies.keys():
                        policies.update({agent.__name__:agent.policy})

                for agent_type in policies.keys():
                    agent_type_policy = policies[agent_type]
                    # if optimizer is not defined yet, define it
                    if agent_type not in self.optimizers.keys():
                        optimizer = optim.Adam(agent_type_policy.parameters(), lr=self.train_config['lr'])
                        self.optimizers.update({agent_type:optimizer})
                    # calculate loss
                    agent_type_actions = saved_actions[agent_type]
                    agent_type_rewards = saved_rewards[agent_type]
                    episode_reward.update({agent_type:np.sum(agent_type_rewards)})
                    loss = self.calculate_loss(rewards = agent_type_rewards,
                                               saved_actions = agent_type_actions)
                    # backpropagate
                    self.optimizers[agent_type].zero_grad()
                    loss.backward()
                    self.optimizers[agent_type].step()
                    # print(agent_type_policy.action_head.weight.grad)

                    # self.optimizer.zero_grad()
                    # loss.backward()
                    # self.optimizer.step()
                    # print(MSC_Policy_obj.action_head.weight.grad)
                self.reset_trainer()
                return episode_reward
            #     self.reset_trainer()
            def calculate_loss(self,rewards,saved_actions):
                """
                Training code. Calculates actor and critic loss and performs backprop.
                """
                R = 0
                saved_actions = saved_actions
                policy_losses = [] # list to save actor (policy) loss
                value_losses = [] # list to save critic (value) loss
                returns = [] # list to save the true values

                # calculate the true value using rewards returned from the environment
                for r in rewards[::-1]:
                    # calculate the discounted value
                    R = r + self.train_config['gamma'] * R
                    returns.insert(0, R)

                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + self.eps)

                for (log_prob, value), R in zip(saved_actions, returns):
                    advantage = R - value.item()

                    # calculate actor (policy) loss 
                    policy_losses.append(-log_prob * advantage)

                    # calculate critic (value) loss using L1 smooth loss
                    value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

                

                # sum up all the values of policy_losses and value_losses
                loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
                return loss
                # perform backprop
                

        return Wrapper
    return inner_func
  
def agent_train(agent_class):
    class Wrapper(agent_class):
        saved_actions = []
        saved_rewards = []
        __name__ = agent_class.__name__
        def reset(self):
            self.saved_actions.clear()
            self.saved_rewards.clear()
    return Wrapper



    