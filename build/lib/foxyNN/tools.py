import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F
from .lossfuncs import AC_lossfunc

def t_env(settings=None):
    """
    Wrapper function for env class. It receives a configuration variable from user to set the
    training settings such as learning rate and policy type. 
    
    :param      settings:  Settings of the training process
    :type       settings:  Dict
    
    :returns:   Wrapped env class. 
    :rtype:     Trainer_env
    """
    def inner_wrapper(env_class):
        """
        True wrapper of the env_class. 
        """
        setattr(env_class,"t_settings",settings) # add the given settings as a new variable to env_class
        env_class.t_settings.update({'lr':env_class.t_settings.get('lr',3e-2)}) # add default lr if it's not given
        env_class.t_settings.update({'gamma':env_class.t_settings.get('gamma',0.99)}) # add default gamma if it's not given
        class Trainer_env(env_class):
            """
            Trainer class for Env class of an agent-based model. It derives the Env class and adds 
            tools for training . 

            """
            def __init__(self,*args,**kargs):
                super().__init__(*args,**kargs)
                self.optimizers = {} # optimizers based on policy types
                self.policies = {}
            def initialize(self):
                """
                Initialize the trainer by collecting the policies from agents and definiting 
                otimizer for each policy.
                """
                for agent in self.agents: # collect the policies based on agent type
                    if agent.__name__ not in self.policies.keys():
                        self.policies.update({agent.__name__:agent.policy})  
                
                for agent_type,policy in self.policies.items(): # define optimizers based on agent type
                    if agent_type not in self.optimizers.keys():
                        optimizer = optim.Adam(policy.parameters(), lr=self.t_settings['lr'])
                        self.optimizers.update({agent_type:optimizer}) 
            def train(self):
                """
                Initialize the model and trains the policies by running the episodes for the given number of types, i.e. t_settings['episodes'].
                At each run, the policies are optimized.

                :returns:   Trained policies. 
                :rtype:     Dict.

                """
                self.initialize() ## initialize the mode
                for i in range(self.t_settings['episodes']):
                    episode_reward = self.episode()
                    print('i: {} reward {}'.format(i,episode_reward))

                return self.policies # trained policies
            def episode(self):
                """
                Runs the simulation by sending a call to episode function of Env class.
                Then, collects rewards and actions from agents, calculates loss, and optimize the policies.
                
                :returns:   rewards,trained policies for each episode 
                :rtype:     float,dict.
                """
                super().episode()
                saved_rewards = {}
                saved_actions = {}
                episode_reward = {}
                
                
                for agent in self.agents: # get rewards and saved actions from agents
                    if agent.__name__ not in saved_rewards.keys():
                        saved_rewards.update({agent.__name__:[*agent.saved_rewards]})
                    else:
                        saved_rewards[agent.__name__].append(*agent.saved_rewards)

                    if agent.__name__ not in saved_actions.keys():
                        saved_actions.update({agent.__name__:agent.saved_actions})
                    else:
                        saved_actions[agent.__name__].append(*agent.saved_actions) # unwrapped saved actions

                
                for agent_type, optimizer in self.optimizers.items(): # go through each policy and optimize it
                    agent_type_actions = saved_actions[agent_type]
                    agent_type_rewards = saved_rewards[agent_type]
                    episode_reward.update({agent_type:np.sum(agent_type_rewards)}) #sum the rewards
                    # calculate loss
                    loss = self.calculate_loss(rewards = agent_type_rewards,
                                               saved_actions = agent_type_actions)
                    # backpropagate
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                self.reset_t_env() # reset training environement
                return episode_reward
            def reset_t_env(self):
                """
                Resets the training environement by sending a reset call to individual agents.
                """
                for agent in self.agents:
                    agent.reset_t_agent()
            def calculate_loss(self,rewards,saved_actions):
                """
                Calculate loss by sending a call to the designated loss functions.
                """
                if self.t_settings['policy_t'] == 'AC':
                    return AC_lossfunc(rewards,saved_actions,self.t_settings['gamma'])
                else:
                    raise ValueError("Loss calculator is not defined for '{}'".format(self.t_settings['policy_t']))
                
        return Trainer_env
    return inner_wrapper
  
def t_agent(agent_class):
    """
    Wrapper function for agent class. 
    
    :param      agent_class:  The agent class
    :type       agent_class:  Agent
    
    :returns:   wrapped agent
    """
    class Trainer_agent(agent_class):
        """
        Trainer class for Agent class of an agent-based model. It derives the agent class and adds 
        tools for training purposes. 

        """
        saved_actions = []
        saved_rewards = []
        __name__ = agent_class.__name__
        def reset_t_agent(self):
            """
            Resets trainer agent 
            """
            self.saved_actions.clear()
            self.saved_rewards.clear()
        def save_action(self,prob_action,state_value):
            """
            Saves actions and state values
            """
            self.saved_actions.append((prob_action,state_value)) 
        def save_reward(self,reward):
            """
            Saves rewards values
            """
            self.saved_rewards.append(reward)
    return Trainer_agent



    