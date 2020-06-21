import mesa
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import gym
import torch
import torch.nn.functional as F
from collections import namedtuple
from torch.distributions import Categorical
import numpy as np
from functools import partialmethod
import functools
import torch.optim as optim

from trainer import env_train, agent_train
from policies import MSC_Policy
class MSC(mesa.Agent):
    def __init__(self,unique_id, env_model,policy_model=None,agent_config=None):
        super().__init__(unique_id, env_model)
        self._alive = True
    @property
    def alive(self):
        return self._alive
    def set_state(self, state):
        self._alive = bool(state)
    set_alive = partialmethod(set_state, True)
    set_dead = partialmethod(set_state, False)




@agent_train
class MSC(mesa.Agent):
    def __init__(self,unique_id, env_model,policy_model=None,agent_config=None):
        super().__init__(unique_id, env_model)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0,4,shape=(1,))

        assert policy_model != None
        self.policy = policy_model
        self.type = 'MSC'
        self.data = {}

    def act(self,obs):
        # assert self.observation_space.contains(obs.item())
        probs,state_value=self.policy(obs)
        m = Categorical(probs)
        action = m.sample()
        assert self.action_space.contains(action.item())
        self.move(action.item())
        self.saved_actions.append((m.log_prob(action),state_value))
        # self.saved_actions.append(action)
    def observe(self):
        # collection the observation
        obs = self.pos[0] # only on x axis counts
        return obs
    def reset(self):
        self.data.clear()
    def step(self):
        obs = self.observe()
        obs = torch.from_numpy(np.array([obs])).float()
        self.act(obs)
        self.reward()
    def move(self,action):
        print("pos {} action {} ".format(self.pos,action))
        # neighbors = self.model.grid.get_neighborhood(self.pos,
        #     moore=True,
        #     include_center=False)
        # new_spot = neighbors[action]
        if action == 1:
            new_spot = (0,0)
        elif self.pos == (4,0):
            return
        else:
            new_spot = (self.pos[0]+1,0)
        self.model.grid.move_agent(self,new_spot)
    def reward(self):
        if self.pos[0] == 0:
            reward = 2
        elif self.pos[0] == 4:
            reward = 10
        else:
            reward = 0
        self.saved_rewards.append(reward)
    def disappear(self):
        self.model.grid.remove_agent(self)


train_config={'lr':0.03}
@env_train(train_config=train_config)
class ABM(gym.Env,mesa.Model):
    def __init__(self,env_configs=None,policy_models=None):
        assert env_configs != None
        self.env_configs = env_configs 
        self.policy_models = policy_models
        self.running  = True
        self.grid = SingleGrid(self.env_configs['width'], self.env_configs['height'], True)
        self.schedule = RandomActivation(self)
        self.agent_class_IDS = [item['ID'] for item in self.env_configs['agents']]
        self.reset()
        # self.datacollector = DataCollector(
        #     agent_reporters={"pos": "pos"})
    def reset(self):
        for agent in self.schedule.agents:
            agent.disappear()
        self.setup_agents(self.env_configs['agents'],self.policy_models)    
        self.data = {}
        self._counter = 0

    def setup_agents(self,agents_configs,policy_models):
        for info_set in agents_configs:
            agent_class = info_set['agent_class']
            ID = info_set['ID']
            agent_config = info_set['agent_config']
            agent_count = info_set['agent_count']
            for i in range(agent_count):
                if ID in policy_models.keys():
                    agent = agent_class(i,self,agent_config= agent_config,policy_model = policy_models[ID])
                else:
                    agent = agent_class(i,self,agent_config)
                self.schedule.add(agent)
                # coords = (self.random.randrange(self.grid.width),
                #     self.random.randrange(self.grid.height))
                coords = (0,0)
                self.grid.place_agent(agent, coords)
        print("Agents setup finished")

    def step(self):
        self.schedule.step()
        # self.collect_training_info()
        self._counter +=1 
        # self.datacollector.collect(self)
        
        done = False
        if self._counter >= 6:
            done = True
        return done,{}
    
    def run(self):
        
        done = False
        while not done:
            done,_ = self.step()
            # saved_actions_rewards.append((saved_actions,rewards))
            if done:
                break
        self.reset()
        print("Episode finished")
        
    
    
policy_models = {
    'MSC':MSC_Policy()
}
env_configs = {
    'agents':[
        {
            'ID':'MSC',
            'agent_class':MSC,
            'agent_config':None,
            'agent_count':1
        }
    ],
    'width': 5,
    'height': 1

}
abmObj = ABM( env_configs = env_configs, policy_models=policy_models)


for i in range(200):
    episode_reward = abmObj.run()
    print('i: {} reward {}'.format(i,episode_reward))

#     actions,probs_values = abmObj.collect_actions()
#     actions = actions['type1'][0]
#     _,reward,done,_ = abmObj.step(actions)
#     if done:
#         break
#         
