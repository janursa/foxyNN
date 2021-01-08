import gym
import torch
from torch.distributions import Categorical
import numpy as np


import sys, os,pathlib
current_file_path = pathlib.Path(__file__).parent.absolute()
sys.path.insert(1,os.path.join(current_file_path,'..','..'))
from foxyNN.tools import t_env, t_agent

sys.path.insert(1,os.path.join(current_file_path,'..','..','..','CppyABM','build'))
from cppyabm.binds import Env, Agent, Patch, grid2
from policies import test_policy
@t_agent
class myAgent(Agent):
    def __init__(self,env_obj,class_name,ID):
        super().__init__(env_obj,class_name)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(10)
        self.policy = test_policy()
        self.ID = ID
    def act(self,obs):
        """
        Agent action for a given observation
        
        :param      obs:  The observation
        :type       obs:  { type_description }
        """
        probs,state_value = self.policy(obs)
        m = Categorical(probs)
        action = m.sample()
        assert self.action_space.contains(action.item())
        self.move(action.item())
        self.save_action(torch.log(probs[action]),state_value)
    def observe(self):
        obs = self.patch.index # location of agent on the patch
        return np.array(obs)
    def reset(self):
        pass
    def step(self):
        obs = self.observe()
        assert self.observation_space.contains(obs)
        self.act(obs)
        self.reward()
    def move(self,action):
        if self.ID == 1:
            if action == 1:
                dest_index = 0
            elif self.patch.index == 4:
                return
            else:
                dest_index = (self.patch.index+1)
        else:
            if action == 1:
                dest_index = 5
            elif self.patch.index == 9:
                return
            else:
                dest_index = (self.patch.index+1)

        # print('Current pos: {} dest: {}'.format(self.patch.index,dest_index))
        if dest_index != self.patch.index:
            self.env.place_agent(dest_index, self)
    def reward(self):
        if self.ID == 1:
            if self.patch.index == 0:
                reward = 2
            elif self.patch.index == 4:
                reward = 10
            else:
                reward = 0
        else:
            if self.patch.index == 5:
                reward = 2
            elif self.patch.index == 9:
                reward = 10
            else:
                reward = 0
        self.save_reward(reward)

settings = {
    'lr':0.03,
    'episodes':20,
    'policy_t':'AC' # type of policty, e.g. actor critic
}
@t_env(settings=settings)
class Domain(gym.Env,Env):
    def __init__(self):
        super().__init__()
        self._repo = []
        self.tick = 0
        mesh = grid2(length = 5*1,width = 2, mesh_length = 1)
        self.setup_domain(mesh)
        for [index,patch] in self.patches.items():
            if patch.index == 0:
                agent = self.generate_agent(1)
                self.place_agent(patch,agent)
            if patch.index == 5:
                agent = self.generate_agent(2)
                self.place_agent(patch,agent)
        self.update()

    def update(self):
        Env().update()
        # self.output()
    def generate_patch(self):
        patch = Patch(self)
        self._repo.append(patch)
        return patch
    def generate_agent(self,ID):
        agent = myAgent(self,'myAgent',ID)
        self.agents.append(agent)
        self._repo.append(agent)
        return agent
    def reset(self):
        self.tick = 0
        for agent in self.agents:
            if agent.ID == 1:
                self.place_agent(0,agent)
            else:
                self.place_agent(5,agent)
            agent.reset()
        
    def step(self):
        self.step_agents()
        if self.tick >= 6:
            done = True
        else:
            done = False
        return done,{}
    
    def episode(self):
        for ii in range(100): # run 5 times before optimizing the NN
            self.reset()
            done = False
            while not done:
                done,_ = self.step()
                if done:
                    break
                self.tick+=1
        
    def output(self):
        file = open('agents.csv','w')
        file.write('x,y,type,size\n')
        for [index,patch] in self.patches.items():
            x = patch.coords[0]
            y = patch.coords[1]
            if patch.empty:
                occupied = 10
            else:
                occupied = 0
            file.write("{},{},{},{}\n".format(x, y, occupied, 10))
        file.close()
      
