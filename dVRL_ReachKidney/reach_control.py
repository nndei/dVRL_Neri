#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import dVRL_simulator
import numpy as np
import time


# In[2]:


env_reach = gym.make("ReachRailKidney-v1")

# In[3]:

for _ in range(30):
	s = env_reach.reset()
	env_reach.render()
	#new = np.array([-90,-20,0])

	for _ in range(100):
		a = np.clip(10*(s['desired_goal'] - s['observation'][:3]), -1, 1)
   		#a = np.append(np.clip(10*(s['desired_goal'] - s['observation'][-3:]), -1, 1), new)
		#s, r, _, info = env_reach.step(a, new)
		s, r, _, info = env_reach.step(a)
		#time.sleep(0.1)
	print(info)

# In[4]:


#env_reach.close()
