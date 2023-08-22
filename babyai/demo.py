import os
import sys
sys.path.append(os.environ['BabyAI'])
import gymnasium as gym
from babyai_utils import LanguageObsWrapper
# from minigrid.wrappers import FullyObsWrapper
import numpy as np
from dotmap import DotMap

seed = 2
np.random.seed(seed)
all_tasks = ['BlockedUnlockPickup', 'UnlockToUnlock', 'UnlockPickup']
selected_task = np.random.choice(all_tasks)
# agent_pov=False
env = gym.make(f"BabyAI-{selected_task}-v0", render_mode='human')
env = LanguageObsWrapper(env)
observation, _ = env.reset(seed=seed)
target_object = DotMap(color='red', type='key')
observation, reward, terminated, truncated, info = env.step((7, target_object))
# if terminated or truncated:
#   observation, info = env.reset()
print(observation['mission'])
print(observation['language'])
print('Done')
