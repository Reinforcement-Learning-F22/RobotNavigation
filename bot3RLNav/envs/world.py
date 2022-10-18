import gym
from gym import spaces
import numpy as np


########################################################################
class World(gym.Env):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self):
        """"""
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(0, 10, shape=(1,), dtype=int),
                "y": spaces.Box(0, 10, shape=(1,), dtype=int),
                "theta": spaces.Box(-np.pi, np.pi, shape=(1,), dtype=int),
                "target": spaces.Box(0, 10, shape=(2,), dtype=float),
             }
        )
        self.action_space = spaces.Dict(
            {
                "v": spaces.Box(0, 10, shape=(1,), dtype=float),
                "w": spaces.Box(0, 10, shape=(1,), dtype=float)
            }
        )
