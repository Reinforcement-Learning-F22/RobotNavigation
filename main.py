import bot3RLNav
import gym
from gym.utils.env_checker import check_env

if __name__ == '__main__':
    env = gym.make('bot3RLNav/DiscreteWorld-v0', map_file="data/gray.jpg")
    check_env(env)
