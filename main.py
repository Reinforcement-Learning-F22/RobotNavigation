import bot3RLNav
import gym
from gym.utils.env_checker import check_env

if __name__ == '__main__':
    env = gym.make('bot3RLNav/World-v1', map_file="data/map.jpg")
    check_env(env)
