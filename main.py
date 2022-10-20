import bot3RLNav
import gym
from gym.utils.env_checker import check_env

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = gym.make('bot3RLNav/World-v0', map_file="gray.jpg")
    check_env(env)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
