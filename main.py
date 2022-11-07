import cv2

import bot3RLNav
import gym
from gym.utils.env_checker import check_env

if __name__ == '__main__':
    env = gym.make('bot3RLNav/World-v2', map_file="data/map01.jpg",
                   robot_file="data/robot.png")
    check_env(env)
    print(env.action_space.sample())
    test_simulate = True
    if test_simulate:
        cv2.namedWindow("bot3")
        wait = 1000  # ms
        for i in range(5):
            img = env.render(mode="rgb_array")
            cv2.imshow("bot3", img)
            cv2.waitKey(wait)
            env.reset()
