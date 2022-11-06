import cv2

import bot3RLNav
import gym
from gym.utils.env_checker import check_env

if __name__ == '__main__':
    env = gym.make('bot3RLNav/DiscreteWorld-v2', map_file="data/map.jpg",
                   robot_file="data/robot.png")
    check_env(env)
    print(env.action_space.sample())
    test_simulate = False
    if test_simulate:
        cv2.namedWindow("bot3")
        img = env.render(mode="rgb_array")
        cv2.imshow("bot3", img)
        wait = 3000  # ms
        cv2.waitKey(wait)
        env.reset()
        img = env.render(mode="rgb_array")
        cv2.imshow("bot3", img)
        cv2.waitKey(wait)
        env.reset()
        img = env.render(mode="rgb_array")
        cv2.imshow("bot3", img)
        cv2.waitKey(wait)

