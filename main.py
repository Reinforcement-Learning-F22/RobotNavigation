import cv2

import bot3RLNav
import gym
from gym.utils.env_checker import check_env


# ----------------------------------------------------------------------
def check():
    """"""
    env = gym.make('bot3RLNav/World-v3', map_file="data/map01.jpg",
                   robot_file="data/robot.png")
    check_env(env)
    print(env.action_space.sample())
    test_simulate = True
    if test_simulate:
        cv2.namedWindow("bot3")
        wait = 500  # ms
        for i in range(10):
            obs = env.reset(options=dict(reset=True))
            img = env.render(mode="rgb_array")
            cv2.imshow("bot3", img)
            cv2.waitKey(wait)


# ----------------------------------------------------------------------
def train():
    """"""
    from stable_baselines3 import DQN
    env = gym.make('bot3RLNav/DiscreteWorld-v2', map_file="data/map01.jpg",
                   robot_file="data/robot.png")
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)

    obs = env.reset()
    name = "bot3"
    cv2.namedWindow(name)
    rate = 100  # frame rate in ms
    count = 500
    while count > 0:
        frame = env.render(mode="rgb_array")
        cv2.imshow("bot3", frame)
        cv2.waitKey(rate)
        action, _states = model.predict(obs, deterministic=True)
        # action = 5
        obs, reward, done, info = env.step(action)
        # print(obs, reward)
        if done:
            frame = env.render(mode="rgb_array")
            cv2.imshow("bot3", frame)
            cv2.waitKey(rate)
            break
        count -= 1
    env.close()


if __name__ == '__main__':
    check()
    # train()
