import cv2

import bot3RLNav
import gym
from gym.utils.env_checker import check_env


# ----------------------------------------------------------------------
def check():
    """"""
    env = gym.make('bot3RLNav/DiscreteWorld-v5', map_file="data/map01.jpg",
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
    from stable_baselines3.common.evaluation import evaluate_policy

    env = gym.make('bot3RLNav/DiscreteWorld-v5', map_file="data/map01.jpg",
                   robot_file="data/robot.png")
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, gamma=0.1, exploration_fraction=0.8)
    model.learn(total_timesteps=120000, log_interval=4)

    obs = env.reset()
    name = "bot3"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    rate = 100  # frame rate in ms
    count = 2000
    while count > 0:
        frame = env.render(mode="rgb_array")
        cv2.imshow("bot3", frame)
        cv2.waitKey(rate)
        action, _states = model.predict(obs, deterministic=True)
        # action = 5
        obs, reward, done, info = env.step(action)
        print(count, info, reward, env.actions[action])
        if done:
            print("done.")
            frame = env.render(mode="rgb_array")
            cv2.imshow("bot3", frame)
            cv2.waitKey(rate*20)
            break
        count -= 1
    print()
    from stable_baselines3.common.monitor import Monitor
    env1 = Monitor(env)
    o = evaluate_policy(model, env1, n_eval_episodes=10, render=False,
                        # return_episode_rewards=True
                        )
    print(o)
    env.close()


if __name__ == '__main__':
    # check()
    # todo: continue use discreteworld4 tomorrow, also remember to label locally before pulling, committing and pushing
    train()
