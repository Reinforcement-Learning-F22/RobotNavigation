from gym.envs.registration import register

register(
    id='bot3RLNav/World-v0',
    entry_point='bot3RLNav.envs:World',
    max_episode_steps=300,
)
