from gym.envs.registration import register

register(
    id='bot3RLNav/World-v0',
    entry_point='bot3RLNav.envs:World',
    max_episode_steps=300,
)

register(
    id='bot3RLNav/World-v1',
    entry_point='bot3RLNav.envs:World1',
    max_episode_steps=300,
)

register(
    id='bot3RLNav/DiscreteWorld-v0',
    entry_point='bot3RLNav.envs:DiscreteWorld',
    max_episode_steps=300,
)