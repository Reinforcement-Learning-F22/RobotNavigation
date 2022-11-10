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

register(
    id='bot3RLNav/World-v2',
    entry_point='bot3RLNav.envs:World2',
    max_episode_steps=1000,
)

register(
    id='bot3RLNav/DiscreteWorld-v1',
    entry_point='bot3RLNav.envs:DiscreteWorld1',
    max_episode_steps=5000,
)

register(
    id='bot3RLNav/DiscreteWorld-v2',
    entry_point='bot3RLNav.envs:DiscreteWorld2',
    max_episode_steps=1000,
)
