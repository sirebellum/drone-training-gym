from gym.envs.registration import register

register(
    id='PathFinder-v0',
    entry_point='path_finder.envs:PathFinderEnv',
    max_episode_steps=50,
    reward_threshold = 0.95,
    nondeterministic = False
)
