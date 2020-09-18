from gym.envs.registration import registry, register, make, spec

register(
		id='dVRLPick-v0',
		entry_point='dVRL_simulator.environments:PSMPickEnv',
		max_episode_steps=150,
)
