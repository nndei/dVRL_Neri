from gym.envs.registration import registry, register, make, spec

register(
		id='ReachRailKidney-v1',
		entry_point='dVRL_simulator.environments:PSMReachEnv',
		max_episode_steps=150,
)


#register(
#		id='dVRLPick-v0',
#		entry_point='dVRL_simulator.environments:PSMPickEnv',
#		max_episode_steps=100,
#	)

