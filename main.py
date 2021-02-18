from tf_agents_testcases import dqn


if __name__ == '__main__':
    """Available environments:
       CartPole-v0(1),
       Breakout-v4,
       gym_halite:halite-v0 
    """
    # agent = dqn.DQNet(env_name='CartPole-v1')
    agent = dqn.DQNet(env_name='BreakoutNoFrameskip-v4')
    # agent = dqn.DQNet(env_name='gym_halite:halite-v0')

    # agent = dqn.CDQNet(env_name='CartPole-v1')
    # agent = dqn.CDQNet(env_name='gym_halite:halite-v0')
    policy = agent.train(num_iterations=10000)
