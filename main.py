import numpy as np
import tensorflow as tf

from tf_agents.environments import suite_gym
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.trajectories import time_step as ts

import kaggle_environments.envs.halite.helpers as hh

from gym_halite.envs.halite_env import get_scalar_features, get_halite_map

from tf_agents_testcases import dqn, models


def get_env_and_critic_model():
    env = suite_gym.load('gym_halite:halite-v0')
    observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(env)
    )
    model = models.CriticNetwork((observation_spec, action_spec))
    return env, model


def get_env_and_actor_model():
    env = suite_gym.load('gym_halite:halite-v0')
    observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(env)
    )
    model = models.ActorDistributionNetwork(
        observation_spec, action_spec,
        continuous_projection_net=(
            tanh_normal_projection_network.TanhNormalProjectionNetwork
        )
    )
    return env, model


def make_a_prediction_with_critic(env, model):
    step = env.reset()
    halite_map, scalars = step.observation
    halite_map = halite_map[np.newaxis, ...]
    scalars = scalars[np.newaxis, ...]
    action = np.array([-0.9], dtype=np.float32)
    action = action[np.newaxis, ...]
    # action = tf.constant([-0.9, ], dtype=tf.float32)
    # action = tf.expand_dims(action, axis=0)
    return model(((halite_map, scalars), action))


def make_a_prediction_with_actor(env, model):
    step = env.reset()
    halite_map, scalars = step.observation
    halite_map = halite_map[np.newaxis, ...]
    scalars = scalars[np.newaxis, ...]
    actions, network_state = model((halite_map, scalars))

    return actions, network_state


def get_halite_agent(policy):
    """halite agent adapted for tf-agents policy"""
    def dqn_halite_agent(obs, config):
        # from tensorflow.python.training.tracking.data_structures import _DictWrapper
        from collections import OrderedDict

        directions = [hh.ShipAction.NORTH,
                      hh.ShipAction.SOUTH,
                      hh.ShipAction.WEST,
                      hh.ShipAction.EAST]

        board = hh.Board(obs, config)
        me = board.current_player

        skalar_features = get_scalar_features(board)
        skalar_features = skalar_features[np.newaxis, ...]
        skalar_features = tf.convert_to_tensor(skalar_features, dtype=tf.float32)
        halite_map = get_halite_map(board)
        halite_map = halite_map[np.newaxis, ...]
        halite_map = tf.convert_to_tensor(halite_map, dtype=tf.float32)
        state = OrderedDict({'halite_map': halite_map, 'scalar_features': skalar_features})
        # state = _DictWrapper(state)

        time_step = ts.transition(state,
                                  reward=np.array([0], dtype=np.float32),
                                  discount=[1.0])

        action_step = policy.action(time_step)
        action_number = action_step.action.numpy()[0]
        try:
            me.ships[0].next_action = directions[action_number]
        except IndexError:
            pass
        return me.next_actions
    return dqn_halite_agent


if __name__ == '__main__':
    """Available environments:
       CartPole-v0,
       gym_halite:halite-v0 
    """
    agent = dqn.DQNet(env_name='CartPole-v0')
    # agent = dqn.DQNet(env_name='gym_halite:halite-v0')
    returns, policy = agent.train(num_iterations=10000)
