import numpy as np
import tensorflow as tf

from tf_agents.environments import suite_gym
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.trajectories import time_step as ts

from kaggle_environments import make
import kaggle_environments.envs.halite.helpers as hh

from gym_halite.envs.halite_env import get_scalar_features, get_feature_maps

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
    feature_maps, scalars = step.observation
    feature_maps = feature_maps[np.newaxis, ...]
    scalars = scalars[np.newaxis, ...]
    action = np.array([-0.9], dtype=np.float32)
    action = action[np.newaxis, ...]
    # action = tf.constant([-0.9, ], dtype=tf.float32)
    # action = tf.expand_dims(action, axis=0)
    return model(((feature_maps, scalars), action))


def make_a_prediction_with_actor(env, model):
    step = env.reset()
    feature_maps, scalars = step.observation
    feature_maps = feature_maps[np.newaxis, ...]
    scalars = scalars[np.newaxis, ...]
    actions, network_state = model((feature_maps, scalars))

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
        feature_maps = get_feature_maps(board)
        feature_maps = feature_maps[np.newaxis, ...]
        feature_maps = tf.convert_to_tensor(feature_maps, dtype=tf.float32)
        state = OrderedDict({'feature_maps': feature_maps, 'scalar_features': skalar_features})
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


def render_halite(policy):
    board_size = 5
    starting_halite = 5000
    env = make("halite",
               configuration={"size": board_size,
                              "startingHalite": starting_halite},
               debug=True)

    halite_agent = get_halite_agent(policy)
    env.run([halite_agent])
    env.render(mode="ipython", width=800, height=600)


if __name__ == '__main__':
    """Available environments:
       CartPole-v0(1),
       gym_halite:halite-v0 
    """
    # agent = dqn.DQNet(env_name='CartPole-v1')
    agent = dqn.DQNet(env_name='gym_halite:halite-v0')
    # agent = dqn.CDQNet(env_name='CartPole-v1')
    # agent = dqn.CDQNet(env_name='gym_halite:halite-v0')
    returns, policy = agent.train(num_iterations=1000)
    render_halite(policy)
