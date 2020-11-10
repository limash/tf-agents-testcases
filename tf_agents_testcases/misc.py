#!/usr/bin/env python3
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. It is distributed
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.

import time
import numpy as np
import tensorflow as tf

from tf_agents.environments import suite_gym
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.trajectories import time_step as ts

from kaggle_environments import make
import kaggle_environments.envs.halite.helpers as hh

from gym_halite.envs.halite_env import get_scalar_features, get_feature_maps

from tf_agents.trajectories import trajectory

from tf_agents_testcases import networks


def print_q_values(environment, policy, q_net):
    time_step = environment.reset()
    while not time_step.is_last():
        action_step = policy.action(time_step)
        q_values, network_state = q_net(time_step.observation)
        time_step = environment.step(action_step.action)
        print(f"Q value is {q_values}")
        print(f"Action is {action_step.action}")


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        # t0 = time.time()
        time_step = environment.reset()
        step = 0
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            step += 1

        total_return += episode_return
        # t1 = time.time()
        # print(f"A number of steps is {step}")
        # print(f"Time elapsed is {t1 - t0}")

    avg_return = total_return / num_episodes
    return avg_return.numpy()


def collect_step(environment, policy, buffer, step):
    # there is a situation possible when
    # the current_time_step is last and the next_time_step is first
    # apparently it restarts the environment
    # but it is unknown what happens in traj during reset and if it saves this traj to buffer
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # if next_time_step.is_last():
    #     print(f"The last time step is {step+1}")
    # if time_step.is_first():
    #     print(f"The first time step is {step+1}")

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for step in range(steps):
        collect_step(env, policy, buffer, step)


def get_env_and_critic_model():
    env = suite_gym.load('gym_halite:halite-v0')
    observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(env)
    )
    model = networks.CriticNetwork((observation_spec, action_spec))
    return env, model


def get_env_and_actor_model():
    env = suite_gym.load('gym_halite:halite-v0')
    observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(env)
    )
    model = networks.ActorDistributionNetwork(
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
