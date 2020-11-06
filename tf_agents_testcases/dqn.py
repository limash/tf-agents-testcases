#!/usr/bin/env python3
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. It is distributed
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.

import time
from collections import OrderedDict

import tensorflow as tf

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import tf_agents_testcases.misc as misc


def get_q_network(env):
    preprocessing_layers = OrderedDict({'halite_map': tf.keras.layers.Flatten(),
                                        'scalar_features': tf.keras.layers.Flatten()})
    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
    fc_layer_params = (1024, 1024)
    q_net = q_network.QNetwork(
        env.observation_spec(),
        env.action_spec(),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=fc_layer_params)
    return q_net


def get_dqn_agent(train_env, q_net):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # tf.compat.v1.enable_v2_behavior()
    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=0.95,
        train_step_counter=train_step_counter)
    agent.initialize()
    return agent


class DQNet:
    def __init__(self, env_name='gym_halite:halite-v0'):
        # Initialize environments --------------------------------------------
        train_env = suite_gym.load(env_name)
        self._train_env = tf_py_environment.TFPyEnvironment(train_env)
        eval_env = suite_gym.load(env_name)
        self._eval_env = tf_py_environment.TFPyEnvironment(eval_env)

        # Initialize Q Network -----------------------------------------------
        self._q_net = get_q_network(self._train_env)

        # Initialize DQN agent -----------------------------------------------
        self._agent = get_dqn_agent(self._train_env, self._q_net)

        random_policy = random_tf_policy.RandomTFPolicy(self._train_env.time_step_spec(),
                                                        self._train_env.action_spec())

        # Evaluate random policy
        self._num_eval_episodes = 3
        avg_return = misc.compute_avg_return(self._train_env, random_policy, self._num_eval_episodes)
        print(f"The average return of random policy is {avg_return}")

        # Initialize a replay buffer -----------------------------------------
        replay_buffer_max_length = 39999
        self._replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self._agent.collect_data_spec,
            batch_size=self._train_env.batch_size,
            max_length=replay_buffer_max_length)

        # Save some initial data to the replay buffer
        self._train_env.reset()
        # There are a reset + 399 steps maximum
        # thus, 400 steps in total
        # for example, the last step of tenth iteration is 3999
        # for example, the first step of eleventh iteration is 4001
        misc.collect_data(self._train_env, random_policy, self._replay_buffer, steps=3999)

        batch_size = 64
        # Dataset generates trajectories with shape [Bx2x...]
        self._dataset = self._replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)
        self._iterator = iter(self._dataset)

    @property
    def get_policy(self):
        return self._agent.policy

    def train(self, num_iterations = 1000):
        # Training -----------------------------------------------------------
        collect_steps_per_iteration = 1
        train_cycles_per_iteration = 1
        eval_interval = 200

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        self._agent.train = common.function(self._agent.train)
        # Reset the train step
        self._agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = misc.compute_avg_return(self._eval_env, self._agent.policy, self._num_eval_episodes)
        returns = [avg_return]
        print(f"The average return of agent policy is {avg_return}")
        print("-----------------------------------------------------------------")

        self._train_env.reset()
        for iteration in range(num_iterations):
            # Collect a few steps using collect_policy and save to the replay buffer.
            # t0 = time.time()
            # print("Start data collecting")
            # if we do not reset, there will be not enough steps for a whole iteration
            # since a step will be consumed for a reset
            # self._train_env.reset()
            for collect_step in range(collect_steps_per_iteration):
                misc.collect_step(self._train_env, self._agent.collect_policy, self._replay_buffer, collect_step)
            # t1 = time.time()
            # print(f"Time elapsed for collecting is {t1 - t0}")

            # Sample a batch of data from the buffer and update the agent's network.
            # print("Start training")
            # t2 = time.time()
            for _ in range(train_cycles_per_iteration):
                experience, unused_info = next(self._iterator)
                self._agent.train(experience)
            # t3 = time.time()
            # print(f"Time elapsed for training is {t3 - t1}")

            step = self._agent.train_step_counter.numpy()
            # print(f"Training step number is {step}")

            if step % eval_interval == 0:
                avg_return = misc.compute_avg_return(self._eval_env, self._agent.policy, self._num_eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)

        return returns, self._agent.policy
