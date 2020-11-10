#!/usr/bin/env python3
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. It is distributed
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.

import time

import tensorflow as tf

from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import tf_agents_testcases.misc as misc
import tf_agents_testcases.networks as networks


def get_dqn_agent(env, q_net):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=0.95,
        train_step_counter=train_step_counter)
    agent.initialize()
    return agent


def get_categorical_dqn_agent(env, cat_q_net, n_step_update):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    train_step_counter = tf.Variable(0)
    agent = categorical_dqn_agent.CategoricalDqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        categorical_q_network=cat_q_net,
        optimizer=optimizer,
        min_q_value=-20,
        max_q_value=20,
        n_step_update=n_step_update,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=0.99,
        train_step_counter=train_step_counter)
    agent.initialize()
    return agent


def get_and_fill_replay_buffer(agent, env, n_step_update=1, replay_buffer_max_length=39999, steps=3999):

    # Initialize a replay buffer -----------------------------------------
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=env.batch_size,
        max_length=replay_buffer_max_length)

    # Save some initial data to the replay buffer
    env.reset()
    # There are a reset + 399 steps maximum
    # thus, 400 steps in total
    # for example, the last step of tenth iteration is 3999
    # for example, the first step of eleventh iteration is 4001
    misc.collect_data(env, agent.collect_policy, replay_buffer, steps)

    batch_size = 64
    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=n_step_update+1).prefetch(3)
    iterator = iter(dataset)
    return replay_buffer, iterator


class QNet:
    def __init__(self, env_name):
        # Initialize environments --------------------------------------------
        train_env = suite_gym.load(env_name)
        self._train_env = tf_py_environment.TFPyEnvironment(train_env)
        eval_env = suite_gym.load(env_name)
        self._eval_env = tf_py_environment.TFPyEnvironment(eval_env)

        random_policy = random_tf_policy.RandomTFPolicy(self._train_env.time_step_spec(),
                                                        self._train_env.action_spec())

        # Evaluate random policy
        self._num_eval_episodes = 3
        avg_return = misc.compute_avg_return(self._train_env, random_policy, self._num_eval_episodes)
        print(f"The average return of random policy is {avg_return}")

        self._agent = None
        self._replay_buffer = None
        self._iterator = None

    @property
    def get_policy(self):
        return self._agent.policy

    def train(self, num_iterations=10000):
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
                # misc.print_q_values(self._eval_env, self._agent.policy, self._q_net)

        return returns, self._agent.policy


class DQNet(QNet):
    NETWORKS = {'CartPole-v0': networks.get_q_network_simple,
                'CartPole-v1': networks.get_q_network_simple,
                'gym_halite:halite-v0': networks.get_q_network_halite}

    def __init__(self, env_name):
        # Initialize environments --------------------------------------------
        super().__init__(env_name)

        # Initialize Q Network -----------------------------------------------
        self._q_net = DQNet.NETWORKS[env_name](self._train_env)

        # Initialize DQN agent -----------------------------------------------
        self._agent = get_dqn_agent(self._train_env, self._q_net)

        # misc.print_q_values(self._train_env, self._agent.policy, self._q_net)

        self._replay_buffer, self._iterator = get_and_fill_replay_buffer(
            self._agent,
            self._train_env
        )


class CDQNet(QNet):
    NETWORKS = {'CartPole-v0': networks.get_categorical_q_network_simple,
                'CartPole-v1': networks.get_categorical_q_network_simple,
                'gym_halite:halite-v0': networks.get_categorical_q_network_halite
                }

    def __init__(self, env_name):
        # Initialize environments --------------------------------------------
        super().__init__(env_name)
        n_step_update = 2

        # Initialize Q Network -----------------------------------------------
        self._q_net = CDQNet.NETWORKS[env_name](self._train_env)

        # Initialize DQN agent -----------------------------------------------
        self._agent = get_categorical_dqn_agent(self._train_env, self._q_net, n_step_update=n_step_update)

        self._replay_buffer, self._iterator = get_and_fill_replay_buffer(
            self._agent,
            self._train_env,
            n_step_update=n_step_update
        )
