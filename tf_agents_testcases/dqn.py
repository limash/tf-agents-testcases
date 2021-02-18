#!/usr/bin/env python3
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. It is distributed
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.

import abc
import logging
# import time

import tensorflow as tf

from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.eval.metric_utils import log_metrics

import tf_agents_testcases.misc as misc
import tf_agents_testcases.networks as networks

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

logging.getLogger().setLevel(logging.INFO)


def get_dqn_agent(env, q_net, n_step_update=1):
    train_step = tf.Variable(0)
    update_period = 4  # run a training step every 4 collect steps

    optimizer = tf.keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0,
                                            epsilon=0.00001, centered=True)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=2.5e-4, decay=0.95, momentum=0.0,
    #                                                 epsilon=0.00001, centered=True)

    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0,  # initial ε
        decay_steps=250000 // update_period,  # <=> 1,000,000 ALE frames
        end_learning_rate=0.01)  # final ε

    agent = dqn_agent.DqnAgent(env.time_step_spec(),
                               env.action_spec(),
                               q_network=q_net,
                               optimizer=optimizer,
                               n_step_update=n_step_update,
                               target_update_period=2000,  # <=> 32,000 ALE frames
                               td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
                               # td_errors_loss_fn=common.element_wise_squared_loss,
                               gamma=0.99,  # discount factor
                               train_step_counter=train_step,
                               epsilon_greedy=lambda: epsilon_fn(train_step)
                               )
    agent.initialize()

    return agent, update_period


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


def get_and_fill_replay_buffer(agent, env, update_period, n_step_update=1, replay_buffer_max_length=200000):
    # Initialize a replay buffer -----------------------------------------
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=env.batch_size,
        max_length=replay_buffer_max_length)

    # Save some initial data to the replay buffer
    # it can be replaced by observer and driver
    # env.reset()
    # misc.collect_data(env, agent.collect_policy, replay_buffer, steps)

    replay_buffer_observer = replay_buffer.add_batch

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    collect_driver = DynamicStepDriver(
        env,
        agent.collect_policy,
        observers=[replay_buffer_observer] + train_metrics,
        num_steps=update_period)  # collect steps for each training iteration

    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        env.time_step_spec(),
        env.action_spec())

    init_driver = DynamicStepDriver(
        env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch, misc.ShowProgress(20000)],
        num_steps=20000)  # <=> 80,000 ALE frames

    final_time_step, final_policy_state = init_driver.run()

    batch_size = 64  # as in dqn2015 article
    # Dataset generates trajectories with shape [Bx'num_steps'x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=n_step_update + 1).prefetch(3)
    iterator = iter(dataset)
    return replay_buffer, iterator, collect_driver, train_metrics


class QNet(abc.ABC):

    @abc.abstractmethod
    def __init__(self, env_name):
        # Initialize environments --------------------------------------------
        if env_name == 'BreakoutNoFrameskip-v4':
            max_episode_steps = 27000  # <=> 108k ALE frames since 1 step = 4 frames
            train_env = suite_atari.load(
                env_name,
                max_episode_steps=max_episode_steps,
                gym_env_wrappers=[AtariPreprocessing, FrameStack4])
            eval_env = suite_atari.load(
                env_name,
                max_episode_steps=max_episode_steps,
                gym_env_wrappers=[AtariPreprocessing, FrameStack4])
        else:
            train_env = suite_gym.load(env_name)
            eval_env = suite_gym.load(env_name)

        self._train_env = tf_py_environment.TFPyEnvironment(train_env)
        self._eval_env = tf_py_environment.TFPyEnvironment(eval_env)

        # random_policy = random_tf_policy.RandomTFPolicy(self._train_env.time_step_spec(),
        #                                                 self._train_env.action_spec())

        # Evaluate random policy
        # self._num_eval_episodes = 3
        # avg_return = misc.compute_avg_return(self._train_env, random_policy, self._num_eval_episodes)
        # print(f"The average return of random policy is {avg_return}")

        self._agent = None
        self._replay_buffer = None
        self._iterator = None
        self._collect_driver = None
        self._train_metrics = None
        self._py_eval_env = eval_env
        
    @property
    def get_py_eval_env(self):
        return self._py_eval_env
    
    @property
    def get_eval_env(self):
        return self._eval_env

    @property
    def get_policy(self):
        return self._agent.policy

    def train(self, num_iterations=10000):
        # Training -----------------------------------------------------------
        # collect_steps_per_iteration = 1
        # train_cycles_per_iteration = 1
        # eval_interval = 200

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        self._agent.train = common.function(self._agent.train)
        self._collect_driver.run = common.function(self._collect_driver.run)
        # Reset the train step
        # self._agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        # avg_return = misc.compute_avg_return(self._eval_env, self._agent.policy, self._num_eval_episodes)
        # returns = [avg_return]
        # print(f"The average return of agent policy is {avg_return}")
        # print("-----------------------------------------------------------------")

        # self._train_env.reset()

        time_step = None
        policy_state = self._agent.collect_policy.get_initial_state(self._train_env.batch_size)

        for iteration in range(num_iterations):

            # Collect a few steps using collect_policy and save to the replay buffer.
            # t0 = time.time()
            # print("Start data collecting")
            # if we do not reset, there will be not enough steps for a whole iteration
            # since a step will be consumed for a reset
            # self._train_env.reset()
            # for collect_step in range(collect_steps_per_iteration):
            #     misc.collect_step(self._train_env, self._agent.collect_policy, self._replay_buffer, collect_step)
            # t1 = time.time()
            time_step, policy_state = self._collect_driver.run(time_step, policy_state)

            # Sample a batch of data from the buffer and update the agent's network.
            # print("Start training")
            # for _ in range(train_cycles_per_iteration):
            #     experience, unused_info = next(self._iterator)
            #     self._agent.train(experience)
            trajectories, buffer_info = next(self._iterator)
            # t2 = time.time()
            train_loss = self._agent.train(trajectories)
            # t3 = time.time()
            print("\rIteration:{} loss:{:.5f}".format(iteration, train_loss.loss.numpy()), end="")
            # print(f"Time elapsed for training is {t3 - t2} ;", end="")
            if iteration % 1000 == 0:
                log_metrics(self._train_metrics)

            # step = self._agent.train_step_counter.numpy()
            # print(f"Training step number is {step}")

            # if step % eval_interval == 0:
            #     avg_return = misc.compute_avg_return(self._eval_env, self._agent.policy, self._num_eval_episodes)
            #     print('Step = {0}: Average Return = {1}'.format(step, avg_return))
            #     print(f"Time elapsed for collecting is {t1 - t0}")
            #     returns.append(avg_return)
            #     # misc.print_q_values(self._eval_env, self._agent.policy, self._q_net)

        return self._agent.policy


class DQNet(QNet):
    NETWORKS = {'CartPole-v0': networks.get_q_network_simple,
                'CartPole-v1': networks.get_q_network_simple,
                'BreakoutNoFrameskip-v4': networks.get_q_network_with_conv,
                'gym_halite:halite-v0': networks.get_q_network_halite}

    def __init__(self, env_name):
        # Initialize environments --------------------------------------------
        super().__init__(env_name)
        n_step_update = 1

        # Initialize Q Network -----------------------------------------------
        self._q_net = DQNet.NETWORKS[env_name](self._train_env)

        # Initialize DQN agent -----------------------------------------------
        self._agent, update_period = get_dqn_agent(self._train_env, self._q_net, n_step_update=n_step_update)
        # useful for debugging
        # self._agent._enable_functions = False

        # misc.print_q_values(self._train_env, self._agent.policy, self._q_net)

        self._replay_buffer, self._iterator, self._collect_driver, self._train_metrics = get_and_fill_replay_buffer(
            self._agent,
            self._train_env,
            update_period,
            n_step_update=n_step_update
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

        self._replay_buffer, self._iterator, self._collect_driver, self._train_metrics = get_and_fill_replay_buffer(
            self._agent,
            self._train_env,
            n_step_update=n_step_update
        )
