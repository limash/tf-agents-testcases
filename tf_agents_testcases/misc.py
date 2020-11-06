#!/usr/bin/env python3
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. It is distributed
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.

import time

from tf_agents.trajectories import trajectory


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
