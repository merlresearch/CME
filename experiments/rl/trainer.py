# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

import numpy as np
import tensorflow as tf
from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.utils import save_path
from tf2rl.misc.get_replay_buffer import get_replay_buffer

from experiments.simulation import get_initial_states_16_points, get_initial_states_n_points_limited


def get_initial_states(enable_limited_area, n_evaluate_points=9):
    if enable_limited_area:
        return get_initial_states_n_points_limited(n=n_evaluate_points)
    else:
        return get_initial_states_16_points()


class MazeRLTrainer(Trainer):
    def __init__(self, *args, enable_limited_area=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_states = get_initial_states(enable_limited_area)
        self._test_episodes = self._initial_states.shape[0]

    def evaluate_policy(self, total_steps):
        tf.summary.experimental.set_step(total_steps)
        if self._normalize_obs:
            self._test_env.normalizer.set_params(*self._env.normalizer.get_params())
        avg_test_return = 0.0
        avg_test_steps = 0.0
        if self._save_test_path:
            replay_buffer = get_replay_buffer(self._policy, self._test_env, size=self._episode_max_steps)

        for i, initial_state in enumerate(self._initial_states):
            episode_return = 0.0
            self._test_env.reset()
            self._test_env.set_state_with_ring_and_theta(4, initial_state[2])
            obs = self._test_env._get_obs()

            for test_steps in range(self._episode_max_steps):
                action = self._policy.get_action(obs, test=True)
                next_obs, reward, done, _ = self._test_env.step(action)
                if self._save_test_path:
                    replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done)

                elif self._show_test_progress:
                    self._test_env.render()
                episode_return += reward
                obs = next_obs
                if done:
                    break
            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(total_steps, i, episode_return)
            if self._save_test_path:
                save_path(
                    replay_buffer._encode_sample(np.arange(self._episode_max_steps)),
                    os.path.join(self._output_dir, prefix + ".pkl"),
                )
                replay_buffer.clear()
            avg_test_return += episode_return
            avg_test_steps += test_steps + 1
        avg_test_steps /= self._test_episodes
        tf.summary.scalar(name="Common/average_test_steps", data=avg_test_steps)
        return avg_test_return / self._test_episodes
