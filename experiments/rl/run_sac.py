# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from tf2rl.algos.sac import SAC

from experiments.rl.env_wrap_rl import CyThetaObsMazeRLWrapper
from experiments.rl.trainer import MazeRLTrainer

if __name__ == "__main__":
    parser = MazeRLTrainer.get_argument()
    parser = SAC.get_argument(parser)
    parser.add_argument("--reset-to-random-ring", action="store_true")
    parser.add_argument("--input-state", action="store_true")
    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=10000)
    parser.set_defaults(episode_max_steps=500)
    parser.set_defaults(max_steps=int(2e6))
    parser.set_defaults(test_interval=10000)
    args = parser.parse_args()

    env = CyThetaObsMazeRLWrapper(reset_to_random_ring=args.reset_to_random_ring, input_state=args.input_state)
    test_env = CyThetaObsMazeRLWrapper(is_render=args.show_test_progress, input_state=args.input_state)

    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
    )
    trainer = MazeRLTrainer(policy, env, args, test_env=test_env)
    if args.evaluate:
        trainer.evaluate_policy(0)
    else:
        trainer()
