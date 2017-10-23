#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines.ppo1.continuous_gridworld import ContinuousGridworld
from baselines import logger
import sys

def train(env_id, num_timesteps, seed, tb_dir=None):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env =ContinuousGridworld('gridworld', visualize=True)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=256,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear', tb_dir=tb_dir
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--tb-dir', type=str, default=None)
    args = parser.parse_args()
    train(args.env, num_timesteps=1e6, seed=args.seed, tb_dir='/Users/chris/tb_test/')#tb_dir=args.tb_dir)


if __name__ == '__main__':
    main()
