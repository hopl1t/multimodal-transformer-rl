# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import argparse
import os
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import pickle
from os import path

from Minecraft import Config
from Minecraft import Minecraft

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
)

def save_run(agent, run_name, args, optimizer, global_step, episode_count, initial_update):
    save_path = path.join(args.save_dir, run_name + '.pkl')
    run_dict = {
        'agent': agent,
        'run_name': run_name,
        'args': args,
        'optimizer': optimizer,
        'global_step': global_step,
        'episode_count': episode_count,
        'initial_update': initial_update

    }
    with open(save_path, 'wb') as f:
        pickle.dump(run_dict, f)
        print(f'💾 Saved run to {save_path} 💾')


def load_run(pkl_path):
    with open(pkl_path, 'rb') as f:
        run_dict = pickle.load(f)
        print(f'💾 Loaded run from {pkl_path} 💾')
    return run_dict['agent'], run_dict['run_name'], run_dict['args'], run_dict['optimizer'], run_dict['global_step'], run_dict['episode_count'], run_dict['initial_update']


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="minecraft",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=1,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=1,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    # New parameters
    parser.add_argument("--save-interval", type=int, default=10000,
        help="Save the agent every n updates. if 0 then agent is not saved at all (Also on exit)")
    parser.add_argument("--save-dir", type=str, default="../saved_agents",
        help="A path to a local folder in which to save the run")
    parser.add_argument("--load-from", type=str, default="",
        help="A path to a local pickle file from which to load the run")
    parser.add_argument("--max-episode-len", type=int, default=10000,
        help="Maximal length of a single episode")
    parser.add_argument("--policy-offset", type=float, default=0.5,
        help="Offsets policy dist to reduce it's std and increase exploration")
    parser.add_argument("--max-episodes", type=int, default=1000000,
        help="Offsets policy dist to reduce it's std and increase exploration")
    parser.add_argument("--attn-type", type=str, default=None,
        help="Attention type to use, either: None if not specified, 'casl' or 'new'")
    parser.add_argument("--fusion-type", type=str, default='concat',
        help="How to fuse feautres: either 'sum' or 'concat'")
    parser.add_argument("--conv-size", type=str, default='big',
        help="Size of initial mutual conv layers: either 'big' or 'small'")
    parser.add_argument("--print-interval", type=int, default=1,
        help="Print stats to stdout every so episodes")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_minecraft_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = Minecraft()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        _ = env.reset()
        env = ClipRewardEnv(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
