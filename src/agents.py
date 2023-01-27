# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from Minecraft import Config
from Minecraft import Minecraft

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from utils import save_run, load_run, parse_args, make_minecraft_env, layer_init


def conv_factory(size='big'):
    if size == 'big':
        return nn.Sequential(
            layer_init(
                nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
    elif size == 'small':
        return nn.Sequential(
            # Out size: (84 - 8) // 4 + 1 = 20
            layer_init(nn.Conv2d(1, 16, 8, stride=4)),
            nn.ReLU(),
            # Out size: (20 - 4) // 2 + 1 = 9
            layer_init(nn.Conv2d(16, 32, 4, stride=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 9 * 9, 256)),
            nn.ReLU(),
        )


class CaslAttention(nn.Module):
    def __init__(self, feature_input_size, device):
        super().__init__()
        self.audio_fc = nn.Linear(feature_input_size, 32)
        self.video_fc = nn.Linear(feature_input_size, 32)
        self.state_fc = nn.Linear(128, 32)
        self.attention = nn.Linear(32, 2)
    
    def forward(self, video_features, audio_features, lstm_state):
        attn_video_features = self.video_fc(video_features)
        attn_audio_features = self.audio_fc(audio_features)
        # h_n as in casl. See https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        attn_lstm_state = self.state_fc(lstm_state[0])
        activated = torch.tanh(attn_video_features + attn_audio_features + attn_lstm_state)
        attention_weights = torch.softmax(self.attention(activated).squeeze(0), axis=-1)
        video_features = attention_weights[:, 0].unsqueeze(1) * video_features
        audio_features = attention_weights[:, 1].unsqueeze(1) * audio_features
        return video_features, audio_features, attention_weights


class NewAttention(nn.Module):
    def __init__(self, feature_input_size, device):
        super().__init__()
        self.fc = nn.Linear(feature_input_size + feature_input_size + 128, 128)
        self.attention = nn.Linear(128, 2)
    
    def forward(self, video_features, audio_features, lstm_state):
        fc = self.fc(torch.cat((video_features, audio_features, lstm_state[0].squeeze(0).repeat_interleave(video_features.shape[0], dim=0)), dim=-1))
        activated = torch.tanh(fc)
        attention_weights = torch.softmax(self.attention(activated), axis=-1)
        video_features = attention_weights[:, 0].unsqueeze(-1) * video_features
        audio_features = attention_weights[:, 1].unsqueeze(-1) * audio_features
        return video_features, audio_features, attention_weights


class MinecraftAgent(nn.Module):
    def __init__(self, envs, device, conv_type='big', attn_type=None, fusion_type='concat'):
        super().__init__()
        print(
            f"🤖Using attention {attn_type}, conv_type: {conv_type}, fusion_type: {fusion_type}🤖")
        self.attn_type = attn_type
        self.fusion_type = fusion_type
        if conv_type == 'big':
            self.feature_size = 512
        else:
            self.feature_size = 256
        if not attn_type:
            self.lstm_size = self.feature_size * 2
        else:
            self.lstm_size =  self.feature_size
            if attn_type == 'casl':
                self.attn = CaslAttention(self.feature_size, device)
            elif attn_type == 'new':
                self.attn = NewAttention(self.feature_size, device)
            else:
                raise NotImplementedError
        
        self.video_net = conv_factory(conv_type)
        self.audio_net = conv_factory(conv_type)

        self.lstm = nn.LSTM(self.lstm_size, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(
            nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)
        self.device = device

    def get_states(self, x, lstm_state, done):
        video_features = self.video_net(torch.index_select(x, 1, torch.tensor([0]).to(self.device)).to(self.device) / 255.0)
        audio_features = self.audio_net(torch.index_select(x, 1, torch.tensor([1]).to(self.device)).to(self.device) / 255.0)
        if self.attn_type:
            video_features, audio_features, attn_weights = self.attn(video_features, audio_features, lstm_state)
        if self.fusion_type == 'concat':
            fused_features = torch.cat([video_features, audio_features])
        elif self.fusion_type == 'sum':
            fused_features = torch.cat((video_features, audio_features), dim=1)
        else:
            raise NotImplementedError
        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        fused_features = fused_features.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(fused_features, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


class GymAgent(nn.Module):
    def __init__(self, envs, conv_type='big'):
        super().__init__()
        self.video_net = conv_factory(conv_type)

        self.lstm = nn.LSTM(512, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(
            nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.video_net(x / 255.0)
        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


class OldAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(512, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(
            nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x / 255.0)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state
