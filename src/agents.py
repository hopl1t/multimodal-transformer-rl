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
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
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

# from utils import save_run, load_run, parse_args, make_minecraft_env, layer_init


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
    def __init__(self, feature_input_size, device, init_bias=0, min_audio_attn=0):
        super().__init__()
        self.init_bias = init_bias
        self.min_audio_attn = min_audio_attn
        self.device = device
        if init_bias or min_audio_attn:
            print(f"# INITIALIZING CASL ATTENTION WITH bias: {init_bias}, min_audio_attn: {min_audio_attn}")
        self.audio_fc = nn.Linear(feature_input_size, 32)
        self.video_fc = nn.Linear(feature_input_size, 32)
        self.state_fc = nn.Linear(128, 32)
        self.attention = nn.Linear(32, 2)
        if init_bias:
            nn.init.normal_(self.attention.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.attention.bias[0], init_bias)
            nn.init.constant_(self.attention.bias[1], 0.0)
    
    def forward(self, video_features, audio_features, lstm_state):
        attn_video_features = self.video_fc(video_features)
        attn_audio_features = self.audio_fc(audio_features)
        # h_n as in casl. See https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        attn_lstm_state = self.state_fc(lstm_state[0])
        activated = torch.tanh(attn_video_features + attn_audio_features + attn_lstm_state)
        attention_weights = torch.softmax(self.attention(activated).squeeze(0), axis=-1)
        if self.min_audio_attn:
            new_audio_weight = torch.maximum(attention_weights[:, 1], torch.tensor(self.min_audio_attn).to(self.device))
            # compensation for video s.t. the sum is 1
            new_video_weight = torch.minimum(attention_weights[:, 0], torch.tensor(1 - self.min_audio_attn).to(self.device))
            attention_weights = torch.cat((new_video_weight, new_audio_weight), -1).unsqueeze(0)
        video_features = attention_weights[:, 0].unsqueeze(1) * video_features
        audio_features = attention_weights[:, 1].unsqueeze(1) * audio_features
        return video_features, audio_features, attention_weights


class SeperateLstmsAttention(nn.Module):
    def __init__(self, feature_input_size):
        super().__init__()
        self.audio_fc = nn.Linear(feature_input_size, 32)
        self.video_fc = nn.Linear(feature_input_size, 32)
        self.video_state_norm = nn.LayerNorm(128)
        self.audio_state_norm = nn.LayerNorm(128)
        self.state_fc = nn.Linear(128, 32)
        self.attention = nn.Linear(32, 2)
    
    def forward(self, video_features, audio_features, lstm_state):
        attn_video_features = self.video_fc(video_features)
        attn_audio_features = self.audio_fc(audio_features)
        # h_n as in casl. See https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        video_lstm_state_norm = self.video_state_norm(lstm_state[0][0])
        audio_lstm_state_norm = self.audio_state_norm(lstm_state[1][0])
        lstm_state_features = self.state_fc(video_lstm_state_norm + audio_lstm_state_norm)
        activated = torch.tanh(attn_video_features + attn_audio_features + lstm_state_features)
        attention_weights = torch.softmax(self.attention(activated).squeeze(0), axis=-1)
        video_features = attention_weights[:, 0].unsqueeze(1) * video_features
        audio_features = attention_weights[:, 1].unsqueeze(1) * audio_features
        return video_features, audio_features, attention_weights


class MinecraftAgent(nn.Module):
    def __init__(self, envs, device, conv_type='big', attn_type='casl', fusion_type='sum', init_bias=False, min_audio_attn=0):
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
            if self.fusion_type == 'concat':
                self.lstm_size = self.feature_size * 2
            if self.fusion_type == 'sum':
                self.lstm_size = self.feature_size
        else:
            self.lstm_size =  self.feature_size
            if attn_type == 'casl':
                self.attn = CaslAttention(
                    self.feature_size, device=device, init_bias=init_bias, min_audio_attn=min_audio_attn)
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
        video_features = self.video_net(torch.index_select(x, 1, torch.tensor([0]).to(self.device)))
        audio_features = self.audio_net(torch.index_select(x, 1, torch.tensor([1]).to(self.device)))
        if self.attn_type:
            video_features, audio_features, attn_weights = self.attn(video_features, audio_features, lstm_state)
        if self.fusion_type == 'concat':
            fused_features = torch.cat([video_features, audio_features])
        elif self.fusion_type == 'sum':
            # fused_features = torch.cat((video_features, audio_features), dim=1)
            fused_features = video_features + audio_features
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


class AlignableCaslAgent(nn.Module):
    def __init__(self, envs, device, norm_type='layer', conv_type='big'):
        super().__init__()
        print(f"🤖Orthogonaly aligned agent🤖")
        if conv_type == 'big':
            self.feature_size = 512
        else:
            self.feature_size = 256

        self.norm_type = norm_type
        self.lstm_size = self.feature_size
        self.attn = CaslAttention(self.feature_size, device=device)

        self.video_net = conv_factory(conv_type)
        self.audio_net = conv_factory(conv_type)

        if self.norm_type == 'layer':
            self.video_norm = nn.LayerNorm(self.feature_size)
            self.audio_norm = nn.LayerNorm(self.feature_size)
            print("## USING LAYER NORM ##")
        elif self.norm_type == 'batch':
            print("## USING BATCH NORM ##")
            self.video_norm = nn.BatchNorm1d(self.feature_size)
            self.audio_norm = nn.BatchNorm1d(self.feature_size)
        elif self.norm_type == 'instance':
            print("## USING INSTANCE NORM ##")
            self.video_norm = nn.InstanceNorm1d(self.feature_size)
            self.audio_norm = nn.InstanceNorm1d(self.feature_size)

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
        video_features = self.video_net(torch.index_select(
            x, 1, torch.tensor([0]).to(self.device)))
        audio_features = self.audio_net(torch.index_select(
            x, 1, torch.tensor([1]).to(self.device)))
        
        # Normalization
        if self.norm_type in ['layer', 'instance']:
            video_features = self.video_norm(video_features)
            audio_features = self.audio_norm(audio_features)
        elif self.norm_type == 'batch':
            if video_features.shape[0] == 1:
                self.eval()
            video_features = self.video_norm(video_features)
            audio_features = self.audio_norm(audio_features)
            self.train()

        # Cross attention
        attn_video_features, attn_audio_features, attn_weights = self.attn(
            video_features, audio_features, lstm_state)

        # Modality fusion
        fused_features = attn_video_features + attn_audio_features
        
        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        fused_features = fused_features.reshape(
            (-1, batch_size, self.lstm.input_size))
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
        return new_hidden, lstm_state, (video_features, audio_features)

    def get_value(self, x, lstm_state, done):
        hidden, _, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state, (normalized_video_hidden, normalized_audio_hidden) = self.get_states(
            x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state, (normalized_video_hidden, normalized_audio_hidden)


class SeperateLstmSumAlignableAgent(nn.Module):
    def __init__(self, envs, device, use_attention=False, conv_type='big', norm_type='layer'):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            print(f"🤖SeperateLstmSumAlignableAgent with attention🤖")
        print(f"## USING NORM: {norm_type.upper()} ##")
        if conv_type == 'big':
            self.feature_size = 512
        else:
            self.feature_size = 256

        self.video_net = conv_factory(conv_type)
        self.audio_net = conv_factory(conv_type)

        if norm_type == 'layer':
            self.video_norm = nn.LayerNorm(self.feature_size)
            self.audio_norm = nn.LayerNorm(self.feature_size)
        elif norm_type == 'instance':
            self.video_norm = nn.InstanceNorm1d(self.feature_size)
            self.audio_norm = nn.InstanceNorm1d(self.feature_size)

        if self.use_attention:
            self.attn = SeperateLstmsAttention(self.feature_size)

        self.video_lstm = nn.LSTM(self.feature_size, 128)
        self.audio_lstm = nn.LSTM(self.feature_size, 128)
        for lstm in (self.video_lstm, self.audio_lstm):
            for name, param in lstm.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)

        self.actor = layer_init(
            nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)
        self.device = device

    def get_states(self, x, lstm_states, done):
        video_lstm_state = lstm_states[0]
        audio_lstm_state = lstm_states[1]
        video_features = self.video_net(torch.index_select(
            x, 1, torch.tensor([0]).to(self.device)))
        audio_features = self.audio_net(torch.index_select(
            x, 1, torch.tensor([1]).to(self.device)))

        # Normalization
        video_features = self.video_norm(video_features)
        audio_features = self.audio_norm(audio_features)

        # Attention
        if self.use_attention:
            video_features, audio_features, _ = self.attn(video_features, audio_features, lstm_states)

        # LSTM logic
        batch_size = video_lstm_state[0].shape[1]
        done = done.reshape((-1, batch_size))
        # LSTM video
        video_features_for_lstm = video_features.reshape(
            (-1, batch_size, self.video_lstm.input_size))
        new_hidden_video = []
        for h, d in zip(video_features_for_lstm, done):
            h, video_lstm_state = self.video_lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * video_lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * video_lstm_state[1],
                ),
            )
            new_hidden_video += [h]
        new_hidden_video = torch.flatten(torch.cat(new_hidden_video), 0, 1)
        # LSTM audio
        audio_features_for_lstm = audio_features.reshape(
            (-1, batch_size, self.audio_lstm.input_size))
        new_hidden_audio = []
        for h, d in zip(audio_features_for_lstm, done):
            h, audio_lstm_state = self.audio_lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * audio_lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * audio_lstm_state[1],
                ),
            )
            new_hidden_audio += [h]
        new_hidden_audio = torch.flatten(torch.cat(new_hidden_audio), 0, 1)
        
        # Fusion
        concatanated_features = new_hidden_video + new_hidden_audio

        return concatanated_features, (video_lstm_state, audio_lstm_state), (video_features, audio_features)

    def get_value(self, x, lstm_state, done):
        hidden, _, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state, modality_features = self.get_states(
            x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state, modality_features


class FixedAttentionAgent(nn.Module):
    def __init__(self, envs, device, conv_type='big', fusion_type='sum', audio_ratio=0.2):
        super().__init__()
        print(f"🤖Using FIXED attention with ratio {audio_ratio}🤖")
        self.fusion_type = fusion_type
        self.audio_ratio = audio_ratio
        if conv_type == 'big':
            self.feature_size = 512
        else:
            self.feature_size = 256
        if self.fusion_type == 'concat':
            self.lstm_size = self.feature_size * 2
        if self.fusion_type == 'sum':
            self.lstm_size = self.feature_size

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
        video_features = self.video_net(torch.index_select(
            x, 1, torch.tensor([0]).to(self.device)))
        audio_features = self.audio_net(torch.index_select(
            x, 1, torch.tensor([1]).to(self.device)))
        
        # Fixed attention
        video_features = (1 - self.audio_ratio) * video_features
        audio_features = self.audio_ratio * audio_features
        
        if self.fusion_type == 'concat':
            fused_features = torch.cat([video_features, audio_features])
        elif self.fusion_type == 'sum':
            # fused_features = torch.cat((video_features, audio_features), dim=1)
            fused_features = video_features + audio_features
        else:
            raise NotImplementedError
        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        fused_features = fused_features.reshape(
            (-1, batch_size, self.lstm.input_size))
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


class IBAgent(nn.Module):
    def __init__(self, envs, device):
        super().__init__()
        print(
            f"🤖Using IB agent🤖")
        self.feature_size = 512
        self.lstm_size = self.feature_size
        self.attn = CaslAttention(self.feature_size, device=device)

        self.video_net = conv_factory('big')
        self.audio_net = conv_factory('big')

        self.video_reconstruction = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.feature_size),
        )

        self.audio_reconstruction = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.feature_size),
        )

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
        video_features = self.video_net(torch.index_select(
            x, 1, torch.tensor([0]).to(self.device)))
        audio_features = self.audio_net(torch.index_select(
            x, 1, torch.tensor([1]).to(self.device)))
        video_features, audio_features, attn_weights = self.attn(
            video_features, audio_features, lstm_state)
        # fused_features = torch.cat((video_features, audio_features), dim=1)
        fused_features = video_features + audio_features

        # Feature reconstruction
        reconstructed_audio = self.audio_reconstruction(video_features)
        reconstructed_video = self.audio_reconstruction(audio_features)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        fused_features = fused_features.reshape(
            (-1, batch_size, self.lstm.input_size))
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

        return new_hidden, lstm_state, (video_features, audio_features), (reconstructed_video, reconstructed_audio)

    def get_value(self, x, lstm_state, done):
        hidden, _, _, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state, (video_features, audio_features), (reconstructed_video, reconstructed_audio) = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state, (video_features, audio_features), (reconstructed_video, reconstructed_audio)


class ModalityAdversary(nn.Module):
    def __init__(self, feature_size=512):
        super().__init__()
        self.feature_size = 512

        self.adverserial_net = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.feature_size),
        )
    def forward(self, modality):
        return self.adverserial_net(modality)


class VIBAgent(nn.Module):
    def __init__(self, envs, device, conv_type='big', use_attn=False, fusion_type='sum', dropout_ratio=0):
        super().__init__()
        print(f"🤖Using VIB agent🤖")
        print(f"🤖Using attention {use_attn}, conv_type: {conv_type}, fusion type: {fusion_type}, dropout: {dropout_ratio}, 🤖")
        self.use_attn = use_attn
        self.fusion_type = fusion_type
        self.dropout_ratio = dropout_ratio
        if conv_type == 'big':
            self.feature_size = 512
        else:
            self.feature_size = 256
        self.lstm_size = self.feature_size
        if use_attn:
            self.attn = CaslAttention(self.feature_size, device=device)

        self.video_net = conv_factory(conv_type)
        self.audio_net = conv_factory(conv_type)

        if self.fusion_type == 'tensor':
            self.pre_fusion_video = nn.Linear(self.feature_size, 50)
            self.pre_fusion_audio = nn.Linear(self.feature_size, 50)
            self.post_fusion_layer = nn.Linear((50 + 1) * (50 + 1), self.feature_size)
            # self.post_fusion_layer_2 = nn.Linear(self.lstm_size, self.lstm_size)

        if dropout_ratio:
            self.post_fusion_dropout = nn.Dropout(p=dropout_ratio)
        
        self.fc_mu = nn.Linear(self.feature_size, self.feature_size)
        self.fc_std = nn.Linear(self.feature_size, self.feature_size)

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
        video_features = self.video_net(torch.index_select(
            x, 1, torch.tensor([0]).to(self.device)))
        audio_features = self.audio_net(torch.index_select(
            x, 1, torch.tensor([1]).to(self.device)))

        # attn
        if self.use_attn:
            video_features, audio_features, attn_weights = self.attn(video_features, audio_features, lstm_state)

        # Fusion
        if self.fusion_type == 'tensor':
            video_features = self.pre_fusion_video(video_features)
            audio_features = self.pre_fusion_audio(audio_features)
            _audio_h = torch.cat(
                (Variable(torch.ones(audio_features.size(0), 1).type(torch.FloatTensor).to(self.device), requires_grad=False), audio_features), dim=1)
            _video_h = torch.cat(
                (Variable(torch.ones(video_features.size(0), 1).type(torch.FloatTensor).to(self.device), requires_grad=False), video_features), dim=1)
            fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
            fusion_tensor = fusion_tensor.view(-1, (audio_features.size(1) + 1) * (video_features.size(1) + 1))
            # fusion_tensor = fusion_tensor.view(-1, (audio_features.size(1) + 1) * (video_features.size(1) + 1), 1)
            fusion_tensor = self.post_fusion_layer(fusion_tensor)
        else:
            fusion_tensor = video_features + audio_features

        if self.dropout_ratio:
            fusion_tensor = self.post_fusion_dropout(fusion_tensor)

        # Reparameterization
        mu = self.fc_mu(fusion_tensor)
        std = F.softplus(self.fc_std(fusion_tensor)-5, beta=1)
        eps = torch.randn_like(std)
        z = mu + std*eps

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        z = z.reshape(
            (-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(z, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state, (mu, std)

    def get_value(self, x, lstm_state, done):
        hidden, _, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state, (mu, std) = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state, (mu, std)
