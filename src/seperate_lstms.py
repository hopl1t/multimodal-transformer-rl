# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# from src.environments.casl_environment import Environment as CASLEnv
# from environments.Minecraft.Minecraft import Minecraft
from Minecraft import Config

from agents import SeperateLstmSumAlignableAgent
from torch.nn.functional import cosine_similarity
from utils import parse_args, make_env, distance_func

MAX_EPISODE_LEN = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALIGNMENT_VECTOR1 = torch.cat((torch.ones(1), torch.zeros(127))).to(torch.device(device))
ALIGNMENT_VECTOR2 = torch.cat((torch.zeros(127), torch.ones(1))).to(torch.device(device))

if __name__ == "__main__":
    args = parse_args()
    if Config.USE_AUDIO:
        print('### 🔊 USING AUDIO 🔊 ###')
    if args.clip_reward:
        print('### ✂️ CLIPPING REWARDS ✂️ ###')
    if args.use_alignment:
        print('### Using alignment ###')
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if device == torch.device('cuda'):
        print("### USING CUDA ###")
    else:
        print("### USING CPU ###")

    envs = make_env(args.env_id, args.seed, 0, args.capture_video, run_name, args.clip_reward)()

    agent = SeperateLstmSumAlignableAgent(envs, device, args.use_attention).to(device)
    print("# USING SeperateLstmSumAlignableAgent Agent #")
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_lstm_state = ((
        torch.zeros(agent.video_lstm.num_layers, args.num_envs, agent.video_lstm.hidden_size).to(device),
        torch.zeros(agent.video_lstm.num_layers, args.num_envs, agent.video_lstm.hidden_size).to(device)),
        (torch.zeros(agent.audio_lstm.num_layers, args.num_envs, agent.audio_lstm.hidden_size).to(device),
        torch.zeros(agent.audio_lstm.num_layers, args.num_envs, agent.audio_lstm.hidden_size).to(device),
    ))  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    num_updates = args.total_timesteps // args.batch_size

    episode_rewards = np.zeros(MAX_EPISODE_LEN)
    episode = 0
    
    for update in range(1, num_updates + 1):
        initial_lstm_state = ((next_lstm_state[0][0].clone(), next_lstm_state[0][1].clone()), (next_lstm_state[1][0].clone(), next_lstm_state[1][1].clone()))
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state, _ = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            # 2 remarks: I don't use vector env so have to manually reset, lstm is reset in get_action_and_value
            if done:
                next_obs = envs.reset()
            info = [info]  # original implementation had many envs
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            episode_rewards[step] = reward
            next_obs, next_done = torch.Tensor(next_obs).to(
                device), torch.Tensor((done,)).to(device)

            if done:
                episode_reward = episode_rewards.sum()
                episode_rewards = np.zeros(MAX_EPISODE_LEN)
                episode += 1
                # print(
                #     f"episode #{episode}, global_step={global_step}, episodic_return={episode_reward}")
                writer.add_scalar("charts/episodic_reward",
                                  episode_reward, global_step)
                writer.add_scalar("charts/episodic_length", step, global_step)


            for item in info:
                if "episode" in item.keys():
                    if not (episode % args.print_interval) and episode:
                        print(
                            f"episode #{episode}, global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return_old", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)): 
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _, modality_features = agent.get_action_and_value(
                    b_obs[mb_inds],
                    ((initial_lstm_state[0][0][:, mbenvinds], initial_lstm_state[0][1][:, mbenvinds]), (
                        initial_lstm_state[1][0][:, mbenvinds], initial_lstm_state[1][1][:, mbenvinds])),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                
                # temporal discrimination loss
                if args.use_alignment:
                    alignment_loss = cosine_similarity(modality_features[0], ALIGNMENT_VECTOR1, dim=-1).sum(
                    ) + cosine_similarity(modality_features[1], ALIGNMENT_VECTOR2, dim=-1).sum()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef - alignment_loss * args.alignment_coef
                else:
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if not (episode % (args.print_interval * 20)) and episode:
            print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
