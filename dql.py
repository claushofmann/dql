import torch
from torch import nn
import copy
import numpy as np
from env.env import BreakoutEnv, BreakoutMultiStepExecutingEnvironmentWrapper
from functools import reduce
from video_renderer import VideoTrajectoryRenderer
from env.renderer import BreakoutRenderer
import wandb
import sys

use_wandb = True

if use_wandb:
    with open('wandb-key.txt') as f:
        key = str(f.read())

    wandb.login(key=key)
    wandb.init(project='q-learning', entity='claushofmann')

config = wandb.config
config.lr = 0.00025
config.replay_memory_size = 1000000
config.no_steps_to_min_eps = 1000000
config.no_steps = 10000000
config.min_replay_steps = 50000
config.gamma = 0.99
config.min_epsilon = 0.1
config.clip = 1.
config.batch_size = 32
config.no_rewards_for_mean = 10000
config.breakout_no_steps = 5
config.breakout_size = 5
config.breakout_time_coef = 0.05
config.l1 = 152
config.l2 = 130


class DQN(nn.Module):
    def __init__(self, obs_shape, action_size):
        super(DQN, self).__init__()

        self.obs_shape = obs_shape
        self.obs_size = reduce(lambda a,b: a*b, obs_shape)
        self.action_size = action_size
        self.input_size = self.obs_size

        self.network = nn.Sequential(
            nn.Linear(self.input_size, config.l1),
            nn.ReLU(),
            nn.Linear(config.l1, config.l2),
            nn.ReLU(),
            nn.Linear(config.l2, self.action_size),
        )

    def forward(self, observation: torch.Tensor, batch_dims=1):
        observation = torch.reshape(observation, observation.shape[:batch_dims] + (-1,))
        return self.network(observation)


class StepRecord:
    def __init__(self, old_observation, new_observation, action, reward, done):
        self.old_observation = old_observation
        self.new_observation = new_observation
        self.action = action
        self.reward = reward
        self.done = done


class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.records = []
        self.oldest_record = 0

    def store_record(self, record: StepRecord):
        if len(self.records) < self.max_size:
            self.records.append(record)
        else:
            self.records[self.oldest_record] = record
            self.oldest_record = self.oldest_record + 1
            if self.oldest_record >= len(self.records):
                self.oldest_record = 0

    def sample_records(self, no_samples):
        record_idxs = np.random.choice(len(self.records), [no_samples]).astype(int)
        return [self.records[idx] for idx in record_idxs]

    def __len__(self):
        return len(self.records)


def deep_q_learning(env: BreakoutEnv, replay_memory_size, total_steps, gamma):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    replay_memory = ReplayMemory(replay_memory_size)
    action_size = env.get_action_size()
    observation_size = env.get_observation_size()
    dqn = DQN(observation_size, action_size).to(device)
    if use_wandb:
        wandb.watch(dqn)
    dqn_old = copy.deepcopy(dqn)
    optimizer = torch.optim.RMSprop(dqn.parameters(), config.lr)
    criterion = nn.MSELoss()
    epsilon = 1

    current_steps = 0
    last_saved = 0

    rewards_for_mean = []

    while True:
        observation = env.reset().get_observation()
        observation = observation.reshape((1,) + observation.shape)
        observation = torch.from_numpy(observation).float()
        done = False
        rewards = []
        losses = []
        while not done:
            if epsilon > config.min_epsilon and len(replay_memory) > config.min_replay_steps:
                epsilon -= 0.9 / config.no_steps_to_min_eps
            if np.random.uniform(0, 1) < epsilon:
                # select random action
                selected_action = torch.from_numpy(np.random.choice(action_size, [observation.shape[0]]))
            else:
                # select greedy action
                action_scores = dqn(observation.to(device))
                selected_action = torch.argmax(action_scores, dim=-1).cpu().detach()
            old_observation = observation
            observation, reward, done = env.step(selected_action)
            observation = observation.get_observation()
            observation = observation.reshape((1,) + observation.shape)
            observation = torch.from_numpy(observation).float()
            rewards.append(reward)
            replay_memory.store_record(StepRecord(old_observation.detach(), observation.detach(), selected_action.detach(), reward, done))

            current_steps += 1

            if current_steps % 2000 == 0:
                dqn_old = copy.deepcopy(dqn)

            if len(replay_memory) > 50000:
                records = replay_memory.sample_records(config.batch_size)
                old_observations = torch.cat([record.old_observation for record in records], dim=0).detach().to(device)
                new_observations = torch.cat([record.new_observation for record in records], dim=0).detach().to(device)
                actions = torch.cat([record.action for record in records], dim=0).long().detach().to(device)
                rewards_tensor = torch.Tensor([record.reward for record in records]).reshape(-1).detach().to(device)
                dones = torch.Tensor([record.done for record in records]).reshape(-1).float().detach().to(device)

                old_q_values = dqn(old_observations)[range(len(actions)), actions].reshape(-1)
                new_q_values, _ = torch.max(dqn_old(new_observations), dim=-1, keepdim=False)

                target = rewards_tensor + gamma * new_q_values * (1 - dones)

                loss = criterion(old_q_values, target.detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dqn.parameters(), config.clip)
                optimizer.step()
                losses.append(loss.cpu().detach())

        if len(rewards_for_mean) >= config.no_rewards_for_mean:
            rewards_for_mean.pop(0)
        rewards_for_mean.append(np.sum(rewards))

        print('Episode reward: {}, loss: {}, Epsilon: {}'.format(np.sum(rewards), np.mean(losses), epsilon))
        if use_wandb:
            wandb.log({'reward': np.mean(rewards_for_mean), 'loss': np.mean(losses), 'epsilon': epsilon, 'current steps': current_steps})

        if current_steps - last_saved > 50000:
            torch.save(dqn.state_dict(), f'models/{wandb.run.name}')
            print(f'model saved: {wandb.run.name}')
            # wandb.save(f'models/{wandb.run.id}')
            last_saved = current_steps

        if current_steps >= config.total_steps:
            torch.save(dqn.state_dict(), f'models/{wandb.run.name}')
            print(f'model saved: {wandb.run.name}')
            break


def main():
    if config.breakout_no_steps == 1:
        env = BreakoutEnv(config.breakout_size, time_coef=config.breakout_time_coef)
    else:
        env = BreakoutMultiStepExecutingEnvironmentWrapper(config.breakout_size, config.breakout_time_coef, no_steps=config.breakout_no_steps)
    print(config.l1)
    deep_q_learning(env, config.replay_memory_size, config.no_steps, config.gamma)


def play():
    env = BreakoutEnv(config.breakout_size, time_coef=config.breakout_time_coef)
    dqn = DQN(env.get_observation_size(), env.get_action_size())
    dqn.load_state_dict(torch.load('dqn.model'))
    render_episode(env, dqn, 0.1, fps=30)


def render_episode(env: BreakoutEnv, dqn, epsilon, fps=30):
    state = env.reset()
    observation = state.get_observation()
    state_record = [state]
    while True:
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(env.get_action_size())
        else:
            observation = observation.reshape((1,) + observation.shape)
            observation = torch.from_numpy(observation).float()
            action_scores = dqn(observation)
            action = torch.argmax(action_scores, dim=1)
        state, reward, done = env.step(action)
        observation = state.get_observation()
        state_record.append(state)
        if done:
            break

    renderer = VideoTrajectoryRenderer(state_record, BreakoutRenderer(env), fps=fps)
    renderer.render()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'play':
        play()
    else:
        main()
