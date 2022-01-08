import torch
from torch import nn
import copy
import numpy as np
from env.env import BreakoutEnv, MultiStepExecutingEvironmentWrapper, BreakoutMultiStepExecutingEnvironmentWrapper
from functools import reduce
from video_renderer import VideoTrajectoryRenderer
from env.renderer import BreakoutRenderer
import wandb

with open('wandb-key.txt') as f:
    key = str(f.read())

wandb.login(key=key)
wandb.init(project='q-learning', entity='claushofmann')

config = wandb.config

config.lr = 0.00025
config.replay_memory_size = 1000000
config.no_steps = 10000000
config.gamma = 0.99


class DQN(nn.Module):
    def __init__(self, obs_shape, action_size):
        super(DQN, self).__init__()

        self.obs_shape = obs_shape
        self.obs_size = reduce(lambda a,b: a*b, obs_shape)
        self.action_size = action_size
        self.input_size = self.obs_size


        self.network = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, self.input_size // 2),
            nn.ReLU(),
            nn.Linear(self.input_size // 2, self.action_size),
        )

    def forward(self, observation:torch.Tensor, batch_dims=1):
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


def deep_q_learning(env:BreakoutEnv, replay_memory_size, total_steps, gamma):
    device = torch.device('cuda:0')
    replay_memory = ReplayMemory(replay_memory_size)
    action_size = env.get_action_size()
    observation_size = env.get_observation_size()
    dqn = DQN(observation_size, action_size).to(device)
    wandb.watch(dqn)
    dqn_old = copy.deepcopy(dqn)
    optimizer = torch.optim.RMSprop(dqn.parameters(), config.lr)
    criterion = nn.MSELoss()
    epsilon = 1

    current_steps = 0

    while True:
        observation = env.reset().get_observation()
        observation = observation.reshape((1,) + observation.shape)
        observation = torch.from_numpy(observation).float()
        done = False
        rewards = []
        losses = []
        while not done:
            if epsilon > 0.1 and len(replay_memory) > 50000:
                epsilon -= 0.9 / total_steps
            if np.random.uniform(0,1) < epsilon:
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
                records = replay_memory.sample_records(128)  # TODO 32?
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
                torch.nn.utils.clip_grad_norm_(dqn.parameters(), 1.)
                optimizer.step()
                losses.append(loss.cpu().detach())

        print('Episode reward: {}, loss: {}, Epsilon: {}'.format(np.sum(rewards), np.mean(losses), epsilon))
        wandb.log({'reward': np.sum(rewards), 'loss': np.mean(losses), 'epsilon': epsilon})

        last_saved = 0
        if total_steps - last_saved > 50000:
            torch.save(dqn.state_dict(), 'dqn.model')
            last_saved = total_steps

        if current_steps >= total_steps:
            break


def main():
    #env = BreakoutEnv(5, time_coef=0.05)
    env = BreakoutMultiStepExecutingEnvironmentWrapper(5, 0.05, no_steps=8)
    deep_q_learning(env, config.replay_memory_size, config.no_steps, config.gamma)
    #dqn = DQN(env.get_observation_size(), env.get_action_size())
    #dqn.load_state_dict(torch.load('dqn.model'))
    #render_episode(env, dqn, 0.1, fps=30)


def render_episode(env: BreakoutEnv, dqn, epsilon, fps=30):
    state = env.reset()
    observation = state.get_observation()
    state_record = [state]
    while True:
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(env.get_action_size())
        else:
            action_scores = dqn(torch.from_numpy(observation).reshape(1,-1).float())
            action = torch.argmax(action_scores)
        state, reward, done = env.step(action)
        observation = state.get_observation()
        state_record.append(state)
        if done:
            break

    renderer = VideoTrajectoryRenderer(state_record, BreakoutRenderer(env), fps=fps)
    renderer.render()


if __name__ == '__main__':
    main()


