# **********************************************************************************************************************
# **********************************************************************************************************************
# ***                          Using Reinforcement Learning for Load Testing of Video Games                          ***
# ***                                                 Game: MsPacman                                                 ***
# ***                                     RELINE: DQN model + injected bugs info                                     ***
# ***                                       Training for 1000 + 1000 episodes                                        ***
# **********************************************************************************************************************
# **********************************************************************************************************************


from lib import dqn_model
from lib import wrappers
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import collections
import cv2
import datetime
import gym
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random


DEFAULT_ENV_NAME = "MsPacmanNoFrameskip-v4"
MEAN_REWARD_BOUND = 400
MAX_ITERATIONS = 10000

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 1500000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

# experience unit : state, action -> new_state, reward, done or not
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


# Experience "container" with a fixed capacity (i.e. max experiences)
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    # add experience
    def append(self, experience):
        self.buffer.append(experience)

    # provide a random batch of the experience
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()
        self.count_total_moves = 0
        self.count_random_moves = 0
        self.bug_flags = [False, False]
    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0
        self.count_total_moves = 0
        self.count_random_moves = 0
        self.bug_flags = [False, False]
        for i in range(65):
            state, reward, done, _ = env.step(0)

    @staticmethod
    def check_bug1():
        folder_bug = 'bug_left/'
        files = os.listdir(folder_bug)
        img_bug = [file for file in files if file.startswith('bug')]
        img = Image.open("current_screen.png")
        left = 0
        top = 90
        right = 15
        bottom = 120
        im1 = img.crop((left, top, right, bottom))
        im1.save('current_test.png')
        imgA = cv2.imread("current_test.png")
        for elem in img_bug:
            imgB = cv2.imread(folder_bug + elem)
            if imgA.shape != imgB.shape:
                imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))
            s = ssim(imgA, imgB, multichannel=True)
            if s > 0.9:
                print(s)
                return True
        return False

    @staticmethod
    def check_bug2():
        folder_bug = 'bug_right/'
        files = os.listdir(folder_bug)
        img_bug = [file for file in files if file.startswith('bug')]
        img = Image.open("current_screen.png")
        left = 305
        top = 42
        right = 320
        bottom = 72
        im1 = img.crop((left, top, right, bottom))
        im1.save('current_test.png')
        imgA = cv2.imread("current_test.png")
        for elem in img_bug:
            imgB = cv2.imread(folder_bug + elem)
            if imgA.shape != imgB.shape:
                imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))
            s = ssim(imgA, imgB, multichannel=True)
            if s > 0.9:
                print(s)
                return True
        return False

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cuda"):
        done_reward = None

        if np.random.random() < epsilon:
            action = random.randint(0, 4)
            self.count_random_moves += 1
            self.count_total_moves += 1
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
            self.count_total_moves += 1

        new_state, reward, is_done, _ = self.env.step(action)

        self.env.env.ale.saveScreenPNG('current_screen.png')
        if not self.bug_flags[0] and self.check_bug1():
            self.bug_flags[0] = True
            reward += 50
        if not self.bug_flags[1] and self.check_bug2():
            self.bug_flags[1] = True
            reward += 50

        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            with open('bug_log_RELINE.txt', 'a+') as f:
                if self.bug_flags[0]: f.write('BUG1 ')
                if self.bug_flags[1]: f.write('BUG2 ')
                f.write('\n')
            done_reward = self.total_reward
            print('tot random moves: %d / %d (%.2f %s) with epsilon: %.2f' % (
                self.count_random_moves, self.count_total_moves,
                (self.count_random_moves * 100 / self.count_total_moves), '%', epsilon))
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cuda"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


class EvalAgent:
    def __init__(self, env):
        self.env = env
        self.state = None
        self.total_reward = 0.0

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0
        for _ in range(65):
            self.state, _, _, _ = self.env.step(0)

    @torch.no_grad()
    def play_episode(self, net, device="cuda"):
        self._reset()
        done = False

        while not done:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

            next_state, reward, done, _ = self.env.step(action)
            self.total_reward += reward
            self.state = next_state

        return self.total_reward



if __name__ == "__main__":
    print('\n\n***********************************************************')
    print("* RELINE model's training on MsPacman game is starting... *")
    print('***********************************************************\n')

    device = "cuda"
    print("✅ Using device:", device)
    env = wrappers.make_env(DEFAULT_ENV_NAME)
    num_actions = 5
    net = dqn_model.DQN(env.observation_space.shape, num_actions).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, num_actions).to(device)
    prev_best_reward = -float('inf')
    if os.path.exists(DEFAULT_ENV_NAME + "-best-RE.dat"):
        print("🔍 Previous best model found. Evaluating...")
        net.load_state_dict(torch.load(DEFAULT_ENV_NAME + "-best-RE.dat", map_location=device))
        net.eval()
        # 평가 환경 생성
        eval_env = wrappers.make_env(DEFAULT_ENV_NAME)
        eval_agent = EvalAgent(eval_env)  # 학습 없이 평가만
        eval_rewards = []
        for _ in range(10):  # 빠르게 평균 보상 측정
            r = 0
            eval_agent._reset()
            done = False
            while not done:
                state_a = np.array([eval_agent.state], copy=False)
                state_v = torch.tensor(state_a).to(device)
                q_vals_v = net(state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
                action = int(act_v.item())
                state, reward, done, _ = eval_env.step(action)
                r += reward
                eval_agent.state = state
            eval_rewards.append(r)
        prev_best_reward = np.mean(eval_rewards)
        print(f"✅ Previous best model mean reward: {prev_best_reward:.2f}")
        eval_env.close()

    
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None
    recent_means = []

    open('bug_log_RELINE.txt', 'w').close()

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, epsilon - 1 / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = float(np.mean(total_rewards[-100:]))

            recent_means.append(mean_reward)
            if len(recent_means) > 10:
                recent_means.pop(0)
                delta = max(recent_means) - min(recent_means)
                if delta < mean_reward * 0.03 and epsilon < 0.3:
                    epsilon = min(0.3, epsilon + 0.01)
                    print("보상이 정체 상태입니다. epsilon 값을 일시적으로 증가")

            print("frames: %d, episodes: %d , mean reward: %.3f, eps: %.2f, speed: %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon, speed))

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best-RE.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if len(total_rewards) == MAX_ITERATIONS:
                print("training ends")
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
            torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-last-RE.dat")
            print('Target net update at frame: %d , games: %d' % (frame_idx, len(total_rewards)))

        if frame_idx == REPLAY_SIZE:
            print('Experience replay buffer full at frame: %d , games: %d' % (frame_idx, len(total_rewards)))

        if frame_idx % REPLAY_SIZE == 0 and frame_idx > REPLAY_SIZE:
            print('Experience replay buffer refilled with new experiences at frame: %d , games: %d'
                  % (frame_idx, len(total_rewards)))

        if frame_idx == REPLAY_START_SIZE:
            print('Training starts at frame: %d , games: %d' % (frame_idx, len(total_rewards)))

        if frame_idx == EPSILON_DECAY_LAST_FRAME:
            print('Epsilon reaches the minimum value at frame: %d , games: %d' % (frame_idx, len(total_rewards)))
            tgt_net.load_state_dict(net.state_dict())
            torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-last-RE.dat")

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()

    env.close()
    # ✅ 학습 종료 후 best 모델 비교
    if best_mean_reward > prev_best_reward:
        print(f"🏆 New best model outperforms previous! {best_mean_reward:.2f} > {prev_best_reward:.2f}")
        torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best-RE.dat")
    else:
        print(f"📉 Best model retained from previous. {best_mean_reward:.2f} <= {prev_best_reward:.2f}")



    lines = [line for line in open('bug_log_RELINE.txt', 'r')]
    lines_1k = lines[-1000:]

    count_0bug = 0
    count_1bug = 0
    count_2bug = 0

    for line in lines_1k:
        if line.strip() == '':
            count_0bug += 1
        elif len(line.strip().split()) == 1:
            count_1bug += 1
        elif len(line.strip().split()) == 2:
            count_2bug += 1

    print('\nReport injected bugs spotted during last 1000 episodes:')
    print('0 injected bug spotted in %d episodes' % count_0bug)
    print('1 injected bug spotted in %d episodes' % count_1bug)
    print('2 injected bugs spotted in %d episodes' % count_2bug)
    print("\    /\ \n )  ( ')  meow!\n(  /  )\n \(__)|")
