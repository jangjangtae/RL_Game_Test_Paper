# **********************************************************************************************************************
# **********************************************************************************************************************
# **********************************************************************************************************************
# ***                          Using Reinforcement Learning for Load Testing of Video Games                          ***
# ***                                                 Game: CartPole                                                 ***
# ***                                RELINE: Cross Entropy Method + info injected bugs                               ***
# ***                       Play 1000 episodes (still training) and save injected bugs spotted                       ***
# **********************************************************************************************************************
# **********************************************************************************************************************
# **********************************************************************************************************************


import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple


HIDDEN_SIZE = 128  # neural network size
BATCH_SIZE = 16    # num episodes
PERCENTILE = 70    # elite episodes
MAX_ITER = 200     # training iterations

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size, file):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    # OBSERVATION:
    # - x coordinate of the stick's center of mass
    # - speed
    # - angle to the platform
    # - angular speed
    sm = nn.Softmax(dim=1)
    flag_injected_bug_spotted = [False, False, False, False]
    #bug_index = 0
    while True:

        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v)) / temperature
        act_probs = act_probs_v.data.numpy()[0]
        act_probs = act_probs / np.sum(act_probs) 
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)

        #env.render()
        if -0.55 < next_obs[0] < -0.5 and not flag_injected_bug_spotted[0]:
            reward += 50
            file.write('BUG1 ')
            flag_injected_bug_spotted[0] = True

        if 0.5 < next_obs[0] < 0.55 and not flag_injected_bug_spotted[1]:

            reward += 50
            file.write('BUG2 ')
            flag_injected_bug_spotted[1] = True
        
        if -1.0 < next_obs[0] < -0.95 and not flag_injected_bug_spotted[2]:
            reward += 50
            file.write('BUG3 ')
            flag_injected_bug_spotted[2] = True

        if 0.95 < next_obs[0] < 1.0 and not flag_injected_bug_spotted[3]:
            reward += 50
            file.write('BUG4 ')
            flag_injected_bug_spotted[3] = True

        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            file.write('\n')
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            flag_injected_bug_spotted = [False, False, False,False]
            bug_index = 0
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean

# **********************************************************************************************************************
# *                                                1000 episodes start                                                 *
# **********************************************************************************************************************


if __name__ == "__main__":
    print('\n\n*************************************************************')
    print("* RLELINE model's playing 1000 episodes (still training)... *")
    print('*************************************************************\n')
    env = gym.make("CartPole-v0")
    env._max_episode_steps = 1000  # episode length
    temperature = 1.0
    last_rewards = []

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
  #임시로 이름을 RE로 해놓음
    net.load_state_dict(torch.load('./best_model_RE'))
    net.eval()

    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    filename = 'injected_bugs_spotted_RE.txt'
    f = open(filename, 'w+')

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE, f)):
    # 평가만 진행 (학습 없음)
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)  # 보상 기록용은 그대로 사용 가능
        print("%d: reward_mean=%.1f, reward_bound=%.1f" % (iter_no, reward_m, reward_b))

        last_rewards.append(reward_m)
        if len(last_rewards) > 5:
            last_rewards.pop(0)
            base = last_rewards[0]
            deltas = [abs(r - base) / (abs(base) + 1e-8) for r in last_rewards[1:]]

            if all(delta < 0.03 for delta in deltas):
                print("평균 보상 변화율이 3% 미만, softmax 온도 조절합니다.")
                temperature = 2.0
            else:
                temperature = 1.0

        if iter_no == 63:  # 1008 episodes
            print('1k episodes end\n\n')
            break
    
    f.close()

    lines = [line for line in open(filename, 'r')]
    lines_1k = lines[:1000]

    count_0bug = 0
    count_1bug = 0
    count_2bug = 0
    count_3bug = 0
    count_4bug = 0

    for line in lines_1k:
        if line.strip() == '':
            count_0bug += 1
        elif len(line.strip().split()) == 1:
            count_1bug += 1
        elif len(line.strip().split()) == 2:
            count_2bug += 1
        elif len(line.strip().split()) == 3:
            count_3bug += 1
        elif len(line.strip().split()) == 4:
            count_4bug += 1
    print('Report injected bugs spotted:')
    print('0 injected bug spotted in %d episodes' % count_0bug)
    print('1 injected bug spotted in %d episodes' % count_1bug)
    print('2 injected bugs spotted in %d episodes' % count_2bug)
    print('3 injected bug spotted in %d episodes' % count_3bug)
    print('4 injected bugs spotted in %d episodes' % count_4bug)


    print("\    /\ \n )  ( ')  meow!\n(  /  )\n \(__)|")

#                                                                                                               \    /\
#                                                                                                                )  ( ')
#                                                                                                               (  /  )
#                                                                                                                \(__)|
