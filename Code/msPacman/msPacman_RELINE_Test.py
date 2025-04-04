import os
import gym
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
from lib import dqn_model, wrappers

DEFAULT_ENV_NAME = "MsPacmanNoFrameskip-v4"

class EvalAgent:
    def __init__(self, env):
        self.env = env
        self._reset()
        self.bug_flags = [False, False]
        self.bug_count = 0

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0
        self.bug_flags = [False, False]
        self.bug_count = 0
        for _ in range(65):
            self.state, _, _, _ = self.env.step(0)

    @staticmethod
    def check_bug(folder, left, top, right, bottom):
        files = os.listdir(folder)
        img_bug = [file for file in files if file.startswith('bug')]
        img = Image.open("current_screen.png")
        im1 = img.crop((left, top, right, bottom))
        im1.save('current_test.png')
        imgA = cv2.imread("current_test.png")
        for elem in img_bug:
            imgB = cv2.imread(os.path.join(folder, elem))
            if imgA.shape != imgB.shape:
                imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))
            s = ssim(imgA, imgB, multichannel=True)
            if s > 0.9:
                print(f"SSIM: {s:.3f}")
                return True
        return False


    def check_bug1(self):
        return self.check_bug('bug_right/', 305, 90, 320, 120)

    def check_bug2(self):
        return self.check_bug('bug_left/', 0, 42, 15, 72)
    
    @torch.no_grad()
    def play_episode(self, net, device="cuda"):
        self._reset()
        done = False
        bug_log = []
        epsilon = 0.1  # ← 여기 추가

        while not done:
            self.env.env.ale.saveScreenPNG('current_screen.png')

            if not self.bug_flags[0] and self.check_bug1():
                self.bug_flags[0] = True
                self.total_reward += 50
                self.bug_count += 1
                bug_log.append("BUG1")

            if not self.bug_flags[1] and self.check_bug2():
                self.bug_flags[1] = True
                self.total_reward += 50
                self.bug_count += 1
                bug_log.append("BUG2")

            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)

            if np.random.random() < epsilon:
                action = self.env.action_space.sample()  # ← 환경에서 무작위 행동 선택
            else:
                q_vals_v = net(state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
                action = int(act_v.item())

            self.state, reward, done, _ = self.env.step(action)
            self.total_reward += reward

        with open("bug_log_eval_RELINE.txt", "a") as f:
            f.write(" ".join(bug_log) + "\n")

        return self.total_reward


if __name__ == "__main__":
    open("bug_log_eval_RELINE.txt", "w").close()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = wrappers.make_env(DEFAULT_ENV_NAME)
    net = dqn_model.DQN(env.observation_space.shape, 5).to(device)
    net.load_state_dict(torch.load(DEFAULT_ENV_NAME + "-best.dat", map_location=device))
    net.eval()

    agent = EvalAgent(env)

    rewards = []
    for i in range(1000):
        r = agent.play_episode(net, device)
        rewards.append(r)
        print(f"Episode {i+1} reward: {r}")

    print(f"\nEvaluation complete. Average reward: {np.mean(rewards):.2f}")
    env.close()

        # 통계 출력
    lines = [line.strip() for line in open("bug_log_eval_RELINE.txt", "r")]
    count_0bug = 0
    count_bug1 = 0
    count_bug2 = 0
    count_both = 0

    for line in lines:
        if line == "":
            count_0bug += 1
        elif line == "BUG1":
            count_bug1 += 1
        elif line == "BUG2":
            count_bug2 += 1
        elif "BUG1" in line and "BUG2" in line:
            count_both += 1

    print("\n📊 Bug Detection Summary (during Evaluation)")
    print(f" - No bug detected     : {count_0bug} episodes")
    print(f" - Only BUG1 detected  : {count_bug1 + count_bug2} episodes")
    print(f" - Both BUG1 & BUG2    : {count_both} episodes")
