import os
import gym
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
from lib import dqn_model, wrappers
import random

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

            # 이미지 크기가 최소 7x7 이상이어야 SSIM 계산 가능
            if imgA.shape[0] < 7 or imgA.shape[1] < 7:
                print("이미지가 너무 작아서 SSIM 계산을 건너뜁니다.")
                continue

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
    def play_episode(self, net, epsilon, device="cuda"):
        self._reset()
        done = False
        bug_log = []

        while not done:
            env.render()
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

            # === epsilon-greedy action selection ===
            if np.random.random() < epsilon:
                action = random.randint(0, 4)
                #print(f"[Random Action] Epsilon={epsilon:.2f} → Action={action}")
            else:
                state_a = np.array([self.state], copy=False)
                state_v = torch.tensor(state_a).to(device)
                q_vals_v = net(state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
                action = int(act_v.item())

            self.state, reward, done, _ = self.env.step(action)
            self.total_reward += reward

        with open("bug_log_eval_RE.txt", "a+") as f:
            f.write(" ".join(bug_log) + "\n")

        return self.total_reward

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = wrappers.make_env(DEFAULT_ENV_NAME)
    net = dqn_model.DQN(env.observation_space.shape, 5).to(device)
    net.load_state_dict(torch.load(DEFAULT_ENV_NAME + "-best-RE.dat", map_location=device))
    net.eval()

    with open("bug_log_eval_RE.txt", "w") as f:
        f.close()

    agent = EvalAgent(env)

    rewards = []
    recent_means = []
    epsilon = 0.1  # 초기 epsilon 값 (테스트용이므로 작게 시작)

    for i in range(1000):
        r = agent.play_episode(net, epsilon, device)
        rewards.append(r)

        # 보상 정체 판단
        recent_means.append(r)
        if len(recent_means) > 10:
            recent_means.pop(0)
            delta = max(recent_means) - min(recent_means)
            if delta < np.mean(recent_means) * 0.01:
                epsilon = min(0.3, epsilon + 0.01)
                print("⚠️ 평가 중 보상 정체 탐지, epsilon 값을 증가시킵니다.")
                print(f"[Random Action] Epsilon: {epsilon:.3f}")

        print(f"Episode {i+1} reward: {r}")

    print(f"\nEvaluation complete. Average reward: {np.mean(rewards):.2f}")
    env.close()

    # 통계 출력
    lines = [line.strip() for line in open("bug_log_eval_RE.txt", "r")]
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
