import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import TD3Policy, Actor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
import json
from datetime import datetime
from torch.autograd import Function

# ==== 1. 目标域环境 Wrapper（加重质量） ====
class MassAdjustedHalfCheetah(gym.Wrapper):
    def __init__(self, env, mass_scale=1.5):
        super().__init__(env)
        for body_id in range(self.unwrapped.model.nbody):
            self.unwrapped.model.body_mass[body_id] *= mass_scale

# ==== 2. 梯度反转层（GRL） ====
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class GradientReversal(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambd)

# ==== 3. 自定义 Actor with Feature Extractor ====
class DANNActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.action_head = nn.Linear(128, action_dim)

    def forward(self, x):
        features = self.feature(x)
        action = torch.tanh(self.action_head(features))
        return action, features

# ==== 4. 域判别器 ====
class DomainClassifier(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ==== 5. 训练入口 ====
def make_env(source=True):
    env = gym.make("HalfCheetah-v5")
    if not source:
        env = MassAdjustedHalfCheetah(env, mass_scale=1.5)
    return Monitor(env)

if __name__ == "__main__":
    # ==== 配置 ====
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/td3_dann_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== 环境 ====
    source_env = DummyVecEnv([lambda: make_env(source=True)])
    target_env = DummyVecEnv([lambda: make_env(source=False)])
    obs_dim = source_env.observation_space.shape[0]
    action_dim = source_env.action_space.shape[0]

    # ==== 模型组件 ====
    actor = DANNActor(obs_dim, action_dim).to(device)
    model = TD3("MlpPolicy", source_env, verbose=0)
    critic = model.critic  # TD3 的 critic 网络
    domain_clf = DomainClassifier().to(device)
    grl = GradientReversal().to(device)

    # ==== 优化器 ====
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    domain_optim = torch.optim.Adam(domain_clf.parameters(), lr=1e-3)

    # ==== TD3 模型（只训练 critic） ====
    model = TD3("MlpPolicy", source_env, verbose=0)
    model.policy.actor = actor  # 替换 actor 网络
    model.policy.actor_target = actor  # 替换目标网络

    # ==== 训练循环 ====
    print("Starting training with DANN...")
    for step in range(1, 100001):
        # === 采样 source ===
        obs, _ = source_env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        action, feat_src = actor(obs_tensor)
        
        # === 采样 target ===
        obs_tar, _ = target_env.reset()
        obs_tar_tensor = torch.tensor(obs_tar, dtype=torch.float32).to(device)
        _, feat_tar = actor(obs_tar_tensor)

        # === 域判别器训练 ===
        feat_all = torch.cat([feat_src, feat_tar], dim=0)
        labels = torch.cat([
            torch.zeros((feat_src.size(0), 1)),
            torch.ones((feat_tar.size(0), 1))
        ], dim=0).to(device)

        domain_preds = domain_clf(grl(feat_all))
        domain_loss = F.binary_cross_entropy_with_logits(domain_preds, labels)

        domain_optim.zero_grad()
        domain_loss.backward()
        domain_optim.step()

        # === Actor 强化学习训练（TD3 本体学习） ===
        model.learn(total_timesteps=10, log_interval=None)

        if step % 1000 == 0:
            print(f"[Step {step}] Domain Loss: {domain_loss.item():.4f}")


    # === 生成保存目录 ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"td3_dann_{timestamp}"
    save_dir = os.path.join("runs", run_name)
    os.makedirs(save_dir, exist_ok=True)

    # === 保存 TD3 模型 ===
    model.save(os.path.join(save_dir, "td3_halfcheetah_dann"))
    print(f"[✔] TD3 model saved at: {save_dir}/td3_halfcheetah_dann.zip")

    # === 保存自定义 actor 和 domain classifier 参数（可单独加载）===
    torch.save(actor.state_dict(), os.path.join(save_dir, "actor.pt"))
    torch.save(domain_clf.state_dict(), os.path.join(save_dir, "domain_classifier.pt"))
    print(f"[✔] Actor and Domain Classifier weights saved.")

    # === 保存训练配置（方便复现实验）===
    config = {
        "env": "HalfCheetah-v5",
        "mass_scale": 1.5,
        "total_steps": 100000,
        "actor_lr": 1e-3,
        "domain_lr": 1e-3,
        "grl_lambda": grl.lambd,
        "obs_dim": obs_dim,
        "action_dim": action_dim
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"[✔] Training config saved.")

    # === 最终提示 ===
    print(f"✅ Training completed. All results saved in: {save_dir}")
