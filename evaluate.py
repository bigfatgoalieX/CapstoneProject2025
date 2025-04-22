import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
import argparse

# 修改目标环境
class TargetDomainWrapper(gym.Wrapper):
    def __init__(self, env, 
        gravity_scale=1.0, 
        friction_scale=1.0, 
        mass_scale=1.0,
        obs_noise_std=0.0, 
        action_noise_std=0.0
    ):
        super().__init__(env)
        self.gravity_scale = gravity_scale
        self.friction_scale = friction_scale
        self.mass_scale = mass_scale
        self.obs_noise_std = obs_noise_std
        self.action_noise_std = action_noise_std
        self.modified = False

    def reset(self, **kwargs):
        if not self.modified:
            self.modify_physics()
            self.modified = True
        result = self.env.reset(**kwargs)
        # 兼容gymnasium的新格式
        if isinstance(result, tuple):
            obs, info = result
            obs = self.add_obs_noise(obs)
            return obs, info
        # 旧版本gym格式
        else:
            obs = self.add_obs_noise(result)
            return obs
        
    def step(self, action):
        # 添加动作噪声
        noisy_action = self.add_action_noise(action)
        obs, reward, terminated, truncated, info = self.env.step(noisy_action)
        # 添加观测噪声
        return self.add_obs_noise(obs), reward, terminated, truncated, info

    def modify_physics(self):
        model = self.env.unwrapped.model
        model.opt.gravity[2] *= self.gravity_scale  # 重力
        model.geom_friction[:, 0] *= self.friction_scale #摩擦
        model.body_mass[:] *= self.mass_scale  # 质量
    
    def add_obs_noise(self, obs):
        # 兼容gymnasium的新格式
        if isinstance(obs, tuple):
            obs = obs[0]
        if self.obs_noise_std > 0.0:
            noise = np.random.normal(0, self.obs_noise_std, size=obs.shape)
            return obs + noise
        return obs
    
    def add_action_noise(self, action):
        # 兼容gymnasium的新格式
        if isinstance(action, tuple):
            action = action[0]
        if self.action_noise_std > 0.0:
            noise = np.random.normal(0, self.action_noise_std, size=action.shape)
            noisy_action = action + noise
            # clip action to valid range
            return np.clip(noisy_action, self.action_space.low, self.action_space.high)
        return action
    

def evaluate_model(model_path, env, episodes=10):
    model = TD3.load(model_path)
    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        episode_reward = 0
        step = 0
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

        rewards.append(episode_reward)
        # print(f"Episode {ep+1}: Reward = {episode_reward}")

    # 计算多个episodes的平均奖励
    avg_reward = np.mean(rewards)
    # print(f"Average reward over {episodes} episodes: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gravity",type = float, default = 1.0)
    parser.add_argument("--friction",type = float, default = 1.0)
    parser.add_argument("--mass",type = float, default = 1.0)
    parser.add_argument("--obs_noise",type = float, default = 0.0)
    parser.add_argument("--action_noise",type = float, default = 0.0)
    args = parser.parse_args()
    
    # 设置目标域环境
    env = gym.make("HalfCheetah-v5")
    env = TargetDomainWrapper(
        env,
        gravity_scale = args.gravity,
        friction_scale = args.friction,
        mass_scale = args.mass,
        obs_noise_std = args.obs_noise,
        action_noise_std = args.action_noise
        )

    models = {
        "baseline": "./runs/td3_baseline_2025-04-10_00-53-48/td3_halfcheetah_baseline.zip",
        "dr": "./runs/td3_dr_2025-04-10_11-30-35/td3_halfcheetah_dr.zip"
    }

    for name, path in models.items():
        print(f"\nEvaluating {name} model...")
        reward = evaluate_model(path, env)
        print(f"{name} Average Reward: {reward}")

