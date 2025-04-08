import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3

# 目标环境（添加扰动）
class TargetDomainWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.modified = False

    def reset(self, **kwargs):
        if not self.modified:
            self.modify_physics()
            self.modified = True
        return self.env.reset(**kwargs)

    def modify_physics(self):
        model = self.env.unwrapped.model
        # 加强重力
        model.opt.gravity[2] = -14.8  # 原为 -9.8 或 -9.81
        # 增加摩擦
        model.geom_friction[:, 0] *= 1.0  # 原为 1.0
        # 增加质量
        model.body_mass[:] *= 1.2

def evaluate_model(model_path, env, episodes=10, render=False):
    model = TD3.load(model_path)
    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        episode_reward = 0
        step = 0
        
        while not (terminated or truncated):
            # 可选开启渲染
            if render:                    
                env.render()
            # 通过模型预测action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

        rewards.append(episode_reward)
        print(f"Episode {ep+1}: Reward = {episode_reward}")

    # 计算多个episodes的平均奖励
    avg_reward = np.mean(rewards)
    print(f"Average reward over {episodes} episodes: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    # 设置目标域环境
    target_env = gym.make("HalfCheetah-v5", render_mode="human")
    target_env = TargetDomainWrapper(target_env)

    # 加载模型并评估
    print("Start evaluating...")
    model_path = "./models_checkpoints/td3_halfcheetah_baseline_20000_steps.zip"
    reward = evaluate_model(model_path, target_env, render=True)
    # print("Evaluating baseline model (no DR)...")
    # baseline_model_path = "./models/td3_halfcheetah_baseline.zip"
    # baseline_reward = evaluate_model(baseline_model_path, target_env)

    # print("\nEvaluating DR model (with domain randomization)...")
    # dr_model_path = "./models/td3_halfcheetah_dr.zip"
    # dr_reward = evaluate_model(dr_model_path, target_env)

    # 输出对比结果
    print("\nEvaluation Results:")
    print(f"Model Average Reward: {reward}")
    # print(f"Baseline Model Average Reward: {baseline_reward}")
    # print(f"DR Model Average Reward: {dr_reward}")
