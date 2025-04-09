import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os
from datetime import datetime
import numpy as np

class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_body_mass = np.copy(self.unwrapped.model.body_mass)
        self.original_geom_friction = np.copy(self.unwrapped.model.geom_friction)
        self.original_joint_friction = np.copy(self.unwrapped.model.dof_joint_friction)

    def reset(self, **kwargs):
        # ===== 质量随机化 =====
        self.unwrapped.model.body_mass[:] = self.original_body_mass * np.random.uniform(0.8, 1.2, size=self.original_body_mass.shape)

        # ===== 摩擦系数随机化 =====
        self.unwrapped.model.geom_friction[:] = self.original_geom_friction * np.random.uniform(0.5, 1.5, size=self.original_geom_friction.shape)

        # ===== 关节摩擦随机化 =====
        # self.unwrapped.model.dof_joint_friction[:] = self.original_joint_friction * np.random.uniform(0.5, 1.5, size=self.original_joint_friction.shape)

        # ===== 观测噪声（随机噪声扰动观测值） =====
        # 如果是图像观测，可以在这里加入图像噪声；对于数值型观测，可以加点噪声。
        noise = np.random.normal(0, 0.1, size=self.observation_space.shape)
        obs, _ = self.env.reset(**kwargs)
        obs += noise  # 加入噪声
        return obs

    def step(self, action):
        # ===== 动作噪声（在执行动作时加入噪声） =====
        noise = np.random.normal(0, 0.05, size=action.shape)  # 加一些扰动
        action += noise
        return self.env.step(action)

def make_env():
    env = gym.make("HalfCheetah-v5")
    env = DomainRandomizationWrapper(env)
    env = Monitor(env)
    return env

if __name__ == "__main__":
    # ===== 创建时间戳路径 =====
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"td3_dr_{timestamp}"
    save_dir = os.path.join("runs", run_name)
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ===== 创建训练环境 =====
    env = DummyVecEnv([make_env])

    # ===== 创建 TD3 模型 =====
    model = TD3(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=os.path.join(save_dir, "tensorboard")
    )

    # ===== 设置 checkpoint 回调函数 =====
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="td3_halfcheetah_dr"
    )

    # ===== 开始训练 =====
    print(f"Training domain-randomized model... (run: {run_name})")
    model.learn(total_timesteps=200000, callback=checkpoint_callback)

    # ===== 保存最终模型 =====
    final_model_path = os.path.join(save_dir, "td3_halfcheetah_dr")
    model.save(final_model_path)

    print(f"Domain randomized model saved to: {final_model_path}")
    print("Training complete.")
