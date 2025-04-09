import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os
from datetime import datetime

def make_env():
    # 创建标准环境
    env = gym.make("HalfCheetah-v5")
    env = Monitor(env)  # 记录每个 episode 的奖励
    return env

if __name__ == "__main__":
    # ===== 创建时间戳 =====
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"td3_baseline_{timestamp}"
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
        name_prefix="td3_halfcheetah_baseline"
    )

    # ===== 开始训练 =====
    print(f"Training baseline model... (run: {run_name})")
    model.learn(total_timesteps=200000, callback=checkpoint_callback)

    # ===== 保存最终模型 =====
    final_model_path = os.path.join(save_dir, "td3_halfcheetah_baseline")
    model.save(final_model_path)

    print(f"Model saved to: {final_model_path}")
    print("Training complete.")
