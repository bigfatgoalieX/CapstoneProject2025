import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def make_env():
    # 创建标准环境
    env = gym.make("HalfCheetah-v5")
    env = Monitor(env)  # 记录每个 episode 的奖励
    return env

if __name__ == "__main__":
    # 创建训练环境
    env = DummyVecEnv([make_env])  # 使用 DummyVecEnv 来包装环境

    # 创建 TD3 模型
    model = TD3("MlpPolicy", env, verbose=1, tensorboard_log="./td3_baseline_logs")

    # 创建 checkpoint 回调函数：每 100,000 steps 保存一次模型
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models_checkpoints/",
        name_prefix="td3_halfcheetah_baseline"
    )

    # 开始训练
    print("Training baseline model without Domain Randomization...")
    model.learn(total_timesteps=100000, callback=checkpoint_callback)

    # 最终保存训练好的模型
    model.save("./models/td3_halfcheetah_baseline")

    print("Baseline model training complete and saved.")
