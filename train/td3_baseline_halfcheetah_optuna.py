import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
import optuna


from evaluate import evaluate_model # 姑且使用evaluate.py中的evaluate_model函数


# 定义环境创建函数
def make_env():
    # 创建标准环境
    env = gym.make("HalfCheetah-v5")
    env = Monitor(env)  # 记录每个 episode 的奖励
    return env


# 定义目标函数，用于评估超参数的效果
def objective(trial):
    # 从 Optuna 搜索空间中选择超参数
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    gamma = trial.suggest_uniform('gamma', 0.9, 0.99)
    tau = trial.suggest_uniform('tau', 0.001, 0.01)

    # 创建训练环境
    env = DummyVecEnv([make_env])

    # 创建 TD3 模型
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        verbose=0,  # 设置为 0 可以避免过多输出
        tensorboard_log=os.path.join(save_dir, "tensorboard")
    )

    # 训练模型
    model.learn(total_timesteps=200000)

    # 假设有一个评估模型的函数，根据最终奖励返回模型性能
    reward = evaluate_model(model)  # 请根据你的实际情况实现这个评估函数
    return reward  # 返回模型在验证集上的表现，Optuna 会最大化该值


# ===== 设置 Optuna 的超参数优化 =====
if __name__ == "__main__":
    # ===== 创建时间戳 =====
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"td3_baseline_{timestamp}"
    save_dir = os.path.join("runs", run_name)
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 创建 Optuna 研究对象
    study = optuna.create_study(direction='maximize')  # 最大化奖励
    study.optimize(objective, n_trials=50)  # 进行50次试验

    print(f"Best hyperparameters: {study.best_params}")

    # 最终保存最佳模型
    best_params = study.best_params
    best_model = TD3(
        "MlpPolicy",
        DummyVecEnv([make_env]),
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        gamma=best_params['gamma'],
        tau=best_params['tau'],
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tensorboard")
    )
    
    best_model.learn(total_timesteps=200000)

    # 最终保存训练好的模型
    final_model_path = os.path.join(save_dir, "td3_halfcheetah_baseline")
    best_model.save(final_model_path)

    print(f"Best model saved to: {final_model_path}")
    print("Training complete.")
