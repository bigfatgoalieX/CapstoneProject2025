import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3

def render_model(model_path,env):
    model = TD3.load(model_path)
    obs, _ = env.reset()
    terminated, truncated = False, False
    step = 0
    
    while not (terminated or truncated):
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        
if __name__ == "__main__":
    target_env = gym.make("HalfCheetah-v5", render_mode="human")
    # maybe use a wrapper here
    print("Rendering model...")
    model_path = "./models_checkpoints/td3_halfcheetah_baseline_20000_steps.zip"
    render_model(model_path,target_env)