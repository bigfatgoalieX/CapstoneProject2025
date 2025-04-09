import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3

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
        model.opt.gravity[2] = -9.81  # 原为 -9.8 或 -9.81
        # 增加摩擦
        model.geom_friction[:, 0] *= 1.0  # 原为 1.0
        # 增加质量
        model.body_mass[:] *= 1.0
        
        
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
    target_env = TargetDomainWrapper(target_env)
    # maybe use a wrapper here
    print("Rendering model...")
    model_path = "./runs/td3_baseline_2025-04-09_20-42-40/checkpoints/td3_halfcheetah_baseline_180000_steps.zip"
    render_model(model_path,target_env)
