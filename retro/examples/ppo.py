"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

import argparse
from copy import deepcopy
import os
import gymnasium as gym
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, VecFrameStack, VecTransposeImage, VecVideoRecorder
)
import retro
import torch
from stable_baselines3.common.callbacks import BaseCallback

class SaveBestModel(BaseCallback):
    def __init__(self, freq:int= 10, save_path:str="."):
        super()
        self.freq = freq
        self.save_path = save_path
        self.best_reward = -float('inf')
        self.n_calls = 0

    def _on_step(self) -> bool:
        self.n_calls += 1
        return True

    def _on_rollout_end(self) -> bool:
        if self.n_calls % self.freq == 0:
            this_reward = self.locals["rewards"][-1]
            if this_reward > self.best_reward:
                print(f"New best reward {self.best_reward} => {this_reward}")
                self.best_reward = this_reward
                print(f"Saving best model to {self.save_path}")
                self.model.save(self.save_path)
                # env=self.model.get_env()
                # env.unwrapped.record_movie(f"best_{best_rew}.bk2")
                # env.reset()
                # for act in acts:
                #     env.step(act)
                # env.unwrapped.stop_record()
        return True

class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Joust-Arcade")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--resume", default=True)
    args = parser.parse_args()

    def make_env():
        env = make_retro(game=args.game, state=args.state, scenario=args.scenario, render_mode='rgb_array')
        env = WarpFrame(env)
        # env = ClipRewardEnv(env)
        return env

    PROCESS_COUNT = 64 

    # torch.set_default_dtype(torch.float32)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")# Use Metal (MPS) backend if available
    # device = "cpu"

    venv =  VecTransposeImage( 
                VecFrameStack(
                    # VecVideoRecorder(
                        SubprocVecEnv( 
                            [make_env] * PROCESS_COUNT 
                        ) 
                    #     , video_folder="videos"
                    #     , record_video_trigger=lambda x: False  # We'll trigger manually
                    #     , video_length=1000
                    # )
                    , n_stack=4
                )
            )

    model = PPO(
        policy="CnnPolicy",
        env=venv,
        learning_rate=lambda f: f * 2.5e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
        device=device,
    )


    best_model_path = "./best_model.zip"
    if args.resume and os.path.exists(best_model_path):
        model = PPO.load(best_model_path, env=venv, device=device)

    model.learn(
        total_timesteps=100_000_000,
        log_interval=1,
        callback = SaveBestModel(save_path=best_model_path)
    )


if __name__ == "__main__":
    main()
