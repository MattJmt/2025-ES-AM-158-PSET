#!/usr/bin/env python3

import upkie.envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import logging
logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)

import upkie.logging
upkie.logging.logger.setLevel("ERROR")


upkie.envs.register()

ENV_ID = "Upkie-Spine-Pendulum"
ENV_KWARGS = dict(frequency=50.0)
SEED = 0

def main():

    # Vectorized environment (1 process is fine for this lightweight task)
    env = make_vec_env(ENV_ID, n_envs=16, env_kwargs=ENV_KWARGS, seed=SEED)

    # Save the best model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./models/pendulum_best/",
        name_prefix="best_model",
    )

    # PPO hyperparameters tuned for this problem
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        n_epochs=10,
        device="auto",
        verbose=1,
    )

    # Train ~1–5 million steps (you can start with 500k)
    model.learn(
        total_timesteps=500_000,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save("./models/ppo_upkie_final")

if __name__ == "__main__":
    main()
