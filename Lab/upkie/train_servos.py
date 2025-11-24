#!/usr/bin/env python3

import os
from typing import Dict, List

import gymnasium as gym
import numpy as np
import upkie.envs
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Import obs wrapper from rollout_policy_servos.py
from rollout_policy_servos import ServoObsFlattenWrapper


# ---------------------------------------------------------
# Safe helper for reading servo fields (float or [float])
# ---------------------------------------------------------
def safe_get(ob: Dict, key: str, default: float = 0.0) -> float:
    """
    Return a float from dictionary ob[key], handling either float or [float].
    """
    if key not in ob:
        return default
    value = ob[key]
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[0]) if len(value) > 0 else default
    return float(value)


# ---------------------------------------------------------
# Torque-based action wrapper (FULL CONTROL)
# ---------------------------------------------------------
class TorqueActionWrapper(gym.ActionWrapper):
    """
    Action: continuous vector in [-1, 1]^N, one per servo.
    Each component scales feedforward_torque in [-tau_max, +tau_max].
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Dict)

        self.joint_names: List[str] = list(env.action_space.spaces.keys())

        # Maximum torque for each joint
        max_torques = []
        for j in self.joint_names:
            joint_space = env.action_space[j]
            max_tau = float(joint_space["maximum_torque"].high[0])
            max_torques.append(max_tau)
        self.max_torque = np.asarray(max_torques, dtype=np.float32)

        # New action space: Box[-1, 1]^num_joints
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.joint_names),),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> Dict:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a = np.clip(a, -1.0, 1.0)

        env_action: Dict[str, Dict[str, float]] = {}

        for i, name in enumerate(self.joint_names):
            tau = float(a[i] * self.max_torque[i])
            env_action[name] = dict(
                position=np.nan,             # disable position loop
                velocity=0.0,                # no velocity targeting
                feedforward_torque=tau,      # direct torque control
                kp_scale=0.0,                # no extra feedback
                kd_scale=0.0,
                maximum_torque=self.max_torque[i] - 1e-6,
            )

        return env_action


# ---------------------------------------------------------
# Fall termination + stabilization reward wrapper
# ---------------------------------------------------------
class StabilizeRewardWrapper(gym.Wrapper):
    """
    Reward:
      + upright (pitch near zero)
      + small torques
      + longer survival
      - big penalty on fall
    """

    def __init__(self, env: gym.Env, fall_pitch: float = 0.9):
        super().__init__(env)
        self.fall_pitch = fall_pitch

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        spine = info["spine_observation"]
        servo_obs = spine["servo"]

        pitch = spine["base_orientation"]["pitch"]  # rad, 0 = upright

        # Sum leg torques for smoothness penalty
        torque_sum = 0.0
        for j in ["left_hip", "left_knee", "right_hip", "right_knee"]:
            torque_sum += abs(safe_get(servo_obs.get(j, {}), "torque", 0.0))

        # ---------------------
        #     REWARD
        # ---------------------
        reward = 0.0

        # 1) Upright (Gaussian around pitch=0)
        reward += 1.0 * np.exp(-(pitch ** 2) / 0.3)

        # 2) Small torque (smoothness)
        reward -= 0.001 * torque_sum

        # 3) Survival bonus (small positive per step)
        reward += 0.01

        # 4) Fall detection: terminate + big penalty
        if abs(pitch) > self.fall_pitch:
            terminated = True
            reward -= 10.0

        return obs, float(reward), terminated, truncated, info


# ---------------------------------------------------------
# Upkie registration
# ---------------------------------------------------------
upkie.envs.register()


# ---------------------------------------------------------
# Wrapped env factory (one instance)
# ---------------------------------------------------------
def make_training_env():
    frequency_hz = 100.0
    max_steps = 300

    env = gym.make(
        "Upkie-Spine-Servos",
        frequency=frequency_hz,
        regulate_frequency=False,   # run as fast as possible
        frequency_checks=False,     # disable late warnings
    )

    # 1) torque-based actions
    env = TorqueActionWrapper(env)

    # 2) flatten observations (position, velocity per joint)
    env = ServoObsFlattenWrapper(env)

    # 3) stabilization reward + early termination
    env = StabilizeRewardWrapper(env, fall_pitch=0.9)

    # 4) Time limit
    env = TimeLimit(env, max_episode_steps=max_steps)
    return env


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
def main():
    os.makedirs("./models/servos_best/", exist_ok=True)

    # 16 parallel envs uses your 22 cores well
    env = make_vec_env(make_training_env, n_envs=16)

    checkpoint = CheckpointCallback(
        save_freq=50_000,
        save_path="./models/servos_best/",
        name_prefix="best_model",
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        device="cpu",          # PPO MLP is usually faster on CPU in SB3
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,           # more epochs for stability
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        target_kl=0.02,
        verbose=1,
    )

    model.learn(
        total_timesteps=1_500_000,  # can go to 3e6 if time allows
        callback=checkpoint,
        progress_bar=True,
    )

    model.save("./models/ppo_upkie_servos_stabilize_final.zip")


if __name__ == "__main__":
    main()
