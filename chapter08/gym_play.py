import numpy as np
import gym

from typing import TypeAlias, Literal
Env: TypeAlias = gym.wrappers.time_limit.TimeLimit
State: TypeAlias = tuple[float, float, float, float]
Action: TypeAlias = Literal[0, 1]
Reward: TypeAlias = float
Info: TypeAlias = dict[str, bool]


if __name__ == "__main__":
    rng: np.random.Generator = np.random.default_rng()
    env: Env = gym.make("CartPole-v0", render_mode="human")
    state: State = env.reset()
    done: bool = False

    while not done:
        env.render()
        action: Action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
    env.close()