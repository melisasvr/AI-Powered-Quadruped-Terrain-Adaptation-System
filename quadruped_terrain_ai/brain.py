"""JAX MLP policy + rollout / benchmark helpers."""
import os, numpy as np, jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from quadruped import QuadrupedEnv, OBS_DIM, ACT_DIM, GOAL_DIST
from quadruped import KINDS

SIZES = [OBS_DIM, 64, 64, ACT_DIM]


def init_params(key):
    ps = []
    for k, (i, o) in zip(jax.random.split(key, len(SIZES) - 1), zip(SIZES[:-1], SIZES[1:])):
        ps.append((jax.random.normal(k, (i, o)) * jnp.sqrt(1.0 / i), jnp.zeros(o)))
    ps[-1] = (ps[-1][0] * 0.0, ps[-1][1])      # start exactly at the baseline gait
    return ps


def act(params, obs):
    x = obs
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    return jnp.tanh(x @ params[-1][0] + params[-1][1])


def flat_init():
    return ravel_pytree(init_params(jax.random.PRNGKey(0)))      # (theta, unravel)


def make_actor(path):
    """obs -> action. Missing file => untrained open-loop trot baseline."""
    if not path or not os.path.exists(path):
        return lambda obs: np.zeros(ACT_DIM)
    theta = jnp.asarray(np.load(path), dtype=jnp.float32)
    unravel = flat_init()[1]
    f = jax.jit(lambda o: act(unravel(theta), o))
    return lambda obs: np.asarray(f(jnp.asarray(obs)))


def rollout(env, actor, seed=0, record=False):
    obs, _ = env.reset(seed=seed)
    frames, log = [], []
    while True:
        obs, r, term, trunc, info = env.step(actor(obs))
        if record:
            s = env.snapshot(); frames.append(s)
            log.append(dict(t=env.t * env.dt, roll=s["roll"], pitch=s["pitch"], power=info["power"],
                            **{f"c{i}": c for i, c in enumerate(s["contacts"])}))
        if term or trunc:
            break
    info["success"] = (not info["fell"]) and info["dist"] >= GOAL_DIST
    return info, frames, log


def benchmark(actor, terrains, episodes=5, push=True):
    rows = []
    for k in terrains:
        env = QuadrupedEnv(k, push=push)
        for ep in range(episodes):
            info, _, _ = rollout(env, actor, seed=1000 + ep)
            rows.append(dict(terrain=k, success=info["success"], dist=info["dist"], energy=info["energy"],
                             fell=info["fell"], recovered=info["recovered"] if push else np.nan))
    return rows


if __name__ == "__main__":
    import argparse, pandas as pd
    from quadruped import KINDS
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", default="policy.npy")
    ap.add_argument("--episodes", type=int, default=5); a = ap.parse_args()
    df = pd.DataFrame(benchmark(make_actor(a.policy), KINDS, a.episodes))
    print(df.groupby("terrain").mean(numeric_only=True).round(2))