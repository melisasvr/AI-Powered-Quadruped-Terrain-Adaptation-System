"""Tiny JAX MLP policy (no gradients needed: trained with evolution strategies)."""
import jax, jax.numpy as jnp
from env import OBS_DIM, ACT_DIM

SIZES = [OBS_DIM, 64, 64, ACT_DIM]


def init_params(key):
    params = []
    for k, (i, o) in zip(jax.random.split(key, len(SIZES) - 1), zip(SIZES[:-1], SIZES[1:])):
        params.append((jax.random.normal(k, (i, o)) * jnp.sqrt(1.0 / i), jnp.zeros(o)))
    W, b = params[-1]
    params[-1] = (W * 0.1, b)
    return params


def act(params, obs):
    x = obs
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return jnp.tanh(x @ W + b)
