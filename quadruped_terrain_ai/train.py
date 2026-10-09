"""Evolution-strategies training (JAX policy, MuJoCo rollouts in a process pool).
   python train.py --iters 150 --pop 16"""
import argparse, numpy as np, multiprocessing as mp

_env = _act = None


def _init():
    global _env, _act
    import jax
    from quadruped import QuadrupedEnv
    from brain import act, flat_init
    _env = QuadrupedEnv("mixed")
    unravel = flat_init()[1]
    _act = jax.jit(lambda th, o: act(unravel(th), o))


def _rollout(args):
    import jax.numpy as jnp
    theta, seed = args
    th = jnp.asarray(theta, dtype=jnp.float32)
    obs, _ = _env.reset(seed=seed)
    total = 0.0
    while True:
        obs, r, term, trunc, _ = _env.step(np.asarray(_act(th, jnp.asarray(obs))))
        total += r
        if term or trunc:
            return total


def centered_ranks(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--pop", type=int, default=16, help="antithetic pairs per iteration")
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--out", default="policy.npy")
    a = ap.parse_args()

    from brain import flat_init
    theta = np.array(flat_init()[0], dtype=np.float64)
    rng = np.random.default_rng(0)
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for it in range(1, a.iters + 1):
            eps = rng.standard_normal((a.pop, theta.size))
            seeds = rng.integers(0, 2 ** 31, a.pop)
            jobs = [(theta + s * a.sigma * e, int(sd)) for e, sd in zip(eps, seeds) for s in (1, -1)]
            R = np.array(pool.map(_rollout, jobs)).reshape(a.pop, 2)
            w = centered_ranks(R.ravel()).reshape(a.pop, 2)
            g = ((w[:, 0] - w[:, 1])[:, None] * eps).sum(0) / (a.pop * a.sigma)
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2                  # Adam
            theta += a.lr * a.sigma * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
            theta *= 0.999
            print(f"iter {it:4d} | mean return {R.mean():8.2f} | best {R.max():8.2f}", flush=True)
            if it % 5 == 0:
                np.save(a.out, theta)
    np.save(a.out, theta)


if __name__ == "__main__":
    main()