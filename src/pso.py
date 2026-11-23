"""Particle Swarm Optimization (PSO) recommender for latent factor matrix factorization."""

from dataclasses import dataclass, asdict
import json
import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from src.base import BaseRecommender
from src.fitness import MFProblem
from src.metrics import rmse
from src.protocol import RATING_MIN, RATING_MAX


@dataclass
class PSOConfig:
    d: int = 10
    n_particles: int = 20
    max_evals: int = 2000
    lam: float = 0.01
    init_std: float = 0.1
    bound: float = 1.0
    w_start: float = 0.9
    w_end: float = 0.4
    c1: float = 1.5
    c2: float = 1.5
    vmax_frac: float = 0.05
    batch_frac: float | None = None
    resample_every: int = 10
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


class PSORecommender(BaseRecommender):
    """PSO-based latent factor recommender."""

    def __init__(self, config: PSOConfig | None = None):
        super().__init__()
        self.config = config if config is not None else PSOConfig()

    def fit(self, train, val=None, init=None, **kwargs) -> "PSORecommender":
        if init is not None:
            raise ValueError("PSORecommender does not support explicit 'init' initialization.")

        cfg = self.config
        N = cfg.n_particles
        rng = np.random.default_rng(cfg.seed)

        problem = MFProblem(train, d=cfg.d, lam=cfg.lam, bound=cfg.bound)
        self.mu = problem.mu
        self._train_csr = train.to_csr()
        self.history = []

        # Mini-batch setup
        batch = None
        if cfg.batch_frac is not None and cfg.batch_frac < 1.0:
            batch = problem.make_batch(cfg.batch_frac, rng)

        # Initial swarm
        X = problem.init_population(N, cfg.init_std, rng)
        Vel = np.zeros_like(X)
        f = problem.fitness_swarm(X, batch)
        n_evals = N

        P = X.copy()
        pf = f.copy()
        g = int(pf.argmin())
        G = P[g].copy()
        gf = float(pf[g])

        vmax = np.float32(cfg.vmax_frac * 2.0 * cfg.bound)
        bound = np.float32(cfg.bound)

        t0 = time.perf_counter()
        iteration = 0

        # Initial history record (iteration 0)
        G_U, G_V = problem.unpack(G)
        self.U = G_U.copy()
        self.V = G_V.copy()
        r_train = problem.rmse(G, batch=None, backend="numba")
        record = {
            "iteration": iteration,
            "n_evals": n_evals,
            "time_s": float(time.perf_counter() - t0),
            "gbest_fitness": gf,
            "rmse_train": float(r_train),
        }
        if val is not None:
            preds_val = self.predict(val.users, val.items, clip=True)
            record["rmse_val"] = float(rmse(val.ratings, preds_val))
        self.history.append(record)

        while n_evals < cfg.max_evals:
            iteration += 1

            # Resampling for mini-batch mode
            if cfg.batch_frac is not None and iteration % cfg.resample_every == 0:
                batch = problem.make_batch(cfg.batch_frac, rng)
                # Re-evaluate all personal bests on the new batch
                pf = problem.fitness_swarm(P, batch)
                n_evals += N
                g = int(pf.argmin())
                G = P[g].copy()
                gf = float(pf[g])

            progress = min(1.0, n_evals / cfg.max_evals)
            w = np.float32(cfg.w_start + (cfg.w_end - cfg.w_start) * progress)

            r1 = rng.random(X.shape, dtype=np.float32)
            r2 = rng.random(X.shape, dtype=np.float32)

            # PSO velocity update
            Vel = w * Vel + cfg.c1 * r1 * (P - X) + cfg.c2 * r2 * (G - X)
            np.clip(Vel, -vmax, vmax, out=Vel)

            # Position update and boundary enforcement
            X += Vel
            out_bounds = (X < -bound) | (X > bound)
            np.clip(X, -bound, bound, out=X)
            Vel[out_bounds] = 0.0

            # Evaluate swarm
            f = problem.fitness_swarm(X, batch)
            n_evals += N

            # Update personal bests
            better = f < pf
            P[better] = X[better]
            pf[better] = f[better]

            # Update global best
            curr_best = int(pf.argmin())
            if pf[curr_best] < gf:
                G = P[curr_best].copy()
                gf = float(pf[curr_best])

            # Unpack G to temporary model factors for logging
            G_U, G_V = problem.unpack(G)
            self.U = G_U.copy()
            self.V = G_V.copy()
            r_train = problem.rmse(G, batch=None, backend="numba")

            record = {
                "iteration": iteration,
                "n_evals": n_evals,
                "time_s": float(time.perf_counter() - t0),
                "gbest_fitness": gf,
                "rmse_train": float(r_train),
            }
            if val is not None:
                preds_val = self.predict(val.users, val.items, clip=True)
                record["rmse_val"] = float(rmse(val.ratings, preds_val))
            self.history.append(record)

        # Final assignments
        final_U, final_V = problem.unpack(G)
        self.U = final_U.copy()
        self.V = final_V.copy()
        self.n_evals = n_evals

        return self
