# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from numpyro.infer.util import log_likelihood

from .move import BasicMoveKernel
from .util import DataModel


class StaticSMCSampler:
    def __init__(
        self,
        model: DataModel,
        initial_samples: dict,
        move: BasicMoveKernel = None,
        n_move: int = 1,
        ess_threshold: float = 0.5,
        parallel: bool = False,
    ):
        """
        A sequential Monte Carlo sampler for static models (i.e., models with parameters which do not change over time).

        :param model: NumPyro model
        :param initial_samples: initial particles
        :param move: particles move kernel to rejuvenate particles distributions once ESS goes below the threshold
        :param n_move: number of moving steps to perform once triggered
        :param ess_threshold: threshold on minimum effective sample size (ESS) to maintain
        :param parallel: passed on to ~numpyro.infer.util.log_likelihood to allow for parallel computations
        """
        self._model = model
        self._particles = initial_samples
        self.ess_threshold = ess_threshold
        if move is None:
            move = BasicMoveKernel()
        self.move = move
        self.n_move = n_move
        self.parallel = parallel
        self._log_weights = None
        self._clear_weights()

    @property
    def num_particles(self) -> int:
        """
        Number of SMC particles
        """
        return len(self._particles[next(iter(self._particles.keys()))])

    @property
    def log_weights(self) -> jnp.ndarray:
        """
        Array with log-probability weights of each SMC particle
        """
        return self._log_weights

    def get_samples(self) -> dict:
        """
        Get current dictionary of SMC samples
        """
        return self._particles

    def effective_sample_size(self):
        """Computes the effective sample size.

        :return: a scalar representing the effective sample size
        :raises AssertionError: case there is a NaN value among the normalised weights the method computes internally.
        """
        p_weights = jnp.exp(
            self._log_weights - logsumexp(self._log_weights)
        )  # compute normalised weights
        assert not jnp.isnan(p_weights).any()
        return 1.0 / (p_weights**2).sum()

    def compute_weights(self, *args, **kwargs) -> jnp.ndarray:
        """
        Computes log-likelihood of each SMC particle according to new data.

        :param args: model arguments
        :param kwargs: model keyword arguments
        """
        log_like = log_likelihood(
            self._model, self._particles, *args, parallel=self.parallel, **kwargs
        )
        log_w = jnp.stack(
            [
                log_like[k].sum(axis=tuple(jnp.arange(1, log_like[k].ndim)))
                for k in log_like.keys()
            ]
        ).sum(axis=0)
        assert not jnp.isnan(log_w).any()
        return log_w

    def _clear_weights(self):
        self._log_weights = jnp.zeros(self.num_particles)

    def resample(self, rng_key):
        """
        Performs multinomial resampling of SMC particles according to their weights.
        """
        sample_idx = jax.random.categorical(
            rng_key, logits=self._log_weights, shape=(self._log_weights.shape[0],)
        )
        self._particles = {k: v[sample_idx] for k, v in self._particles.items()}
        self._clear_weights()

    def step(self, rng_key: jax.random.PRNGKey, *args, **kwargs):
        """Performs the SMC update based on the given new observations.

        :param rng_key: random number generator key
        :param args: model arguments
        :param kwargs: model keyword arguments
        """
        rng_key_w, rng_key_s, rng_key_m = jax.random.split(rng_key, 3)
        self._log_weights += self.compute_weights(*args, **kwargs)
        self._model.update_data(**kwargs)
        ess = self.effective_sample_size()
        if ess < self.ess_threshold * self.num_particles:
            self.resample(rng_key_s)
            rng_keys = jax.random.split(rng_key_m, self.n_move)
            for i in range(self.n_move):
                self._particles = self.move(rng_keys[i], self._model, self._particles)
