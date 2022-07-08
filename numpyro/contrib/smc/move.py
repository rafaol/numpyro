# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

from numpyro.distributions import Distribution, MultivariateNormal
from numpyro.infer.util import initialize_model

from .util import (
    DataModel,
    compute_acceptance_probs,
    compute_moments,
    unconstrain_samples,
)


class BasicMoveKernel:
    def __init__(self, cov_nugget: float = 1e-6):
        """
        MCMC kernel to move particles while maintaining the posterior distribution as the invariant distribution.

        :param cov_nugget: small floating point number to add to the diagonal of the sample covariance matrix to avoid
        numerical issues with positive-definiteness assumptions.
        """
        self.last_p_accepted = None
        self.p_accepted = 0
        self.n_steps = 0
        self.cov_nugget = cov_nugget
        self.model_trace = None
        self.postprocess_fn = None
        self.unpack_tf = None

    def make_proposal(self, samples: dict) -> Distribution:
        """
        Make a moment-matched Gaussian proposal distribution out of samples.

        :param samples: dictionary with samples for the random variables in the model
        """
        latent_mean, latent_covariance = compute_moments(
            unconstrain_samples(samples, self.model_trace)
        )
        return MultivariateNormal(
            latent_mean,
            covariance_matrix=latent_covariance
            + self.cov_nugget * jnp.eye(latent_mean.shape[0]),
        )

    def __call__(
        self, rng_key: jax.random.PRNGKey, model: DataModel, samples: dict
    ) -> dict:
        """
        Moves particles according to independent Gaussian proposal based on moment-matching of the samples in their
        unconstrained space.

        :param rng_key: random number generator key for internal sampling
        :param model: target model whose parameters are being inferred
        :param samples: dictionary with samples for the random variables in the model
        """
        if self.model_trace is None:
            init_params, _, self.postprocess_fn, self.model_trace = initialize_model(
                rng_key, model, model_kwargs=model.data
            )
            _, unpack_tf = ravel_pytree(init_params[0])
            self.unpack_tf = unpack_tf

        rng_key_m, rng_key_s, rng_key_u, rng_key_p = jax.random.split(rng_key, 4)
        n_samples = len(samples[next(iter(samples.keys()))])

        proposal_dist = self.make_proposal(samples)
        new_unconstrained_samples = jax.vmap(self.unpack_tf)(
            proposal_dist.sample(rng_key_s, (n_samples,))
        )
        new_samples = self.postprocess_fn(new_unconstrained_samples)

        # Compute Metropolis-Hastings acceptance probabilities
        accept_log_prob = compute_acceptance_probs(
            samples, new_samples, self.model_trace, model, proposal_dist
        )

        # Sample accepted particles
        log_u = jnp.log(jax.random.uniform(rng_key_u, shape=(n_samples,)))
        accepted = (log_u < accept_log_prob).squeeze()
        n_accepted = accepted.sum()

        # Update performance stats
        self.last_p_accepted = n_accepted / n_samples
        self.p_accepted += self.last_p_accepted
        self.n_steps += 1

        # Assemble accepted particles
        if n_accepted > 0:
            result = {}
            for k in samples.keys():
                result[k] = jnp.concatenate(
                    [new_samples[k][accepted], samples[k][jnp.logical_not(accepted)]]
                )
        else:
            result = samples

        return result
