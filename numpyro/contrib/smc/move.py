# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
import abc

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

from numpyro.infer.util import initialize_model, log_density
from numpyro.util import soft_vmap

from .util import DataModel
from .proposal import Proposal, BasicMixedProposal


class MoveKernel:
    @abc.abstractmethod
    def __call__(
            self, rng_key: jax.random.PRNGKey, model: DataModel, samples: dict, log_weights: jnp.ndarray
            ) -> dict:
        pass


class BasicMoveKernel:
    def __init__(self, model: DataModel, proposal: Proposal = None):
        """
        MCMC kernel to move particles while maintaining the posterior distribution as the invariant distribution.
        """
        init_params, _, self.postprocess_fn, self.model_trace = initialize_model(
            jax.random.PRNGKey(0), model, model_kwargs=model.data
            )
        _, unpack_tf = ravel_pytree(init_params[0])
        self.unpack_tf = unpack_tf
        if proposal is None:
            proposal = BasicMixedProposal(self.model_trace)
        self.proposal = proposal
        self.last_p_accepted = None
        self.p_accepted = 0
        self.n_steps = 0
        self.model_trace = None
        self.postprocess_fn = None
        self.unpack_tf = None

    def __call__(
        self, rng_key: jax.random.PRNGKey, model: DataModel, samples: dict, log_weights: jnp.ndarray
            ) -> dict:
        """
        Moves particles according to independent Gaussian proposal based on moment-matching of the samples in their
        unconstrained space.

        :param rng_key: random number generator key for internal sampling
        :param model: target model whose parameters are being inferred
        :param samples: dictionary with samples for the random variables in the model
        """
        rng_key_m, rng_key_s, rng_key_u, rng_key_p = jax.random.split(rng_key, 4)
        n_samples = len(samples[next(iter(samples.keys()))])

        sample_idx = jax.random.categorical(rng_key_s, logits=log_weights, shape=(n_samples,))
        resampled = {k: v[sample_idx] for k, v in samples.items()}

        new_samples = self.proposal(rng_key_p, samples, log_weights, model_trace=self.model_trace)

        log_prop_ratio = self.proposal.log_prob(new_samples) - self.proposal.log_prob(resampled)

        def single_log_joint(x: dict):
            return log_density(model, tuple(), model.data, x)[0]

        # Compute Metropolis-Hastings acceptance probabilities
        log_joints_new = soft_vmap(single_log_joint, new_samples)
        log_joints_old = soft_vmap(single_log_joint, resampled)
        accept_log_prob = jnp.clip(log_joints_new - log_joints_old - log_prop_ratio, a_max=0)

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
                result[k] = jnp.concatenate([new_samples[k][accepted],
                                             resampled[k][jnp.logical_not(accepted)]])
        else:
            result = resampled

        return result
