# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
import abc
from typing import Callable, List

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

from numpyro.infer.util import initialize_model, log_density, log_likelihood
from numpyro.util import soft_vmap
from numpyro.handlers import block

from .util import filter_samples
from .proposal import Proposal, BasicMixedProposal


class MoveKernel:
    @abc.abstractmethod
    def __call__(
            self, rng_key: jax.random.PRNGKey, model: Callable, samples: dict, log_weights: jnp.ndarray
            ) -> dict:
        pass


class BasicMoveKernel:
    def __init__(self, model: Callable, *model_args, proposal: Proposal = None, **model_kwargs):
        """
        MCMC kernel to move particles while maintaining the posterior distribution as the invariant distribution.
        """
        init_params, _, self.postprocess_fn, self.model_trace = initialize_model(
            jax.random.PRNGKey(0), model, model_args=model_args, model_kwargs=model_kwargs
            )
        _, unpack_tf = ravel_pytree(init_params[0])
        self.unpack_tf = unpack_tf
        if proposal is None:
            proposal = BasicMixedProposal(self.model_trace)
        self.proposal = proposal
        self.last_p_accepted = None
        self.p_accepted = 0
        self.n_steps = 0

    def __call__(
        self, rng_key: jax.random.PRNGKey, model: Callable, model_args: List[tuple], model_kwargs: List[dict],
            samples: dict, log_weights: jnp.ndarray) -> dict:
        """
        Moves particles according to independent Gaussian proposal based on moment-matching of the samples in their
        unconstrained space.

        :param rng_key: random number generator key for internal sampling
        :param model: target model whose parameters are being inferred
        :param samples: dictionary with samples for the random variables in the model
        """
        samples = filter_samples(samples, self.model_trace)
        rng_key_m, rng_key_s, rng_key_u, rng_key_p = jax.random.split(rng_key, 4)
        n_samples = len(samples[next(iter(samples.keys()))])

        sample_idx = jax.random.categorical(rng_key_s, logits=log_weights, shape=(n_samples,))
        resampled = {k: v[sample_idx] for k, v in samples.items()}

        new_samples = self.proposal(rng_key_p, samples, log_weights, model_trace=self.model_trace)

        log_prop_ratio = self.proposal.log_prob(new_samples) - self.proposal.log_prob(resampled)

        # Evaluate log-likelihood of samples
        def single_log_like(params, *args, **kwargs):
            log_like = log_likelihood(model, params, *args, **kwargs)
            return jnp.stack([log_like[k].sum(axis=tuple(jnp.arange(1, log_like[k].ndim))) for k in log_like.keys()]
                             ).sum(axis=0)

        def compute_log_like(params):
            log_likes = jax.tree_map(lambda args, kwargs: single_log_like(params, *args, **kwargs),
                                     model_args, model_kwargs,
                                     is_leaf=lambda node: (type(node) == dict or type(node) == tuple))
            return jnp.sum(jnp.stack(log_likes), axis=0)

        log_like_new = compute_log_like(new_samples)
        log_like_old = compute_log_like(resampled)

        # Evaluate log-prior probability
        def hide_observed(site):
            if site['type'] == 'sample':
                return site['is_observed']
            else:
                return False
        masked_model = block(model, hide_fn=hide_observed)

        def single_log_prior(params):   # TODO: Consider all arguments for models whose arguments influence priors
            return log_density(masked_model, model_args[0], model_kwargs[0], params)[0]

        log_prior_new = soft_vmap(single_log_prior, new_samples)
        log_prior_old = soft_vmap(single_log_prior, resampled)

        # log_joints_new = soft_vmap(single_log_joint, new_samples)
        # log_joints_old = soft_vmap(single_log_joint, resampled)
        log_joints_new = log_like_new + log_prior_new
        log_joints_old = log_like_old + log_prior_old

        # Compute Metropolis-Hastings acceptance probabilities
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
