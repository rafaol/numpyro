import abc
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from numpyro.distributions import Distribution, LowRankMultivariateNormal, Categorical

from .util import (
    compute_moments,
    unconstrain_samples,
    get_continuous_samples,
    get_discrete_samples,
    unpack_samples,
    constrain_samples,
    compute_jacobians
    )


class Proposal:
    @abc.abstractmethod
    def __call__(self, rng_key: jax.random.PRNGKey, samples: dict, log_weights: jnp.ndarray,
                 *args, **kwargs) -> dict:
        pass

    @abc.abstractmethod
    def log_prob(self, value: dict) -> jnp.ndarray:
        pass


class BasicContinuousProposal(Proposal):
    def __init__(self, model_trace: dict, cov_nugget=1e-6):
        super().__init__()
        self.cov_nugget = cov_nugget
        self.model_trace = model_trace
        self.proposal_dist = None

    def make_continuous_proposal(self, unconstrained_samples: jnp.ndarray, log_weights: jnp.ndarray) -> Distribution:
        """
        Make a moment-matched Gaussian proposal distribution out of samples.
        """
        mean, covariance = compute_moments(unconstrained_samples, log_weights)
        eigvals, eigvecs = jnp.linalg.eigh(covariance)
        positive_mask = (eigvals > 0)
        if not positive_mask.any():
            raise AssertionError("No positive eigenvalues found for proposal covariance matrix")
        cov_factor = eigvecs[..., positive_mask]
        cov_diag = jnp.ones_like(eigvals) * self.cov_nugget + jnp.clip(eigvals, a_min=0)

        return LowRankMultivariateNormal(mean, cov_factor=cov_factor, cov_diag=cov_diag)

    def __call__(self, rng_key: jax.random.PRNGKey, samples: dict, log_weights: jnp.ndarray,
                 *args, **kwargs) -> dict:
        n_samples = len(samples[next(iter(samples.keys()))])
        old_unconstrained, unpack_continuous = unconstrain_samples(samples, self.model_trace,
                                                                   return_unpack=True)
        proposal_dist = self.make_continuous_proposal(old_unconstrained, log_weights)
        new_unconstrained_continuous = proposal_dist.sample(rng_key, (n_samples,))
        new_unconstrained_unpacked = unpack_samples(new_unconstrained_continuous, unpack_continuous)
        new_continuous_samples = constrain_samples(new_unconstrained_unpacked, self.model_trace)
        self.proposal_dist = proposal_dist

        return new_continuous_samples

    def log_prob(self, values: dict) -> jnp.ndarray:
        unconstrained = unconstrain_samples(values, self.model_trace)
        log_probs = self.proposal_dist.log_prob(unconstrained) - compute_jacobians(values, self.model_trace)
        return log_probs


class BasicMixedProposal(BasicContinuousProposal):
    def __init__(self, model_trace: dict, cov_nugget=1e-6):
        super().__init__(model_trace, cov_nugget=cov_nugget)
        self.discrete_proposals = dict()

    def __call__(self, rng_key: jax.random.PRNGKey, samples: dict, log_weights: jnp.ndarray,
                 *args, **kwargs) -> dict:
        rng_cont, rng_disc = jax.random.split(rng_key)
        continuous_samples = get_continuous_samples(samples, self.model_trace)
        new_continuous_samples = super().__call__(rng_cont, continuous_samples, log_weights, self.model_trace)

        n_samples = len(samples[next(iter(samples.keys()))])
        discrete_samples = get_discrete_samples(samples, self.model_trace)

        if len(discrete_samples.keys()) > 0:
            new_discrete_samples = {}
            relative_probs = jnp.exp(log_weights - logsumexp(log_weights))

            for name, values in discrete_samples.items():
                # _, counts = jnp.unique(values, return_counts=True)
                # p = (counts / n_samples) * relative_probs
                reshaped_values = values.reshape((n_samples, -1)).T
                p = jnp.stack([jnp.bincount(v, relative_probs) for v in reshaped_values])
                cat = Categorical(probs=p)
                rng_disc, _ = jax.random.split(rng_disc)

                sampled = cat.sample(rng_disc, (n_samples,)).reshape(values.shape)
                # log_prob_new = cat.log_prob(sampled)
                # log_prob_old = cat.log_prob(values)
                # log_prop_ratio += [log_prob_new - log_prob_old]
                new_discrete_samples[name] = sampled
                self.discrete_proposals[name] = cat

            # log_prop_ratio = jnp.stack(log_prop_ratio).sum(axis=0) + log_prop_ratio_cont

            merged_samples = {**new_continuous_samples, **new_discrete_samples}
        else:
            merged_samples = new_continuous_samples
            # log_prop_ratio = log_prop_ratio_cont

        return merged_samples

    def log_prob(self, values: dict) -> jnp.ndarray:
        continuous_samples = get_continuous_samples(values, self.model_trace)
        log_prob_cont = super().log_prob(continuous_samples)
        discrete_samples = get_discrete_samples(values, self.model_trace)
        if len(discrete_samples.keys()) > 0:
            n_samples = len(discrete_samples[next(iter(discrete_samples.keys()))])
            log_prob_disc = [self.discrete_proposals[k].log_prob(v.reshape((n_samples, -1))).sum(axis=-1)
                             for k, v in discrete_samples.items()]
            log_prob_disc = jnp.stack(log_prob_disc).sum(axis=0)
            log_prob = log_prob_cont + log_prob_disc
        else:
            log_prob = log_prob_cont
        return log_prob
