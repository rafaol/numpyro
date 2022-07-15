import abc
import jax
import jax.numpy as jnp

from numpyro.distributions import Distribution, MultivariateNormal, Categorical

from .util import (
    compute_moments,
    unconstrain_samples,
    get_continuous_samples,
    get_discrete_samples,
    pack_samples,
    unpack_samples,
    constrain_samples,
    compute_jacobians
    )


class Proposal:
    @abc.abstractmethod
    def __call__(self, rng_key: jax.random.PRNGKey, samples: dict, *args, **kwargs) -> (dict, jnp.ndarray):
        pass


class BasicContinuousProposal(Proposal):
    def __init__(self, cov_nugget=1e-6):
        super().__init__()
        self.cov_nugget = cov_nugget

    def make_continuous_proposal(self, samples: dict, model_trace: dict) -> Distribution:
        """
        Make a moment-matched Gaussian proposal distribution out of samples.

        :param samples: dictionary with samples for the random variables in the model
        :param model_trace: model trace dictionary
        """
        latent_mean, latent_covariance = compute_moments(
            unconstrain_samples(samples, model_trace)
            )
        return MultivariateNormal(
            latent_mean, covariance_matrix=latent_covariance + self.cov_nugget * jnp.eye(latent_mean.shape[0]),
            )

    def __call__(self, rng_key: jax.random.PRNGKey, samples: dict, model_trace: dict = None,
                 *args, **kwargs):
        n_samples = len(samples[next(iter(samples.keys()))])
        continuous_samples = get_continuous_samples(samples, model_trace)
        old_unconstrained, unpack_continuous = unconstrain_samples(continuous_samples, model_trace, return_unpack=True)
        proposal_dist = self.make_continuous_proposal(continuous_samples, model_trace)
        new_unconstrained_continuous = proposal_dist.sample(rng_key, (n_samples,))
        new_unconstrained_unpacked = unpack_samples(new_unconstrained_continuous, unpack_continuous)
        new_continuous_samples = constrain_samples(new_unconstrained_unpacked, model_trace)

        log_probs_new = proposal_dist.log_prob(new_unconstrained_continuous) \
                        - compute_jacobians(new_continuous_samples, model_trace)
        log_probs_old = proposal_dist.log_prob(unconstrain_samples(samples, model_trace)) \
                        - compute_jacobians(samples, model_trace)
        log_prop_ratio = log_probs_new - log_probs_old

        return new_continuous_samples, log_prop_ratio


class BasicMixedProposal(BasicContinuousProposal):
    def __init__(self, cov_nugget=1e-6):
        super().__init__(cov_nugget=cov_nugget)

    def __call__(self, rng_key: jax.random.PRNGKey, samples: dict, model_trace: dict = None,
                 *args, **kwargs):
        rng_cont, rng_disc = jax.random.split(rng_key)
        new_continuous_samples, log_prop_ratio_cont = super().__call__(rng_key, samples, model_trace)

        n_samples = len(samples[next(iter(samples.keys()))])
        discrete_samples = get_discrete_samples(samples, model_trace)

        if len(discrete_samples.keys()) > 0:
            log_prop_ratio = []
            new_discrete_samples = {}

            for name, values in discrete_samples.items():
                _, counts = jnp.unique(values, return_counts=True)

                cat = Categorical(probs=counts / n_samples)
                rng_disc, _ = jax.random.split(rng_disc)

                sampled = cat.sample(rng_disc, (n_samples,))
                log_prob_new = cat.log_prob(sampled)
                log_prob_old = cat.log_prob(values)
                log_prop_ratio += [log_prob_new - log_prob_old]
                new_discrete_samples[name] = sampled

            log_prop_ratio = jnp.stack(log_prop_ratio).sum(axis=0) + log_prop_ratio_cont

            merged_samples = {**new_continuous_samples, **new_discrete_samples}
        else:
            merged_samples = new_continuous_samples
            log_prop_ratio = log_prop_ratio_cont

        return merged_samples, log_prop_ratio
