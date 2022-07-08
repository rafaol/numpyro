# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

from numpyro.distributions import Distribution
from numpyro.distributions.transforms import biject_to
from numpyro.infer.util import log_density
from numpyro.util import soft_vmap


class DataModel:
    def __init__(self, arg_names: list):
        self.arg_names = arg_names
        self._data = None
        self.clear_data()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, **kwargs):
        assert self.arg_names == list(kwargs.keys())
        self._data = kwargs

    def update_data(self, **kwargs):
        assert self.arg_names == list(kwargs.keys())
        if self._data is None:
            self._data = kwargs
        else:
            for k in self.arg_names:
                self._data[k] = jnp.concatenate([self._data[k], kwargs[k]], axis=0)

    def clear_data(self):
        self._data = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


def cov(m: jnp.ndarray, weights: jnp.ndarray = None, rowvar: bool = False):
    """Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Original: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable (if `rowvar` is`True`),
            and each column a single observation of all those variables.
        weights: 1-D array of normalised sample weights
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.ndim < 2:
        m = m.reshape((1, -1))
    if not rowvar and m.shape[0] != 1:
        m = m.T
    if weights is None:
        fact = 1.0 / (m.shape[1] - 1)
        m -= m.mean(axis=1, keepdims=True)
        mt = m.T
        return fact * m @ mt.squeeze()
    else:
        fact = 1.0 / (1 - (weights**2).sum())
        m -= (m * weights).mean(axis=1, keepdims=True)
        return fact * (weights * m) @ m.T


def compute_moments(unconstrained_samples):
    return unconstrained_samples.mean(axis=0), cov(unconstrained_samples)


def unconstrain_single_sample(sample: dict, model_trace: dict):
    unconstrained_sample = {}
    for k, v in sample.items():
        site = model_trace[k]
        transform = biject_to(site["fn"].support)
        unconstrained_sample[k] = transform.inv(sample[k])
    return ravel_pytree(unconstrained_sample)[0]


def unconstrain_samples(samples, model_trace: dict):
    return soft_vmap(lambda x: unconstrain_single_sample(x, model_trace), samples)


def compute_single_jacobian(sample: dict, model_trace: dict):
    jacs = []
    for k, v in sample.items():
        site = model_trace[k]
        transform = biject_to(site["fn"].support)
        unconstrained_sample = transform.inv(sample[k])
        jac = transform.log_abs_det_jacobian(unconstrained_sample, sample[k])
        jacs += [jac.sum()]
    return jnp.stack(jacs).sum(axis=0)


def compute_jacobians(samples: dict, model_trace: dict, batch_n_dims=1, chunk_size=1):
    def single_jac(x):
        return compute_single_jacobian(x, model_trace)

    return soft_vmap(single_jac, samples, batch_n_dims, chunk_size)


def compute_acceptance_probs(
    old_samples: dict,
    new_samples: dict,
    model_trace: dict,
    model: DataModel,
    proposal_dist: Distribution,
):
    n_samples = len(old_samples[next(iter(old_samples.keys()))])

    def compute_single_log_ratio(sample):
        log_prop = proposal_dist.log_prob(
            unconstrain_single_sample(sample, model_trace)
        ) - compute_single_jacobian(sample, model_trace)
        log_joint, _ = log_density(model, tuple(), model.data, sample)
        assert log_prop.shape == log_joint.shape
        return log_joint - log_prop

    log_ratios_new = soft_vmap(compute_single_log_ratio, new_samples, 1, n_samples)
    log_ratios_old = soft_vmap(compute_single_log_ratio, old_samples, 1, n_samples)

    log_accept_prob = jnp.clip(log_ratios_new - log_ratios_old, a_max=0)
    return log_accept_prob
