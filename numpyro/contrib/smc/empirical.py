import jax
import jax.numpy as jnp

from numpyro.distributions import Distribution, Categorical, constraints


class Empirical(Distribution):
    """
    A port from Pyro's empirical distribution implementation to NumPyro.
    """
    arg_constraints = {}
    support = constraints.real
    has_enumerate_support = True

    def __init__(self, samples: jnp.ndarray, log_weights: jnp.ndarray, validate_args=None):
        self._samples = samples
        self._log_weights = log_weights
        sample_shape, weight_shape = samples.shape, log_weights.shape
        if len(weight_shape) > 1:
            raise ValueError("Batched parameters are not currently supported")
        if (
                weight_shape > sample_shape
                or weight_shape != sample_shape[: len(weight_shape)]
        ):
            raise ValueError(
                "The shape of ``log_weights`` ({}) must match "
                "the leftmost shape of ``samples`` ({})".format(
                    weight_shape, sample_shape
                    )
                )
        self._aggregation_dim = log_weights.ndim - 1
        event_shape = sample_shape[len(weight_shape):]
        self._categorical = Categorical(logits=self._log_weights)
        super().__init__(
            batch_shape=weight_shape[:-1],
            event_shape=event_shape,
            validate_args=validate_args,
            )

    @property
    def sample_size(self):
        """
        Number of samples that constitute the empirical distribution.

        :return int: number of samples collected.
        """
        return self._log_weights.size

    def sample(self, rng_key: jax.random.PRNGKey, sample_shape=tuple()):
        sample_idx = self._categorical.sample(rng_key, sample_shape)  # sample_shape x batch_shape
        # reorder samples to bring aggregation_dim to the front:
        # batch_shape x num_samples x event_shape -> num_samples x batch_shape x event_shape
        samples = (
            self._samples[None, ...].swapaxes(0, self._aggregation_dim + 1).squeeze(self._aggregation_dim + 1)
        )
        # make sample_idx.shape compatible with samples.shape: sample_shape_numel x batch_shape x event_shape
        sample_idx = sample_idx.reshape(
            (-1,) + self.batch_shape + (1,) * len(self.event_shape)
            )
        return samples.take(sample_idx, axis=0).reshape(sample_shape + samples.shape[1:])

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Returns the log of the probability mass function evaluated at ``value``.

        :param jnp.ndarray value: scalar or tensor value to be scored.
        """
        sample_shape = value.shape[:-len(self.event_shape)]
        if self.batch_shape:
            value = jnp.expand_dims(value, self._aggregation_dim)
        selection_mask = (
                self._samples == jnp.expand_dims(value, -1 - len(self.event_shape))
        ).all(axis=-1 - jnp.arange(len(self.event_shape), dtype=int))
        indices = jnp.where(selection_mask)[-1]

        return jnp.log((self._categorical.probs[indices]).reshape(sample_shape))

    def _weighted_mean(self, value, keepdim=False):
        weights = self._log_weights.reshape(
            self._log_weights.shape
            + tuple([1] * (value.ndim - self._log_weights.ndim))
            )
        dim = self._aggregation_dim
        max_weight = weights.max(axis=dim, keepdims=True)[0]
        relative_probs = jnp.exp(weights - max_weight)
        return (value * relative_probs).sum(
            axis=dim, keepdims=keepdim
            ) / relative_probs.sum(axis=dim, keepdims=keepdim)

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def mean(self):
        if self._samples.dtype in (jnp.int32, jnp.int64):
            raise ValueError(
                "Mean for discrete empirical distribution undefined. "
                + "Consider converting samples to ``jax.numpy.float32`` "
                + "or ``jax.numpy.float64``."
                )
        return self._weighted_mean(self._samples)

    @property
    def variance(self):
        if self._samples.dtype in (jnp.int32, jnp.int64):
            raise ValueError(
                "Variance for discrete empirical distribution undefined. "
                + "Consider converting samples to ``torch.float32`` "
                + "or ``torch.float64``. If these are samples from a "
                + "`Categorical` distribution, consider converting to a "
                + "`OneHotCategorical` distribution."
                )
        mean = jnp.expand_dims(self.mean, self._aggregation_dim)
        deviation_squared = (self._samples - mean) ** 2
        return self._weighted_mean(deviation_squared)

    @property
    def log_weights(self):
        return self._log_weights

    def enumerate_support(self, expand=True):
        # Empirical does not support batching, so expanding is a no-op.
        return self._samples

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, q):
        raise NotImplementedError
