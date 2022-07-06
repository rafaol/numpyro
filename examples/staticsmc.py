import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as nd
from numpyro.infer.util import Predictive
from tqdm import trange

from numpyro.contrib import smc
from numpyro.contrib.smc.util import cov


class SimpleModel(smc.DataModel):
    def __init__(self, n_dim=3):
        super(SimpleModel, self).__init__(arg_names=['y'])
        self.n_dim = n_dim

    def __call__(self, y, n_data: int = 1):
        mean = numpyro.sample("mean", nd.MultivariateNormal(jnp.zeros(self.n_dim), jnp.eye(self.n_dim)))
        variance = numpyro.sample("variance", nd.InverseGamma(100, 1))

        if y is not None:
            n_data = y.shape[0]
        with numpyro.plate("individual", n_data):
            individual_mean = numpyro.sample("individual_mean", nd.MultivariateNormal(mean, jnp.eye(self.n_dim)))

        numpyro.sample("y", nd.Normal(individual_mean, scale=variance**0.5), obs=y)


if __name__ == "__main__":
    n_particles = 200
    model = SimpleModel()

    prior = Predictive(model, num_samples=n_particles)
    rng_key = jax.random.PRNGKey(0)
    prior_samples = prior(rng_key, y=jnp.zeros((1, model.n_dim)))
    param_samples = {k: v for k, v in prior_samples.items() if k != 'y'}

    smc_sampler = smc.StaticSMCSampler(model, param_samples, n_move=1)

    rng_key, rng_key_gen = jax.random.split(rng_key)
    sampled_idx = jax.random.randint(rng_key_gen, minval=jnp.zeros(()), maxval=n_particles, shape=())
    true_params = {k: v[sampled_idx] for k, v in param_samples.items()}
    obs_gen = Predictive(model, posterior_samples=true_params, batch_ndims=0, return_sites=['y'])

    n_obs = 100

    t_iter = trange(n_obs)
    for t in t_iter:
        rng_key, rng_key_gen = jax.random.split(rng_key)
        datum = obs_gen(rng_key_gen, None)
        smc_sampler.step(rng_key, **datum)
        if smc_sampler.move.n_steps > 0:
            avg_p_acc = smc_sampler.move.p_accepted / smc_sampler.move.n_steps
        else:
            avg_p_acc = 0
        t_iter.set_postfix(ess=smc_sampler.effective_sample_size(), p_acc=avg_p_acc)

    p_weights = jnp.exp(smc_sampler.log_weights - logsumexp(smc_sampler.log_weights))
    print("Final approximation:")

    def print_cov(x):
        cov_mat = cov(x, p_weights)
        print(f"Cov:\n{cov_mat}")

    final_samples = smc_sampler.get_samples()
    for name, true_value in true_params.items():
        print(f"Parameter: {name}")
        print(f"True value: {true_value}")
        samples = final_samples[name]
        print(f"Mean: {jnp.tensordot(p_weights, samples, axes=1)}")
        if samples.ndim <= 2:
            print_cov(samples)

    print("Done")
