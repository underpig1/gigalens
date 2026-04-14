import jax
import jax.numpy as jnp
from jax import pmap, vmap
import blackjax

# ------------------------
# CONFIG
# ------------------------
n_chains = 16
num_warmup_steps = 50
num_results = 20
dim = 2
init_step_size = 0.1

# Devices
n_devices = jax.device_count()
n_chains = (n_chains // n_devices) * n_devices
chains_per_device = n_chains // n_devices

key = jax.random.PRNGKey(0)
device_keys = jax.random.split(key, n_devices)

# ------------------------
# Fake “posterior” sampler
# ------------------------
class FakeQZ:
    def sample(self, n, seed):
        k = jax.random.split(seed, 1)[0]
        return jax.random.normal(k, (n, dim))

qz = FakeQZ()

# ------------------------
# Initialize chains
# ------------------------
init_z = jnp.stack([qz.sample(chains_per_device, seed=k) for k in device_keys])
print("init_z.shape:", init_z.shape)       # (n_devices, chains_per_device, dim)
print("device_keys.shape:", device_keys.shape)  # (n_devices, 2)

# ------------------------
# Log-prob function
# ------------------------
def log_prob_fn_single(z):
    return -0.5 * jnp.sum(z**2)

# ------------------------
# Warmup / adaptation
# ------------------------
def warmup_chain(z0, key):
    keys = jax.random.split(key, z0.shape[0])

    def run_single(z, k):
        adapt = blackjax.window_adaptation(
            algorithm=blackjax.nuts,
            logdensity_fn=log_prob_fn_single,
            is_mass_matrix_diagonal=True,
            initial_step_size=init_step_size,
            target_acceptance_rate=0.8,
        )
        (last_state, params), _ = adapt.run(k, z, num_warmup_steps)
        return last_state, params

    states, params = vmap(run_single)(z0, keys)
    return states, params

warmup_chain_pmap = pmap(warmup_chain)
states, params = warmup_chain_pmap(init_z, device_keys)

# ------------------------
# Sampling
# ------------------------
def run_chain(states, params, key):
    keys = jax.random.split(key, states.position.shape[0])

    def run_single(state, param, k):
        kernel = blackjax.nuts(
            log_prob_fn_single,
            param["step_size"],
            param["inverse_mass_matrix"],
        ).step

        def one_step(carry, key_inner):
            state = carry
            state, _ = kernel(key_inner, state)
            return state, state.position

        inner_keys = jax.random.split(k, num_results)
        _, samples = jax.lax.scan(one_step, state, inner_keys)
        return samples

    samples = vmap(run_single)(states, params, keys)
    return samples

run_chain_pmap = pmap(run_chain)
samples = run_chain_pmap(states, params, device_keys)
print("samples.shape:", samples.shape)

# ------------------------
# Flatten to (n_chains, num_results, dim)
# ------------------------
samples = jnp.reshape(samples, (n_chains, num_results, dim))
print("final samples shape:", samples.shape)