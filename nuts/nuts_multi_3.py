import jax
import jax.numpy as jnp
from jax import jit, pmap, vmap
import numpy as np
import blackjax
import time
import jax.experimental.multihost_utils as mhu
import gigalens.jax.simulator as sim
from window_adaptation import window_adaptation
from jax.experimental import shard_map

def NUTS(
    model_seq,
    q_z = None,
    n_chains=16,
    num_burnin_steps=100,
    num_results=500,
    init_step_size=1.0,
    target_acceptance_rate=0.8,
    max_tree_depth=8,
    seed=0
):
    n_devices = jax.device_count()
    n_local_devices = jax.local_device_count()
    process_idx = jax.process_index()
    dim = q_z.event_shape[0]

    n_chains = max(n_devices, (n_chains // n_devices) * n_devices)
    chains_per_device = n_chains // n_devices

    lens_sim = sim.LensSimulator(
        model_seq.phys_model,
        model_seq.sim_config,
        bs=1,
    )

    master_key = jax.random.PRNGKey(seed)
    
    @jit
    def log_prob(z):
        z_batched = z[None, :]
        lp = model_seq.prob_model.log_prob(lens_sim, z_batched)[0]
        return lp[0]

    inv_mass_matrix_diag = np.diag(np.array(q_z.covariance()))

    process_key = jax.random.fold_in(master_key, process_idx)
    local_device_keys = jax.random.split(process_key, n_local_devices)
    
    local_init_states = np.stack([np.array(q_z.sample(chains_per_device, seed=k)) for k in local_device_keys])
    local_device_keys_np = np.array(local_device_keys)

    def warmup_device_chain(states_device, key_device):
        keys = jax.random.split(key_device, states_device.shape[0])

        def run_chain(state_init, key_chain):
            adapt = window_adaptation(
                algorithm=blackjax.nuts,
                logdensity_fn=log_prob,
                is_mass_matrix_diagonal=True,
                initial_step_size=init_step_size,
                target_acceptance_rate=target_acceptance_rate,
                initial_inv_mass_matrix=inv_mass_matrix_diag
            )

            (last_state, params), _ = adapt.run(key_chain, state_init, num_burnin_steps)
            return last_state, params

        states, params = vmap(run_chain)(states_device, keys)
        return states, params

    local_devices = jax.local_devices()
    
    warmup_device_chain_pmap = pmap(warmup_device_chain, devices=local_devices)
    start = time.time()
    print(f'Starting burnin on {n_devices} devices...')
    local_warmup_states, local_warmup_params = warmup_device_chain_pmap(local_init_states, local_device_keys_np)
    print(f'Burnin took {time.time() - start:.1f}s')

    def sample_device_chain(states_device, params_device, key_device):
        keys = jax.random.split(key_device, states_device.position.shape[0])
        
        def run_chain(state, params, key_chain):
            inv_mass_matrix = params["inverse_mass_matrix"]# if no_svi else jnp.linalg.inv(q_z.covariance())
            kernel = blackjax.nuts(
                log_prob,
                params["step_size"],
                inv_mass_matrix,
                # inverse_mass_matrix=jnp.eye(dim) if no_svi else q_z.covariance(),
                max_num_doublings=max_tree_depth
            ).step

            def one_step(carry, key_inner):
                state = carry
                state, _ = kernel(key_inner, state)
                return state, state.position

            inner_keys = jax.random.split(key_chain, num_results)
            _, samples = jax.lax.scan(one_step, state, inner_keys)
            return samples

        samples = vmap(run_chain)(states_device, params_device, keys)
        return samples

    sample_device_chain_pmap = pmap(sample_device_chain, devices=local_devices)
    local_samples = sample_device_chain_pmap(local_warmup_states, local_warmup_params, local_device_keys_np)

    samples = mhu.process_allgather(local_samples)
    
    end = time.time()
    if process_idx == 0: print(f'Sampling took {(end - start):.1f}s')

    samples = jnp.reshape(samples, (-1, num_results, dim))
    return samples