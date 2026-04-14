import jax
import jax.numpy as jnp
from jax import jit, pmap, vmap
import blackjax
import time
import gigalens.jax.simulator as sim
from window_adaptation import window_adaptation

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
    # no_svi = q_z is None
    n_devices = jax.device_count()
    n_chains = (n_chains // n_devices) * n_devices
    chains_per_device = n_chains // n_devices

    lens_sim = sim.LensSimulator(
        model_seq.phys_model,
        model_seq.sim_config,
        bs=chains_per_device,
    )

    master_key = jax.random.PRNGKey(seed)
    device_keys = jax.random.split(master_key, n_devices)
    
    @jit
    def log_prob(z):
        z_batched = z[None, :]
        lp = model_seq.prob_model.log_prob(lens_sim, z_batched)[0]
        return lp[0]

    # from jax.tree_util import tree_leaves
    # dim = sum(leaf.size for leaf in tree_leaves(model_seq.prob_model.prior.mean()))

    # init_states = None
    # if no_svi:
    #     init_states = jnp.stack([
    #         jax.random.normal(k, (chains_per_device, dim))
    #         for k in device_keys
    #     ])
    # else:
    init_states = jnp.stack([q_z.sample(chains_per_device, seed=k) for k in device_keys])

    def warmup_device_chain(states_device, key_device):
        keys = jax.random.split(key_device, states_device.shape[0])

        def run_chain(state_init, key_chain):
            adapt = window_adaptation(
                algorithm=blackjax.nuts,
                logdensity_fn=log_prob,
                is_mass_matrix_diagonal=False,
                initial_step_size=init_step_size,
                target_acceptance_rate=target_acceptance_rate,
                initial_inv_mass_matrix=jnp.asarray(q_z.covariance())
            )

            (last_state, params), _ = adapt.run(key_chain, state_init, num_burnin_steps)
            return last_state, params

        states, params = vmap(run_chain)(states_device, keys)
        return states, params

    warmup_device_chain = pmap(warmup_device_chain)
    start = time.time()
    print('Starting burnin')
    warmup_states, warmup_params = warmup_device_chain(init_states, device_keys)
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

    sample_device_chain = pmap(sample_device_chain)
    samples = sample_device_chain(warmup_states, warmup_params, device_keys)

    end = time.time()
    print(f'Sampling took {(end - start):.1f}s')

    samples = jnp.reshape(samples, (n_chains, num_results, -1))
    return samples