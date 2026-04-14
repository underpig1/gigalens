"""
Optimized NUTS (No U-Turn Sampler) implementation for GIGALens
Includes multi-device parallelism and performance optimizations

References:
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path 
  Lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623.
- BlackJAX documentation: https://blackjax-devs.github.io/blackjax/
"""

import jax.numpy as jnp
import jax
import blackjax
from typing import Optional
import time

import gigalens.jax.simulator as sim


def NUTS(
    model_seq,
    qz,
    n_chains=16,
    num_warmup_steps=1000,
    num_results=2000,
    mass_matrix_adapt=True,
    init_step_size=None,
    target_acceptance_rate=0.8,
    max_tree_depth=10,
    progress_bar=False,
    seed=0,
    use_pmap=False,
    num_devices=None,
    precompile=True,
):
    devices = jax.devices()
    if use_pmap:
        if num_devices is not None:
            devices = devices[:num_devices]
        n_devices = len(devices)
        print(f"Using pmap with {n_devices} devices: {[d.device_kind for d in devices]}")

        if n_chains % n_devices != 0:
            old_chains = n_chains
            n_chains = ((n_chains + n_devices - 1) // n_devices) * n_devices
            # print(f"   Adjusted n_chains: {old_chains} -> {n_chains}")
        
        chains_per_device = n_chains // n_devices
    else:
        n_devices = 1
        chains_per_device = n_chains
    
    lens_sim = sim.LensSimulator(
        model_seq.phys_model,
        model_seq.sim_config,
        bs=1,
    )
    
    def log_prob(z):
        return model_seq.prob_model.log_prob(lens_sim, z)[0]
    
    rng_key = jax.random.key(seed)
    init_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    
    initial_positions = qz.sample((n_chains,), seed=init_key)
    dim = initial_positions.shape[-1]
    
    if init_step_size is None:
        init_step_size = 1.0
    
    temp_nuts = blackjax.nuts(
        logdensity_fn=log_prob,
        step_size=init_step_size,
        inverse_mass_matrix=jnp.eye(dim) if mass_matrix_adapt else qz.covariance(),
    )
    
    init_keys = jax.random.split(init_key, n_chains)
    initial_states = jax.vmap(temp_nuts.init)(initial_positions)
    
    if mass_matrix_adapt:
        print("Starting mass matrix adaptation...")
        starttime = time.perf_counter()
        
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            log_prob,
            target_acceptance_rate=target_acceptance_rate,
            initial_step_size=init_step_size,
        )
        
        warmup_keys = jax.random.split(warmup_key, n_chains)
        
        if use_pmap:
            warmup_keys_per_device = jax.random.split(warmup_key, n_devices)
            
            def split_for_chains(device_key):
                return jax.random.split(device_key, chains_per_device)
            
            warmup_keys_pmap = jax.vmap(split_for_chains)(warmup_keys_per_device)

            initial_positions_pmap = initial_positions.reshape(n_devices, chains_per_device, -1)
            
            def run_warmup_on_device(positions, keys):
                """Runs on single device, vmapped over chains"""
                def run_single(pos, key):
                    (state, parameters), _ = warmup.run(key, pos, num_steps=num_warmup_steps)
                    return state, parameters
                return jax.vmap(run_single)(positions, keys)
            
            if precompile:
                print("   Compiling...")
                run_warmup_pmap = jax.pmap(run_warmup_on_device)
                _ = jax.block_until_ready(run_warmup_pmap(
                    initial_positions_pmap, warmup_keys_pmap
                ))
                print("   Running warmup...")
                starttime = time.perf_counter()
            else:
                run_warmup_pmap = jax.pmap(run_warmup_on_device)
            
            final_states_pmap, final_parameters_pmap = run_warmup_pmap(
                initial_positions_pmap, warmup_keys_pmap
            )
            
            final_states = jax.tree.map(
                lambda x: x.reshape(n_chains, *x.shape[2:]), 
                final_states_pmap
            )
            if isinstance(final_parameters_pmap, dict):
                final_parameters = {
                    k: v.reshape(n_chains, *v.shape[2:]) if v.ndim > 1 else v.reshape(n_chains)
                    for k, v in final_parameters_pmap.items()
                }
            else:
                final_parameters = final_parameters_pmap
        else:
            def run_single_warmup(pos, key):
                (state, parameters), _ = warmup.run(key, pos, num_steps=num_warmup_steps)
                return state, parameters
            
            if precompile:
                print("   Compiling...")
                run_warmup_jit = jax.jit(jax.vmap(run_single_warmup))
                _ = jax.block_until_ready(run_warmup_jit(initial_positions, warmup_keys))
                print("   Running warmup...")
                starttime = time.perf_counter()
            else:
                run_warmup_jit = jax.jit(jax.vmap(run_single_warmup))
            
            final_states, final_parameters = run_warmup_jit(initial_positions, warmup_keys)
        
        warmup_time = time.perf_counter() - starttime
        print(f"Warmup Time: {warmup_time:.2f}s")
        
        if isinstance(final_parameters, dict):
            step_size = jnp.mean(final_parameters['step_size'])
            inverse_mass_matrix = jnp.mean(final_parameters['inverse_mass_matrix'], axis=0)
        else:
            step_sizes = jnp.array([
                p['step_size'] if isinstance(p, dict) else p 
                for p in final_parameters
            ])
            step_size = jnp.mean(step_sizes)
            inverse_mass_matrices = jnp.array([
                p['inverse_mass_matrix'] if isinstance(p, dict) else p 
                for p in final_parameters
            ])
            inverse_mass_matrix = jnp.mean(inverse_mass_matrices, axis=0)
        
        print(f"   step_size={step_size:.6f}")
        print(f"   mass_matrix diagonal=[{jnp.min(jnp.diag(inverse_mass_matrix)):.4f}, "
                f"{jnp.max(jnp.diag(inverse_mass_matrix)):.4f}]")
        
        nuts_kernel = blackjax.nuts(
            logdensity_fn=log_prob,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
        ).step
        
    else:
        print("Starting step size adaptation...")
        starttime = time.perf_counter()
        
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            log_prob,
            is_mass_matrix_diagonal=False,
            target_acceptance_rate=target_acceptance_rate,
            initial_step_size=init_step_size,
        )
        
        warmup_keys = jax.random.split(warmup_key, n_chains)
        inverse_mass_matrix = qz.covariance()
        
        if use_pmap:
            warmup_keys_per_device = jax.random.split(warmup_key, n_devices)
            
            def split_for_chains(device_key):
                return jax.random.split(device_key, chains_per_device)
            
            warmup_keys_pmap = jax.vmap(split_for_chains)(warmup_keys_per_device)
            
            initial_positions_pmap = initial_positions.reshape(n_devices, chains_per_device, -1)
            
            def run_warmup_on_device(positions, keys):
                def run_single(pos, key):
                    (state, parameters), _ = warmup.run(key, pos, num_steps=num_warmup_steps)
                    return state, parameters['step_size']
                return jax.vmap(run_single)(positions, keys)
            
            if precompile:
                print("   Compiling...")
                run_warmup_pmap = jax.pmap(run_warmup_on_device)
                _ = jax.block_until_ready(run_warmup_pmap(
                    initial_positions_pmap, warmup_keys_pmap
                ))
                print("   Running warmup...")
                starttime = time.perf_counter()
            else:
                run_warmup_pmap = jax.pmap(run_warmup_on_device)
            
            final_states_pmap, step_sizes_pmap = run_warmup_pmap(
                initial_positions_pmap, warmup_keys_pmap
            )
            
            final_states = jax.tree.map(
                lambda x: x.reshape(n_chains, *x.shape[2:]), 
                final_states_pmap
            )
            step_sizes = step_sizes_pmap.reshape(n_chains)
        else:
            def run_single_warmup(pos, key):
                (state, parameters), _ = warmup.run(key, pos, num_steps=num_warmup_steps)
                return state, parameters['step_size']
            
            if precompile:
                print("   Compiling...")
                run_warmup_jit = jax.jit(jax.vmap(run_single_warmup))
                _ = jax.block_until_ready(run_warmup_jit(initial_positions, warmup_keys))
                print("   Running warmup...")
                starttime = time.perf_counter()
            else:
                run_warmup_jit = jax.jit(jax.vmap(run_single_warmup))
            
            final_states, step_sizes = run_warmup_jit(initial_positions, warmup_keys)
        
        warmup_time = time.perf_counter() - starttime
        print(f"Warmup Time: {warmup_time:.2f}s")
        
        step_size = jnp.mean(step_sizes)
        
        print(f"   step_size={step_size:.6f}")
        
        nuts_kernel = blackjax.nuts(
            logdensity_fn=log_prob,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
        ).step
    
    print(f"Starting sampling ({num_results} steps)...")
    starttime = time.perf_counter()
    
    sample_keys = jax.random.split(sample_key, num_results)
    
    if use_pmap:
        final_states_pmap = jax.tree.map(
            lambda x: x.reshape(n_devices, chains_per_device, *x.shape[1:]),
            final_states
        )
        
        def sampling_step_device(state, key):
            """Single step on one device"""
            keys = jax.random.split(key, chains_per_device)
            new_states, infos = jax.vmap(nuts_kernel)(keys, state)
            return new_states, new_states.position
        
        sampling_step = jax.pmap(sampling_step_device)
        
        if precompile:
            print("   Compiling sampling...")
            test_key = sample_keys[0]
            test_keys_per_device = jax.random.split(test_key, n_devices)
            _ = jax.block_until_ready(sampling_step(final_states_pmap, test_keys_per_device))
            print("   Compilation complete, running sampling...")
            starttime = time.perf_counter()
        
        if progress_bar:
            from tqdm.auto import tqdm
            samples_list = []
            current_states = final_states_pmap
            
            for key in tqdm(sample_keys, desc="Sampling"):
                keys_per_device = jax.random.split(key, n_devices)
                current_states, positions = sampling_step(current_states, keys_per_device)
                positions_flat = positions.reshape(n_chains, -1)
                samples_list.append(positions_flat)
            
            samples = jnp.stack(samples_list, axis=0)
        else:
            def run_sampling(initial_state, keys):
                def scan_step(state, key):
                    keys_per_device = jax.random.split(key, n_devices)
                    new_state, positions = sampling_step(state, keys_per_device)
                    positions_flat = positions.reshape(n_chains, -1)
                    return new_state, positions_flat
                
                return jax.lax.scan(scan_step, initial_state, keys)
            
            _, samples = run_sampling(final_states_pmap, sample_keys)
    
    else:
        def sampling_step(state, key):
            keys = jax.random.split(key, n_chains)
            new_states, infos = jax.vmap(nuts_kernel)(keys, state)
            return new_states, new_states.position
        
        sampling_step = jax.jit(sampling_step)
        
        if precompile:
            print("   Compiling sampling...")
            _ = jax.block_until_ready(sampling_step(final_states, sample_keys[0]))
            print("   Compilation complete, running sampling...")
            starttime = time.perf_counter()
        
        if progress_bar:
            from tqdm.auto import tqdm
            samples_list = []
            current_states = final_states
            
            for key in tqdm(sample_keys, desc="Sampling"):
                current_states, positions = sampling_step(current_states, key)
                samples_list.append(positions)
            
            samples = jnp.stack(samples_list, axis=0)
        else:
            @jax.jit
            def run_sampling(initial_state, keys):
                return jax.lax.scan(sampling_step, initial_state, keys)
            
            _, samples = run_sampling(final_states, sample_keys)
    
    sampling_time = time.perf_counter() - starttime
    print(f"Sampling time: {sampling_time:.2f}s")
    print(f"Total effective samples: {num_results * n_chains}")
    print(f"Samples/second: {num_results * n_chains / sampling_time:.1f}")
    
    return samples