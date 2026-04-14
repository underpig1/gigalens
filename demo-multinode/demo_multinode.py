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


def synchronize_value_across_hosts(value):
    """Average a scalar/array value across all hosts"""
    if jax.process_count() == 1:
        return value
    # Sum across all processes, then divide
    total = jax.lax.psum(value, axis_name='hosts')
    return total / jax.process_count()


def gather_from_all_hosts(local_data):
    """Gather data from all hosts into a single array on each host"""
    if jax.process_count() == 1:
        return local_data
    # Use all_gather to collect from all processes
    from jax.experimental import multihost_utils
    return multihost_utils.process_allgather(local_data, tiled=True)


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
        
        # For multi-host: only use LOCAL devices per process
        local_devices = jax.local_devices()
        n_local_devices = len(local_devices)
        
        print(f"Total devices: {n_devices}, Local devices: {n_local_devices}")
        print(f"Local device kinds: {[d.device_kind for d in local_devices]}")

        if n_chains % n_devices != 0:
            old_chains = n_chains
            n_chains = ((n_chains + n_devices - 1) // n_devices) * n_devices
            print(f"   Adjusted n_chains: {old_chains} -> {n_chains}")
        
        chains_per_device = n_chains // n_devices
        chains_per_local_device = n_chains // n_devices  # Same on each host
    else:
        n_devices = 1
        n_local_devices = 1
        chains_per_device = n_chains
        chains_per_local_device = n_chains
    
    lens_sim = sim.LensSimulator(
        model_seq.phys_model,
        model_seq.sim_config,
        bs=1,
    )
    
    def log_prob(z):
        return model_seq.prob_model.log_prob(lens_sim, z)[0]
    
    rng_key = jax.random.key(seed)
    init_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    
    # For multi-host: each process generates its own chains
    if use_pmap and jax.process_count() > 1:
        process_id = jax.process_index()
        n_processes = jax.process_count()
        chains_per_process = n_chains // n_processes
        
        # Fold in process_id so each host gets different samples
        init_key_local = jax.random.fold_in(init_key, process_id)
        initial_positions = qz.sample((chains_per_process,), seed=init_key_local)
        
        print(f"[Process {process_id}] Generating {chains_per_process} initial positions")
    else:
        initial_positions = qz.sample((n_chains,), seed=init_key)
    
    dim = initial_positions.shape[-1]
    
    if init_step_size is None:
        init_step_size = 1.0
    
    temp_nuts = blackjax.nuts(
        logdensity_fn=log_prob,
        step_size=init_step_size,
        inverse_mass_matrix=jnp.eye(dim) if mass_matrix_adapt else qz.covariance(),
    )
    
    # Initialize states for local chains only
    n_local_chains = initial_positions.shape[0]
    init_keys = jax.random.split(init_key, n_local_chains)
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
        
        warmup_keys = jax.random.split(warmup_key, n_local_chains)
        
        if use_pmap:
            # Each process already has its local chains
            # Just reshape for local devices
            local_initial_positions_pmap = initial_positions.reshape(n_local_devices, chains_per_local_device, -1)
            
            # Split keys for local devices
            warmup_key_local = jax.random.fold_in(warmup_key, jax.process_index())
            warmup_keys_per_device = jax.random.split(warmup_key_local, n_local_devices)
            
            def split_for_chains(device_key):
                return jax.random.split(device_key, chains_per_local_device)
            
            warmup_keys_pmap = jax.vmap(split_for_chains)(warmup_keys_per_device)
            
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
                    local_initial_positions_pmap, warmup_keys_pmap
                ))
                print("   Running warmup...")
                starttime = time.perf_counter()
            else:
                run_warmup_pmap = jax.pmap(run_warmup_on_device)
            
            final_states_pmap, final_parameters_pmap = run_warmup_pmap(
                local_initial_positions_pmap, warmup_keys_pmap
            )
            
            # Reshape back - but each process only has its local chains
            final_states = jax.tree.map(
                lambda x: x.reshape(n_local_chains, *x.shape[2:]), 
                final_states_pmap
            )
            if isinstance(final_parameters_pmap, dict):
                final_parameters = {
                    k: v.reshape(n_local_chains, *v.shape[2:]) if v.ndim > 1 else v.reshape(n_local_chains)
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
        print(f"✓ Warmup Time: {warmup_time:.2f}s")
        
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
        
        # Synchronize parameters across hosts
        if jax.process_count() > 1:
            print(f"[Process {jax.process_index()}] Synchronizing parameters across {jax.process_count()} hosts...")
            # Need to use pmap with axis_name for psum to work
            step_size_arr = jnp.array([step_size])
            step_size_arr = jax.pmap(lambda x: jax.lax.psum(x, 'i') / jax.process_count(), axis_name='i')(
                step_size_arr.reshape(1, 1)
            )[0, 0]
            step_size = float(step_size_arr)
            
            # Synchronize mass matrix
            inverse_mass_matrix_sync = jax.pmap(
                lambda x: jax.lax.psum(x, 'i') / jax.process_count(), 
                axis_name='i'
            )(inverse_mass_matrix.reshape(1, -1))[0]
            inverse_mass_matrix = inverse_mass_matrix_sync
        
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
        
        warmup_keys = jax.random.split(warmup_key, n_local_chains)
        inverse_mass_matrix = qz.covariance()
        
        if use_pmap:
            # Reshape local chains for local devices
            local_initial_positions_pmap = initial_positions.reshape(n_local_devices, chains_per_local_device, -1)
            
            warmup_key_local = jax.random.fold_in(warmup_key, jax.process_index())
            warmup_keys_per_device = jax.random.split(warmup_key_local, n_local_devices)
            
            def split_for_chains(device_key):
                return jax.random.split(device_key, chains_per_local_device)
            
            warmup_keys_pmap = jax.vmap(split_for_chains)(warmup_keys_per_device)
            
            def run_warmup_on_device(positions, keys):
                def run_single(pos, key):
                    (state, parameters), _ = warmup.run(key, pos, num_steps=num_warmup_steps)
                    return state, parameters['step_size']
                return jax.vmap(run_single)(positions, keys)
            
            if precompile:
                print("   Compiling...")
                run_warmup_pmap = jax.pmap(run_warmup_on_device)
                _ = jax.block_until_ready(run_warmup_pmap(
                    local_initial_positions_pmap, warmup_keys_pmap
                ))
                print("   Running warmup...")
                starttime = time.perf_counter()
            else:
                run_warmup_pmap = jax.pmap(run_warmup_on_device)
            
            final_states_pmap, step_sizes_pmap = run_warmup_pmap(
                local_initial_positions_pmap, warmup_keys_pmap
            )
            
            final_states = jax.tree.map(
                lambda x: x.reshape(n_local_chains, *x.shape[2:]), 
                final_states_pmap
            )
            step_sizes = step_sizes_pmap.reshape(n_local_chains)
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
        
        # Synchronize step size across hosts
        if jax.process_count() > 1:
            print(f"[Process {jax.process_index()}] Synchronizing step size across {jax.process_count()} hosts...")
            step_size_arr = jnp.array([step_size])
            step_size_arr = jax.pmap(lambda x: jax.lax.psum(x, 'i') / jax.process_count(), axis_name='i')(
                step_size_arr.reshape(1, 1)
            )[0, 0]
            step_size = float(step_size_arr)
        
        print(f"   step_size={step_size:.6f}")
        
        nuts_kernel = blackjax.nuts(
            logdensity_fn=log_prob,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
        ).step
    
    print(f"Starting sampling ({num_results} steps)...")
    starttime = time.perf_counter()
    
    sample_keys = jax.random.split(sample_key, num_results)
    
    # Track per-chain statistics (for local chains only in multi-host)
    chain_acceptance = jnp.zeros(n_local_chains)
    chain_divergences = jnp.zeros(n_local_chains)
    chain_num_steps = jnp.zeros(n_local_chains)
    
    if use_pmap:
        final_states_pmap = jax.tree.map(
            lambda x: x.reshape(n_local_devices, chains_per_local_device, *x.shape[1:]),
            final_states
        )
        
        def sampling_step_device(state, key):
            """Single step on one device"""
            keys = jax.random.split(key, chains_per_local_device)
            new_states, infos = jax.vmap(nuts_kernel)(keys, state)
            return new_states, (new_states.position, infos)
        
        sampling_step = jax.pmap(sampling_step_device)
        
        if precompile:
            print("   Compiling sampling...")
            test_key = sample_keys[0]
            sample_key_local = jax.random.fold_in(test_key, jax.process_index())
            test_keys_per_device = jax.random.split(sample_key_local, n_local_devices)
            _ = jax.block_until_ready(sampling_step(final_states_pmap, test_keys_per_device))
            print("   Compilation complete, running sampling...")
            starttime = time.perf_counter()
        
        if progress_bar:
            from tqdm.auto import tqdm
            samples_list = []
            current_states = final_states_pmap
            
            for key in tqdm(sample_keys, desc="Sampling"):
                sample_key_local = jax.random.fold_in(key, jax.process_index())
                keys_per_device = jax.random.split(sample_key_local, n_local_devices)
                current_states, (positions, infos) = sampling_step(current_states, keys_per_device)
                positions_flat = positions.reshape(n_local_chains, -1)
                samples_list.append(positions_flat)
                
                # Accumulate chain statistics
                infos_flat = jax.tree.map(lambda x: x.reshape(n_local_chains, *x.shape[2:]), infos)
                chain_acceptance += infos_flat.acceptance_rate
                chain_divergences += infos_flat.is_divergent.astype(jnp.float32)
                chain_num_steps += infos_flat.num_integration_steps
            
            samples = jnp.stack(samples_list, axis=0)
        else:
            def run_sampling(initial_state, keys):
                def scan_step(state, key):
                    sample_key_local = jax.random.fold_in(key, jax.process_index())
                    keys_per_device = jax.random.split(sample_key_local, n_local_devices)
                    new_state, (positions, infos) = sampling_step(state, keys_per_device)
                    positions_flat = positions.reshape(n_local_chains, -1)
                    infos_flat = jax.tree.map(lambda x: x.reshape(n_local_chains, *x.shape[2:]), infos)
                    return new_state, (positions_flat, infos_flat)
                
                return jax.lax.scan(scan_step, initial_state, keys)
            
            _, (samples, all_infos) = run_sampling(final_states_pmap, sample_keys)
            
            # Accumulate statistics
            chain_acceptance = jnp.sum(all_infos.acceptance_rate, axis=0)
            chain_divergences = jnp.sum(all_infos.is_divergent.astype(jnp.float32), axis=0)
            chain_num_steps = jnp.sum(all_infos.num_integration_steps, axis=0)
    
    else:
        def sampling_step(state, key):
            keys = jax.random.split(key, n_chains)
            new_states, infos = jax.vmap(nuts_kernel)(keys, state)
            return new_states, (new_states.position, infos)
        
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
            
            for step_idx, key in enumerate(tqdm(sample_keys, desc="Sampling")):
                current_states, (positions, infos) = sampling_step(current_states, key)
                samples_list.append(positions)
                
                # Accumulate chain statistics
                chain_acceptance += infos.acceptance_rate
                chain_divergences += infos.is_divergent.astype(jnp.float32)
                chain_num_steps += infos.num_integration_steps
            
            samples = jnp.stack(samples_list, axis=0)
        else:
            @jax.jit
            def run_sampling(initial_state, keys):
                def scan_step(state, key):
                    new_state, (positions, infos) = sampling_step(state, key)
                    return new_state, (positions, infos)
                return jax.lax.scan(scan_step, initial_state, keys)
            
            _, (samples, all_infos) = run_sampling(final_states, sample_keys)
            
            # Accumulate statistics
            chain_acceptance = jnp.sum(all_infos.acceptance_rate, axis=0)
            chain_divergences = jnp.sum(all_infos.is_divergent.astype(jnp.float32), axis=0)
            chain_num_steps = jnp.sum(all_infos.num_integration_steps, axis=0)
    
    sampling_time = time.perf_counter() - starttime
    print(f"Sampling time: {sampling_time:.2f}s")
    
    # Gather samples from all hosts
    if jax.process_count() > 1:
        print(f"[Process {jax.process_index()}] Gathering results from all hosts...")
        from jax.experimental import multihost_utils
        
        # Gather samples - shape (num_results, n_local_chains, dim) -> (num_results, n_chains, dim)
        samples = multihost_utils.process_allgather(samples, tiled=True)
        
        # Gather chain statistics
        chain_acceptance = multihost_utils.process_allgather(chain_acceptance, tiled=False)
        chain_divergences = multihost_utils.process_allgather(chain_divergences, tiled=False)
        chain_num_steps = multihost_utils.process_allgather(chain_num_steps, tiled=False)
        
        n_total_chains = samples.shape[1]
        print(f"[Process {jax.process_index()}] Gathered {n_total_chains} total chains")
    else:
        n_total_chains = n_local_chains
    
    print(f"Total effective samples: {num_results * n_total_chains}")
    print(f"Samples/second: {num_results * n_total_chains / sampling_time:.1f}")
    
    # Per-chain diagnostics
    print("\n" + "="*60)
    print("PER-CHAIN DIAGNOSTICS")
    print("="*60)
    
    chain_acceptance_rate = chain_acceptance / num_results
    chain_divergence_rate = chain_divergences / num_results
    chain_avg_steps = chain_num_steps / num_results
    
    # Identify problematic chains
    mean_acceptance = jnp.mean(chain_acceptance_rate)
    std_acceptance = jnp.std(chain_acceptance_rate)
    mean_steps = jnp.mean(chain_avg_steps)
    std_steps = jnp.std(chain_avg_steps)
    
    # Flag chains that are >2 std devs away from mean
    slow_chains = jnp.where(chain_avg_steps > mean_steps + 2 * std_steps)[0]
    low_acceptance_chains = jnp.where(chain_acceptance_rate < mean_acceptance - 2 * std_acceptance)[0]
    divergent_chains = jnp.where(chain_divergence_rate > 0.01)[0]  # >1% divergence rate
    
    print(f"\nAcceptance rates: mean={mean_acceptance:.3f}, std={std_acceptance:.3f}")
    print(f"  Range: [{jnp.min(chain_acceptance_rate):.3f}, {jnp.max(chain_acceptance_rate):.3f}]")
    
    print(f"\nAvg integration steps: mean={mean_steps:.1f}, std={std_steps:.1f}")
    print(f"  Range: [{jnp.min(chain_avg_steps):.1f}, {jnp.max(chain_avg_steps):.1f}]")
    
    print(f"\nDivergence rates: mean={jnp.mean(chain_divergence_rate):.3f}")
    print(f"  Range: [{jnp.min(chain_divergence_rate):.3f}, {jnp.max(chain_divergence_rate):.3f}]")
    
    # Report problematic chains
    if len(slow_chains) > 0:
        print(f"\n⚠️  SLOW CHAINS (>2σ steps): {list(slow_chains)}")
        for chain_id in slow_chains:
            print(f"   Chain {chain_id}: {chain_avg_steps[chain_id]:.1f} steps/iter "
                  f"(acceptance={chain_acceptance_rate[chain_id]:.3f})")
    
    if len(low_acceptance_chains) > 0:
        print(f"\n⚠️  LOW ACCEPTANCE (<2σ): {list(low_acceptance_chains)}")
        for chain_id in low_acceptance_chains:
            print(f"   Chain {chain_id}: acceptance={chain_acceptance_rate[chain_id]:.3f} "
                  f"({chain_avg_steps[chain_id]:.1f} steps/iter)")
    
    if len(divergent_chains) > 0:
        print(f"\n⚠️  HIGH DIVERGENCE (>1%): {list(divergent_chains)}")
        for chain_id in divergent_chains:
            print(f"   Chain {chain_id}: {chain_divergence_rate[chain_id]:.1%} divergences "
                  f"({int(chain_divergences[chain_id])} total)")
    
    if len(slow_chains) == 0 and len(low_acceptance_chains) == 0 and len(divergent_chains) == 0:
        print("\n✓ All chains performing well!")
    
    print("="*60)
    
    # Compute ESS metrics
    print("\nComputing ESS metrics...")
    try:
        # samples shape: (num_results, n_total_chains, dim)
        # blackjax expects: sample_axis=0, chain_axis=1
        ess_bulk = blackjax.diagnostics.effective_sample_size(samples, chain_axis=1, sample_axis=0)
        
        ess_bulk_min = float(jnp.min(ess_bulk))
        ess_bulk_mean = float(jnp.mean(ess_bulk))
        
        print(f"ESS (bulk): min={ess_bulk_min:.1f}, mean={ess_bulk_mean:.1f}")
        print(f"ESS/sec (bulk): {ess_bulk_mean / sampling_time:.1f}")
        
        # Calculate gradient evaluations from integration steps
        # NUTS uses num_integration_steps per sample (stored in chain_num_steps)
        total_grads = float(jnp.sum(chain_num_steps))
        avg_grads_per_sample = total_grads / (num_results * n_total_chains)
        
        print(f"\nGradient evaluations: {int(total_grads)}")
        print(f"Gradients/sample: {avg_grads_per_sample:.1f}")
        print(f"Gradients/ESS: {total_grads / ess_bulk_mean:.1f}")
        
    except Exception as e:
        print(f"Warning: Could not compute ESS metrics: {e}")
        import traceback
        traceback.print_exc()
    
    # Return chain diagnostics as well
    chain_diagnostics = {
        'acceptance_rate': chain_acceptance_rate,
        'divergence_rate': chain_divergence_rate,
        'avg_steps': chain_avg_steps,
        'slow_chains': slow_chains,
        'low_acceptance_chains': low_acceptance_chains,
        'divergent_chains': divergent_chains,
    }
    
    return samples, chain_diagnostics