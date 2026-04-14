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
    stop_slow_chains=False,
    check_interval=100,
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
    
    # Track per-chain statistics
    chain_acceptance = jnp.zeros(n_chains)
    chain_divergences = jnp.zeros(n_chains)
    chain_num_steps = jnp.zeros(n_chains)
    
    # Active chain mask (True = still sampling)
    active_chains = jnp.ones(n_chains, dtype=bool)
    stopped_chains = []
    
    if use_pmap:
        final_states_pmap = jax.tree.map(
            lambda x: x.reshape(n_devices, chains_per_device, *x.shape[1:]),
            final_states
        )
        
        def sampling_step_device(state, key):
            """Single step on one device"""
            keys = jax.random.split(key, chains_per_device)
            new_states, infos = jax.vmap(nuts_kernel)(keys, state)
            return new_states, (new_states.position, infos)
        
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
                current_states, (positions, infos) = sampling_step(current_states, keys_per_device)
                positions_flat = positions.reshape(n_chains, -1)
                samples_list.append(positions_flat)
                
                # Accumulate chain statistics
                infos_flat = jax.tree.map(lambda x: x.reshape(n_chains, *x.shape[2:]), infos)
                chain_acceptance += infos_flat.acceptance_rate
                chain_divergences += infos_flat.is_divergent.astype(jnp.float32)
                chain_num_steps += infos_flat.num_integration_steps
            
            samples = jnp.stack(samples_list, axis=0)
        else:
            def run_sampling(initial_state, keys):
                def scan_step(state, key):
                    keys_per_device = jax.random.split(key, n_devices)
                    new_state, (positions, infos) = sampling_step(state, keys_per_device)
                    positions_flat = positions.reshape(n_chains, -1)
                    infos_flat = jax.tree.map(lambda x: x.reshape(n_chains, *x.shape[2:]), infos)
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
                
                # Check for slow chains periodically
                if stop_slow_chains and (step_idx + 1) % check_interval == 0 and step_idx > 0:
                    chain_avg_steps_so_far = chain_num_steps / (step_idx + 1)
                    mean_steps = jnp.mean(chain_avg_steps_so_far[active_chains])
                    std_steps = jnp.std(chain_avg_steps_so_far[active_chains])
                    
                    # Mark chains as inactive if they're >2.5σ slow
                    slow_threshold = mean_steps + 2.5 * std_steps
                    newly_stopped = jnp.where(
                        active_chains & (chain_avg_steps_so_far > slow_threshold)
                    )[0]
                    
                    if len(newly_stopped) > 0:
                        for chain_id in newly_stopped:
                            active_chains = active_chains.at[chain_id].set(False)
                            stopped_chains.append((int(chain_id), step_idx + 1))
                        print(f"\n⚠️  Stopped {len(newly_stopped)} slow chain(s) at step {step_idx + 1}: {list(newly_stopped)}")
            
            samples = jnp.stack(samples_list, axis=0)
        else:
            if stop_slow_chains:
                print("⚠️  Warning: stop_slow_chains requires progress_bar=True. Ignoring early stopping.")
            
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
    print(f"Total effective samples: {num_results * n_chains}")
    print(f"Samples/second: {num_results * n_chains / sampling_time:.1f}")
    
    # Per-chain diagnostics
    print("\n" + "="*60)
    print("PER-CHAIN DIAGNOSTICS")
    print("="*60)
    
    if len(stopped_chains) > 0:
        print(f"\n🛑 EARLY STOPPED CHAINS: {len(stopped_chains)}")
        for chain_id, stop_step in stopped_chains:
            print(f"   Chain {chain_id}: stopped at step {stop_step}/{num_results}")
    
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
        # samples shape: (num_results, n_chains, dim)
        # blackjax expects: sample_axis=0, chain_axis=1
        ess_bulk = blackjax.diagnostics.effective_sample_size(samples, chain_axis=1, sample_axis=0)
        
        ess_bulk_min = float(jnp.min(ess_bulk))
        ess_bulk_mean = float(jnp.mean(ess_bulk))
        
        print(f"ESS (bulk): min={ess_bulk_min:.1f}, mean={ess_bulk_mean:.1f}")
        print(f"ESS/sec (bulk): {ess_bulk_mean / sampling_time:.1f}")
        
        # Calculate gradient evaluations from integration steps
        # NUTS uses num_integration_steps per sample (stored in chain_num_steps)
        total_grads = float(jnp.sum(chain_num_steps))
        avg_grads_per_sample = total_grads / (num_results * n_chains)
        
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
        'stopped_chains': jnp.array([c[0] for c in stopped_chains]) if stopped_chains else jnp.array([]),
        'active_chains': active_chains,
    }
    
    # Auto-filter stopped chains if early stopping was used
    if len(stopped_chains) > 0:
        print(f"\n🔧 Auto-filtering {len(stopped_chains)} stopped chains from results...")
        chains_to_keep = [i for i in range(n_chains) if active_chains[i]]
        samples = samples[:, chains_to_keep, :]
        print(f"   Returning samples from {len(chains_to_keep)} active chains")
    
    return samples, chain_diagnostics


def filter_chains(samples, chain_diagnostics, exclude_slow=True, exclude_divergent=True):
    """
    Filter out problematic chains from samples.
    
    Args:
        samples: Array of shape (num_results, n_chains, dim)
        chain_diagnostics: Dictionary returned by NUTS function
        exclude_slow: Whether to exclude chains with >2σ integration steps
        exclude_divergent: Whether to exclude chains with >1% divergence rate
    
    Returns:
        Filtered samples array with problematic chains removed
    """
    chains_to_exclude = set()
    
    if exclude_slow:
        chains_to_exclude.update(chain_diagnostics['slow_chains'].tolist())
    
    if exclude_divergent:
        chains_to_exclude.update(chain_diagnostics['divergent_chains'].tolist())
    
    if len(chains_to_exclude) == 0:
        print("No chains to exclude.")
        return samples
    
    n_chains = samples.shape[1]
    chains_to_keep = [i for i in range(n_chains) if i not in chains_to_exclude]
    
    print(f"\nExcluding {len(chains_to_exclude)} problematic chains: {sorted(chains_to_exclude)}")
    print(f"Keeping {len(chains_to_keep)} chains: {chains_to_keep}")
    
    filtered_samples = samples[:, chains_to_keep, :]
    
    return filtered_samples