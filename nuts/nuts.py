"""
NUTS (No U-Turn Sampler) implementation for GIGALens
Based on BlackJAX's NUTS implementation

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
):
    """
    GIGALens-like wrapper for NUTS sampling using BlackJAX.
    
    The No-U-Turn Sampler (NUTS) is an extension of Hamiltonian Monte Carlo (HMC) that
    automatically tunes the number of leapfrog steps by using a recursive algorithm to
    build a set of likely candidate points that spans a wide swath of the target distribution.
    
    Parameters
    ----------
    model_seq : ModellingSequence
        The GIGALens modeling sequence containing the physical and probabilistic models
    qz : Distribution
        Surrogate distribution for initialization
    n_chains : int, default=16
        Number of parallel chains to run
    num_warmup_steps : int, default=1000
        Number of warmup (adaptation) steps
    num_results : int, default=2000
        Number of sampling steps after warmup
    mass_matrix_adapt : bool, default=True
        Whether to adapt the mass matrix during warmup
    init_step_size : float, optional
        Initial step size. If None, will be set to 1.0
    target_acceptance_rate : float, default=0.8
        Target acceptance rate for dual averaging step size adaptation
    max_tree_depth : int, default=10
        Maximum depth of the binary tree built by NUTS
    progress_bar : bool, default=False
        Whether to display a progress bar during sampling
    seed : int, default=0
        Random seed for reproducibility
        
    Returns
    -------
    samples : jnp.array, shape (num_results, n_chains, num_params)
        The NUTS samples from all chains
        
    References
    ----------
    Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path 
    Lengths in Hamiltonian Monte Carlo. JMLR, 15(1), 1593-1623.
    """
    
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
        
        def run_single_warmup(pos, key):
            (state, parameters), _ = warmup.run(key, pos, num_steps=num_warmup_steps)
            return state, parameters
        
        run_warmup_jit = jax.jit(jax.vmap(run_single_warmup))
        final_states, final_parameters = run_warmup_jit(initial_positions, warmup_keys)
        
        warmup_time = time.perf_counter() - starttime
        print(f"Warmup Time: {warmup_time:.2f}s")
        
        if isinstance(final_parameters, dict):
            step_size = jnp.mean(final_parameters['step_size'])
            inverse_mass_matrix = jnp.mean(final_parameters['inverse_mass_matrix'], axis=0)
        else:
            step_sizes = jnp.array([p['step_size'] if isinstance(p, dict) else p for p in final_parameters])
            step_size = jnp.mean(step_sizes)
            inverse_mass_matrices = jnp.array([p['inverse_mass_matrix'] if isinstance(p, dict) else p for p in final_parameters])
            inverse_mass_matrix = jnp.mean(inverse_mass_matrices, axis=0)
        
        print(f"adapted step_size={step_size:.6f}")
        print(f"mass matrix diagonal range=[{jnp.min(jnp.diag(inverse_mass_matrix)):.6f}, "f"{jnp.max(jnp.diag(inverse_mass_matrix)):.6f}]")
        
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
        
        def run_single_warmup(pos, key):
            (state, parameters), _ = warmup.run(key, pos, num_steps=num_warmup_steps)
            return state, parameters['step_size']
        
        run_warmup_jit = jax.jit(jax.vmap(run_single_warmup))
        final_states, step_sizes = run_warmup_jit(initial_positions, warmup_keys)
        
        warmup_time = time.perf_counter() - starttime
        print(f"Warmup Time: {warmup_time:.2f}s")
        
        step_size = jnp.mean(step_sizes)
        
        print(f"adapted step_size={step_size:.6f}")
        
        nuts_kernel = blackjax.nuts(
            logdensity_fn=log_prob,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
        ).step
    
    print(f"Starting sampling ({num_results} steps)...")
    starttime = time.perf_counter()
    
    def sampling_step(state, key):
        keys = jax.random.split(key, n_chains)
        new_states, infos = jax.vmap(nuts_kernel)(keys, state)
        return new_states, new_states.position
    
    sampling_step = jax.jit(sampling_step)
    
    sample_keys = jax.random.split(sample_key, num_results)
    
    if progress_bar:
        from tqdm.auto import tqdm
        final_states = final_states
        samples_list = []
        
        for key in tqdm(sample_keys, desc="Sampling"):
            final_states, positions = sampling_step(final_states, key)
            samples_list.append(positions)
        
        samples = jnp.stack(samples_list, axis=0)
    else:
        @jax.jit
        def run_sampling(initial_state, keys):
            return jax.lax.scan(sampling_step, initial_state, keys)
        
        final_states, samples = run_sampling(final_states, sample_keys)
    
    sampling_time = time.perf_counter() - starttime
    print(f"Sampling took {sampling_time:.2f}s")
    
    # (num_results, n_chains, dim)
    return samples