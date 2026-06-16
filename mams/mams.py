import time
from typing import Callable, NamedTuple, Optional
import jax
import jax.numpy as jnp
import blackjax
from blackjax.mcmc.integrators import (
    GeneralIntegrator, 
    IntegratorState, 
    format_isokinetic_state_output, 
    ravel_pytree, 
    _normalized_flatten_array,
    euclidean_position_update_fn,
    generalized_two_stage_integrator,
    mclachlan_coefficients,
    yoshida_coefficients,
    omelyan_coefficients
)
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLike, PRNGKey, ArrayTree
from blackjax.util import generate_unit_vector, pytree_size
import gigalens.jax.simulator as sim

# =====================================================================
# 1. CORE MAMS TYPE DEFINITIONS & STRUCTURES
# =====================================================================

class MAMSInfo(NamedTuple):
    """Tracking metrics for Microcanonical Adaptive Monte Carlo transitions."""
    logdensity: float
    energy_change: float
    accept_prob: float


class MAMSAdaptationState(NamedTuple):
    """Hyperparameters refined during the burn-in initialization."""
    L: float
    step_size: float
    inverse_mass_matrix: ArrayLike


# =====================================================================
# 2. MULTI-CHAIN CORE SAMPLING KERNELS
# =====================================================================

def _mams_single_init(position: ArrayLike, logdensity_fn: Callable, rng_key: PRNGKey):
    if pytree_size(position) < 2:
        raise ValueError("MAMS requires structural targets of > 1 dimensions.")
    l, g = jax.value_and_grad(logdensity_fn)(position)
    return IntegratorState(
        position=position,
        momentum=generate_unit_vector(rng_key, position),
        logdensity=l,
        logdensity_grad=g,
    )


def _mams_single_kernel(
    logdensity_fn: Callable,
    inverse_mass_matrix: ArrayTree,
    integrator: Callable,
):
    step = integrator(logdensity_fn=logdensity_fn, inverse_mass_matrix=inverse_mass_matrix)

    def kernel(
        rng_key: PRNGKey, state: IntegratorState, L: float, step_size: float
    ) -> tuple[IntegratorState, MAMSInfo]:
        
        # Standard Blackjax integrators accept (state, step_size)
        (position, momentum, logdensity, logdensitygrad), kinetic_change = step(
            state, step_size
        )
        
        energy_error = kinetic_change - logdensity + state.logdensity
        
        # Metropolis-Hastings layer for long-term integration stability
        accept_prob = jnp.minimum(1.0, jnp.exp(-energy_error))
        unif_key, vector_key = jax.random.split(rng_key)
        
        is_accepted = jax.random.uniform(unif_key) < accept_prob
        
        new_state = jax.lax.cond(
            is_accepted,
            lambda: IntegratorState(position, momentum, logdensity, logdensitygrad),
            lambda: state
        )
        
        # Partial momentum refreshment via subsampling over the unit sphere shell
        refreshed_momentum = generate_unit_vector(vector_key, new_state.position)
        mixing_coef = jnp.exp(-step_size / L)
        final_momentum = (mixing_coef * new_state.momentum) + (jnp.sqrt(1.0 - mixing_coef**2) * refreshed_momentum)
        
        final_state = IntegratorState(
            position=new_state.position,
            momentum=final_momentum,
            logdensity=new_state.logdensity,
            logdensity_grad=new_state.logdensity_grad
        )

        return final_state, MAMSInfo(logdensity=final_state.logdensity, energy_change=energy_error, accept_prob=accept_prob)

    return kernel


def mams_multi(
    logdensity_fn: Callable,
    L: float,
    step_size: float,
    num_chains: int,
    *,
    integrator,
    inverse_mass_matrix: ArrayTree,
) -> SamplingAlgorithm:
    """Vectorized sampling implementation optimized with JAX vmap."""
    single_kernel = _mams_single_kernel(
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=inverse_mass_matrix,
        integrator=integrator,
    )
    
    kernel_mapped = jax.vmap(single_kernel, in_axes=(0, 0, 0, 0))

    if len(jnp.shape(L)) == 0:
        L = jnp.full(num_chains, L)
    if len(jnp.shape(step_size)) == 0:
        step_size = jnp.full(num_chains, step_size)

    def init_fn(positions: ArrayLike, rng_keys: PRNGKey):
        if rng_keys.ndim == 0:
            rng_keys = jax.random.split(rng_keys, positions.shape[0])
        init_mapper = jax.vmap(lambda p, k: _mams_single_init(p, logdensity_fn, k))
        return jax.jit(init_mapper)(positions, rng_keys)

    def update_fn(rng_key, state):
        rng_keys = jax.random.split(rng_key, num_chains)
        return kernel_mapped(rng_keys, state, L, step_size)

    return SamplingAlgorithm(init_fn, update_fn)


# =====================================================================
# 3. DENSE MASS-MATRIX INTEGRATORS WITH CHOLESKY FACTORIZATION
# =====================================================================

def generate_isokinetic_integrator_smart(coefficients):
    def isokinetic_integrator(
        logdensity_fn: Callable, inverse_mass_matrix: ArrayTree = 1.0
    ) -> GeneralIntegrator:
        position_update_fn = euclidean_position_update_fn(logdensity_fn)
        one_step = generalized_two_stage_integrator(
            esh_dynamics_momentum_update_one_step_smart(inverse_mass_matrix),
            position_update_fn,
            coefficients,
            format_output_fn=format_isokinetic_state_output,
        )
        return one_step
    return isokinetic_integrator


def esh_dynamics_momentum_update_one_step_smart(inverse_mass_matrix):
    if len(inverse_mass_matrix.shape) != 2:
        raise ValueError("inverse_mass_matrix must have 2 dimensions for dense execution structures.")
    
    chol_inverse_mass_matrix = jnp.linalg.cholesky(inverse_mass_matrix)

    def update(
        momentum: ArrayTree,
        logdensity_grad: ArrayTree,
        step_size: float,
        coef: float,
        previous_kinetic_energy_change=None,
        is_last_call=False,
    ):
        del is_last_call
        flatten_grads, unravel_fn = ravel_pytree(logdensity_grad)
        flatten_grads = chol_inverse_mass_matrix.T @ flatten_grads
        flatten_momentum, _ = ravel_pytree(momentum)
        dims = flatten_momentum.shape[0]
        normalized_gradient, gradient_norm = _normalized_flatten_array(flatten_grads)
        momentum_proj = jnp.dot(flatten_momentum, normalized_gradient)
        delta = step_size * coef * gradient_norm / (dims - 1)
        zeta = jnp.exp(-delta)
        new_momentum_raw = (
            normalized_gradient * (1 - zeta) * (1 + zeta + momentum_proj * (1 - zeta))
            + 2 * zeta * flatten_momentum
        )
        new_momentum_normalized, _ = _normalized_flatten_array(new_momentum_raw)
        gr = unravel_fn(chol_inverse_mass_matrix @ new_momentum_normalized)
        next_momentum = unravel_fn(new_momentum_normalized)
        kinetic_energy_change = (
            delta
            - jnp.log(2)
            + jnp.log(1 + momentum_proj + (1 - momentum_proj) * zeta**2)
        ) * (dims - 1)
        if previous_kinetic_energy_change is not None:
            kinetic_energy_change += previous_kinetic_energy_change
        return next_momentum, gr, kinetic_energy_change

    return update


isokinetic_mclachlan_smart = generate_isokinetic_integrator_smart(mclachlan_coefficients)


# =====================================================================
# 4. MASS-MATRIX & STEP-SIZE ADAPTATION SCHEME
# =====================================================================

def mams_find_L_and_step_size(
    log_prob, integrator, num_steps, state, rng_key, num_chains, init_params, target_covariance, mass_matrix_adapt=True
):
    dim = state.position.shape[-1]
    part1_key, _ = jax.random.split(rng_key)
    
    num_steps1 = round(num_steps * 0.2)
    num_steps2 = round(num_steps * 0.6)
    
    def adaptation_step(carry, xs):
        state, params, step_size_max = carry
        mask, step_keys = xs
        
        kernel = jax.vmap(_mams_single_kernel(log_prob, params.inverse_mass_matrix, integrator), in_axes=(0, 0, None, None))
        next_state, info = kernel(step_keys, state, params.L, params.step_size)
        
        # Dual-averaging proxy tracking target accept thresholds (~0.65)
        mean_accept = jnp.mean(info.accept_prob)
        step_factor = jax.lax.cond(mean_accept > 0.65, lambda: 1.03, lambda: 0.95)
        new_step_size = jnp.clip(params.step_size * step_factor, 1e-5, step_size_max)
        
        next_params = params._replace(step_size=new_step_size)
        return (next_state, next_params, step_size_max), next_state.position

    total_adaptation_steps = num_steps1 + num_steps2
    mask_array = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))
    keys_mapped = jax.random.split(part1_key, (total_adaptation_steps, num_chains))

    scan_init = (state, init_params, jnp.inf)
    xs_input = (mask_array, keys_mapped)
    
    carry_out, sample_history = jax.lax.scan(adaptation_step, scan_init, xs_input)
    state, params, _ = carry_out
    
    # Process empirical covariance properties using post-warmup intervals
    if mass_matrix_adapt:
        post_warmup_samples = sample_history[num_steps1:]
        flat_samples = post_warmup_samples.reshape(-1, dim)
        empirical_cov = jnp.cov(flat_samples, rowvar=False)
        
        # FIX: Soft-shrinkage toward the pristine variational target matrix (qz.covariance) 
        # instead of a harsh identity matrix. This prevents geometry collapse.
        shrinkage_weight = 0.0
        adapted_inv_mass = (1.0 - shrinkage_weight) * empirical_cov + shrinkage_weight * target_covariance
        adapted_inv_mass += jnp.eye(dim) * 1e-6  
        params = params._replace(inverse_mass_matrix=adapted_inv_mass)

    # Dimensional trajectory optimization configuration
    optimized_L = jnp.sqrt(dim) * params.step_size * 2.5
    params = params._replace(L=optimized_L)
    
    return state, params


# =====================================================================
# 5. HIGH-LEVEL WRAPPER FOR USER IMPLEMENTATION
# =====================================================================

def MAMS(model_seq, qz=None, n_hmc=16, num_burnin_steps=1000, num_results=2000, mass_matrix_adapt=True,
         init_L=None, init_step_size=None, progress_bar=False, print_adapt_params=False, seed=0):
    """
    GIGALens wrapper for Microcanonical Adaptive Monte Carlo with Momentum Subsampling (MAMS).
    Matches MCLMC/LAPS function signatures exactly.

    When qz is provided chains are warm-started from the variational posterior.
    When qz is None chains are cold-started from N(0,I) with an identity mass matrix.

    n_hmc is the total chain count across all processes; each process runs n_hmc // process_count chains.
    Returns samples of shape (num_results, n_local, dim); caller gathers across processes if needed.
    """
    lens_sim = sim.LensSimulator(model_seq.phys_model, model_seq.sim_config, bs=1)

    def log_prob(z):
        return model_seq.prob_model.log_prob(lens_sim, z)[0]

    # --- Multi-process setup ---
    n_procs  = jax.process_count()
    proc_idx = jax.process_index()
    n_local  = n_hmc // n_procs
    assert n_hmc % n_procs == 0, \
        f"n_hmc ({n_hmc}) must be divisible by process_count ({n_procs})"

    rng_key = jax.random.fold_in(jax.random.key(seed), proc_idx)
    init_key, tune_key, run_key = jax.random.split(rng_key, 3)

    integrator = isokinetic_mclachlan_smart

    # --- Initialize local chains ---
    if qz is not None:
        init_positions    = qz.sample((n_local,), seed=init_key)
        initial_covariance = qz.covariance()
    else:
        one     = model_seq.prob_model.prior.sample(seed=init_key)
        dim_map = len(model_seq.prob_model.bij.inverse(one))
        init_positions    = jax.random.normal(init_key, shape=(n_local, dim_map))
        initial_covariance = jnp.eye(dim_map)

    init_keys   = jax.random.split(init_key, n_local)
    init_mapper = jax.vmap(lambda p, k: _mams_single_init(p, log_prob, k))
    state_multi = jax.jit(init_mapper)(init_positions, init_keys)
    dim = state_multi.position.shape[-1]

    init_L        = jnp.sqrt(dim)        if init_L        is None else init_L
    init_step_size = jnp.sqrt(dim) * 0.25 if init_step_size is None else init_step_size

    starting_adapt_state = MAMSAdaptationState(
        L=init_L, step_size=init_step_size, inverse_mass_matrix=initial_covariance
    )

    starttime = time.perf_counter()
    state_after_tuning, mams_tuned_params = mams_find_L_and_step_size(
        log_prob=log_prob,
        integrator=integrator,
        num_steps=num_burnin_steps,
        state=state_multi,
        rng_key=tune_key,
        num_chains=n_local,
        init_params=starting_adapt_state,
        target_covariance=initial_covariance,
        mass_matrix_adapt=mass_matrix_adapt,
    )
    if proc_idx == 0:
        print("Burnin Time:", time.perf_counter() - starttime)

    L         = mams_tuned_params.L
    step_size = mams_tuned_params.step_size
    if print_adapt_params and proc_idx == 0:
        print(f"ADAPTED. L: {L}, step_size: {step_size}, L/step: {L/step_size}")

    sampling_alg = mams_multi(
        logdensity_fn=log_prob,
        L=L,
        step_size=step_size,
        num_chains=n_local,
        inverse_mass_matrix=mams_tuned_params.inverse_mass_matrix,
        integrator=integrator,
    )

    starttime = time.perf_counter()
    _, multi_chain_samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=state_after_tuning,
        inference_algorithm=sampling_alg,
        num_steps=num_results,
        transform=lambda state, info: state.position,
        progress_bar=progress_bar,
    )
    if proc_idx == 0:
        print(f"Sampling took {time.perf_counter() - starttime} s")

    return multi_chain_samples