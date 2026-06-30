"""
Late Adjusted Parallel Sampler (LAPS)
Robnik & Seljak, arXiv:2601.16696

todos:
- EEVPD formula
- make L not frozen after warmup (bisection)
"""

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
    with_isokinetic_maruyama,
)
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLike, PRNGKey, ArrayTree
from blackjax.util import generate_unit_vector, pytree_size
import gigalens.jax.simulator as sim


class LAPSAdaptationState(NamedTuple):
    L: float
    step_size: float
    inverse_mass_matrix: ArrayLike


# =====================================================================
# mass matrix integrator (based on Linus's mclmc_alt.py)
# =====================================================================

def _esh_momentum_update_smart(inverse_mass_matrix):
    """isokinetic uses full mass matrix"""
    if len(inverse_mass_matrix.shape) != 2:
        raise ValueError("inverse_mass_matrix must be 2D")
    chol = jnp.linalg.cholesky(inverse_mass_matrix)

    def update(momentum, logdensity_grad, step_size, coef,
               previous_kinetic_energy_change=None, is_last_call=False):
        del is_last_call
        flat_grad, unravel = ravel_pytree(logdensity_grad)
        flat_grad = chol.T @ flat_grad
        flat_mom, _ = ravel_pytree(momentum)
        d = flat_mom.shape[0]
        norm_grad, grad_norm = _normalized_flatten_array(flat_grad)
        mom_proj = jnp.dot(flat_mom, norm_grad)
        delta = step_size * coef * grad_norm / (d - 1)
        zeta = jnp.exp(-delta)
        raw = (norm_grad * (1 - zeta) * (1 + zeta + mom_proj * (1 - zeta))
               + 2 * zeta * flat_mom)
        norm_raw, _ = _normalized_flatten_array(raw)
        gr = unravel(chol @ norm_raw)
        next_mom = unravel(norm_raw)
        ke_change = (delta - jnp.log(2)
                     + jnp.log(1 + mom_proj + (1 - mom_proj) * zeta ** 2)) * (d - 1)
        if previous_kinetic_energy_change is not None:
            ke_change = ke_change + previous_kinetic_energy_change
        return next_mom, gr, ke_change

    return update


def _make_isokinetic_integrator(coefficients):
    def integrator(logdensity_fn: Callable,
                   inverse_mass_matrix: ArrayTree = 1.0) -> GeneralIntegrator:
        pos_update = euclidean_position_update_fn(logdensity_fn)
        one_step = generalized_two_stage_integrator(
            _esh_momentum_update_smart(inverse_mass_matrix),
            pos_update,
            coefficients,
            format_output_fn=format_isokinetic_state_output,
        )
        return one_step
    return integrator


isokinetic_mclachlan_smart = _make_isokinetic_integrator(mclachlan_coefficients)


# =====================================================================
# kernels
# =====================================================================

def _unadjusted_kernel(logdensity_fn, inverse_mass_matrix, integrator):
    """mclmc kernel"""
    step = with_isokinetic_maruyama(
        integrator(logdensity_fn=logdensity_fn, inverse_mass_matrix=inverse_mass_matrix)
    )

    def kernel(rng_key, state, L, step_size):
        (position, momentum, logdensity, logdensity_grad), kinetic_change = step(
            state, step_size, L, rng_key
        )
        energy_change = kinetic_change - logdensity + state.logdensity
        ok = jnp.isfinite(energy_change) & (jnp.abs(energy_change) < 1000.0)
        new_state = IntegratorState(
            position        = jnp.where(ok, position,        state.position),
            momentum        = momentum,   # always keep refreshed momentum
            logdensity      = jnp.where(ok, logdensity,      state.logdensity),
            logdensity_grad = jnp.where(ok, logdensity_grad, state.logdensity_grad),
        )
        return new_state, jnp.where(ok, energy_change, 0.0)

    return kernel


def _adjusted_kernel(logdensity_fn, inverse_mass_matrix, integrator):
    """mams kernel"""
    step = integrator(logdensity_fn=logdensity_fn, inverse_mass_matrix=inverse_mass_matrix)

    def kernel(rng_key, state, L, step_size):
        (position, momentum, logdensity, logdensity_grad), kinetic_change = step(state, step_size)
        energy_error = kinetic_change - logdensity + state.logdensity
        accept_prob = jnp.minimum(1.0, jnp.exp(-energy_error))

        unif_key, vec_key = jax.random.split(rng_key)
        accepted = jax.random.uniform(unif_key) < accept_prob
        new_state = jax.lax.cond(
            accepted,
            lambda: IntegratorState(position, momentum, logdensity, logdensity_grad),
            lambda: state,
        )
        refreshed = generate_unit_vector(vec_key, new_state.position)
        mix = jnp.exp(-step_size / L)
        final_mom = mix * new_state.momentum + jnp.sqrt(1.0 - mix ** 2) * refreshed
        final_state = IntegratorState(
            new_state.position, final_mom,
            new_state.logdensity, new_state.logdensity_grad,
        )
        return final_state, accept_prob

    return kernel


# =====================================================================
# init
# =====================================================================

def _single_init(position, logdensity_fn, rng_key):
    l, g = jax.value_and_grad(logdensity_fn)(position)
    return IntegratorState(
        position=position,
        momentum=generate_unit_vector(rng_key, position),
        logdensity=l,
        logdensity_grad=g,
    )


def _init_multi(positions, logdensity_fn, rng_key):
    rng_keys = jax.random.split(rng_key, positions.shape[0])
    mapper = jax.vmap(lambda p, k: _single_init(p, logdensity_fn, k))
    return jax.jit(mapper)(positions, rng_keys)


# =====================================================================
# phase 1
# =====================================================================

def laps_find_hyperparams(log_prob, num_steps, state, rng_key,
                          num_chains, init_params, target_covariance,
                          mass_matrix_adapt=True,
                          desired_energy_var=5e-4):
    """
    sction 3 of the paper
    """
    dim = state.position.shape[-1]
    step_size_cap = jnp.sqrt(dim) * 5.0

    # Precompute vmapped kernel once
    kernel_v = jax.jit(jax.vmap(
        _unadjusted_kernel(log_prob, init_params.inverse_mass_matrix, isokinetic_mclachlan_smart),
        in_axes=(0, 0, None, None),
    ))

    def adaptation_scan_step(carry, rng_keys):
        state, params = carry
        next_state, energy_changes = kernel_v(rng_keys, state, params.L, params.step_size)

        # L from live ensemble variance (paper: L = 2*sqrt(sum_i Var[x_i]))
        ensemble_var = jnp.var(next_state.position, axis=0)   #(dim,)
        new_L = jnp.maximum(2.0 * jnp.sqrt(jnp.sum(ensemble_var)), 1e-2)

        # step size from energy-variance controller
        sq_energy = jnp.mean(energy_changes ** 2)
        xi = sq_energy / (dim * desired_energy_var) + 1e-8
        step_factor = jnp.where(xi < 1.0, 1.02, 0.97)
        new_step = jnp.clip(params.step_size * step_factor, 1e-3, step_size_cap)

        new_params = params._replace(step_size=new_step, L=new_L)
        return (next_state, new_params), next_state.position

    keys = jax.random.split(rng_key, (num_steps, num_chains))
    (state, params), positions = jax.lax.scan(
        adaptation_scan_step,
        (state, init_params),
        keys,
    )
    # positions = (num_steps, num_chains, dim)

    if mass_matrix_adapt:
        #  second half
        flat = positions[num_steps // 2:].reshape(-1, dim)
        emp_cov = jnp.cov(flat, rowvar=False)
        emp_cov = emp_cov + jnp.eye(dim) * 1e-6
        shrink = 0.05
        new_inv_mass = (1.0 - shrink) * emp_cov + shrink * target_covariance
        params = params._replace(inverse_mass_matrix=new_inv_mass)

    return state, params


# =====================================================================
# phase 2
# =====================================================================

def _laps_adjusted_multi(logdensity_fn, L, step_size, num_chains, *,
                         integrator, inverse_mass_matrix) -> SamplingAlgorithm:
    kernel_single = _adjusted_kernel(logdensity_fn, inverse_mass_matrix, integrator)
    kernel_v = jax.vmap(kernel_single, in_axes=(0, 0, None, None))

    def init_fn(positions, rng_key):
        return _init_multi(positions, logdensity_fn, rng_key)

    def update_fn(rng_key, state):
        keys = jax.random.split(rng_key, num_chains)
        return kernel_v(keys, state, L, step_size)

    return SamplingAlgorithm(init_fn, update_fn)


# =====================================================================
# wrappeR for gigalens
# =====================================================================

def LAPS(model_seq, qz=None, map_estimate=None, n_chains=16, num_burnin_steps=1000, num_results=2000,
         mass_matrix_adapt=True, init_L=None, init_step_size=None,
         devices=None, progress_bar=False, print_adapt_params=False, seed=0):
    lens_sim = sim.LensSimulator(model_seq.phys_model, model_seq.sim_config, bs=1)

    def log_prob(z):
        return model_seq.prob_model.log_prob(lens_sim, z)[0]

    n_procs   = jax.process_count()
    proc_idx  = jax.process_index()
    n_local   = n_chains // n_procs
    assert n_chains % n_procs == 0, \
        f"n_chains ({n_chains}) must be divisible by process_count ({n_procs})"

    rng_key  = jax.random.fold_in(jax.random.key(seed), proc_idx)
    init_key, tune_key, run_key = jax.random.split(rng_key, 3)

    if qz is not None:
        init_positions = qz.sample((n_local,), seed=init_key)
        init_inv_mass  = qz.covariance()
    elif map_estimate is not None:
        flat_map = jnp.squeeze(map_estimate[0])
        dim_map = flat_map.shape[-1] 
        init_inv_mass = jnp.eye(dim_map) # Or your metric if you have one
        
        # Jitter chains using the scale of your expected distribution, not a microscopic ball
        chol = jnp.linalg.cholesky(init_inv_mass)
        noise = jax.random.normal(init_key, shape=(n_local, dim_map))
        # Scale the noise so the ensemble initial variance is macroscopically healthy (e.g., 0.1 to 0.5)
        init_positions = flat_map + 0.2 * (noise @ chol.T)
    else:
        one     = model_seq.prob_model.prior.sample(seed=init_key)
        dim_map = len(model_seq.prob_model.bij.inverse(one))
        init_positions = jax.random.normal(init_key, shape=(n_local, dim_map))
        init_inv_mass  = jnp.eye(dim_map)

    state = _init_multi(init_positions, log_prob, init_key)
    dim   = state.position.shape[-1]

    L0   = jnp.sqrt(dim) if init_L        is None else init_L
    eps0 = jnp.sqrt(dim) * 0.25 if init_step_size is None else init_step_size

    starting_params = LAPSAdaptationState(
        L=L0, step_size=eps0, inverse_mass_matrix=init_inv_mass
    )

# phase 1
    t0 = time.perf_counter()
    state, tuned_params = laps_find_hyperparams(
        log_prob=log_prob,
        num_steps=num_burnin_steps,
        state=state,
        rng_key=tune_key,
        num_chains=n_local,
        init_params=starting_params,
        target_covariance=init_inv_mass,
        mass_matrix_adapt=mass_matrix_adapt,
    )
    if proc_idx == 0:
        print(f"Burnin (unadjusted phase) time: {time.perf_counter() - t0:.2f}s")

    L         = tuned_params.L
    step_size = tuned_params.step_size
    if print_adapt_params and proc_idx == 0:
        print(f"ADAPTED — L: {float(L):.4f}, step_size: {float(step_size):.6f}, "
              f"L/step: {float(L)/float(step_size):.2f}")

    # pahse 2
    sampling_alg = _laps_adjusted_multi(
        logdensity_fn=log_prob,
        L=L,
        step_size=step_size,
        num_chains=n_local,
        inverse_mass_matrix=tuned_params.inverse_mass_matrix,
        integrator=isokinetic_mclachlan_smart,
    )
    t0 = time.perf_counter()
    _, samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=state,
        inference_algorithm=sampling_alg,
        num_steps=num_results,
        transform=lambda s, info: s.position,
        progress_bar=progress_bar,
    )
    if proc_idx == 0:
        print(f"Sampling (adjusted phase) time: {time.perf_counter() - t0:.2f}s")

    # samples: (num_results, n_local, dim)
    return samples