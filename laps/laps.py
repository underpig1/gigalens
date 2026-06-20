"""
Late Adjusted Parallel Sampler (LAPS)
Robnik & Seljak, arXiv:2601.16696
"""

import time
from typing import Callable

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
from blackjax.types import ArrayTree
from blackjax.util import generate_unit_vector
import gigalens.jax.simulator as sim


# =====================================================================
# 1. DIAGONAL-MASS ISOKINETIC INTEGRATOR
# =====================================================================

def _esh_momentum_update_diag(inv_mass_diag):
    """Isokinetic ESH momentum update with diagonal mass matrix (1-D array)."""
    sqrt_m = jnp.sqrt(inv_mass_diag)

    def update(momentum, logdensity_grad, step_size, coef,
               previous_kinetic_energy_change=None, is_last_call=False):
        del is_last_call
        flat_grad, unravel = ravel_pytree(logdensity_grad)
        flat_mom,  _       = ravel_pytree(momentum)
        d = flat_mom.shape[0]
        g_pre = flat_grad * sqrt_m
        norm_grad, grad_norm = _normalized_flatten_array(g_pre)
        mom_proj  = jnp.dot(flat_mom, norm_grad)
        delta     = step_size * coef * grad_norm / (d - 1)
        zeta      = jnp.exp(-delta)
        raw       = (norm_grad * (1 - zeta) * (1 + zeta + mom_proj * (1 - zeta))
                     + 2 * zeta * flat_mom)
        norm_raw, _ = _normalized_flatten_array(raw)
        ke_change = (delta - jnp.log(2)
                     + jnp.log(1 + mom_proj + (1 - mom_proj) * zeta ** 2)) * (d - 1)
        if previous_kinetic_energy_change is not None:
            ke_change = ke_change + previous_kinetic_energy_change
        return unravel(norm_raw), unravel(norm_raw), ke_change

    return update


def _make_diag_integrator(coefficients):
    def integrator(logdensity_fn: Callable, inverse_mass_matrix=None) -> GeneralIntegrator:
        pos_update = euclidean_position_update_fn(logdensity_fn)
        one_step   = generalized_two_stage_integrator(
            _esh_momentum_update_diag(inverse_mass_matrix),
            pos_update,
            coefficients,
            format_output_fn=format_isokinetic_state_output,
        )
        return one_step
    return integrator


isokinetic_mclachlan_diag = _make_diag_integrator(mclachlan_coefficients)


# =====================================================================
# 2. KERNELS
# =====================================================================

def _unadjusted_kernel(logdensity_fn, inv_mass_diag, integrator):
    """MCLMC step (no MH). Reverts position on divergence. Paper §3."""
    step = with_isokinetic_maruyama(
        integrator(logdensity_fn=logdensity_fn, inverse_mass_matrix=inv_mass_diag)
    )

    def kernel(rng_key, state, L, step_size):
        (position, momentum, logdensity, logdensity_grad), kinetic_change = step(
            state, step_size, L, rng_key
        )
        energy_change = kinetic_change - logdensity + state.logdensity
        ok = jnp.isfinite(energy_change) & (jnp.abs(energy_change) < 1000.0)
        new_state = IntegratorState(
            position        = jnp.where(ok, position,        state.position),
            momentum        = momentum,
            logdensity      = jnp.where(ok, logdensity,      state.logdensity),
            logdensity_grad = jnp.where(ok, logdensity_grad, state.logdensity_grad),
        )
        return new_state, jnp.where(ok, energy_change, 0.0)

    return kernel


def _adjusted_kernel(logdensity_fn, inv_mass_diag, integrator, n_steps=1):
    """Paper §3 MAMS: full velocity resample -> n_steps leapfrog -> MH.
    n_steps is a static int baked in at creation time."""
    step = integrator(logdensity_fn=logdensity_fn, inverse_mass_matrix=inv_mass_diag)

    def kernel(rng_key, state, _L, step_size):
        del _L
        refresh_key, unif_key = jax.random.split(rng_key)
        fresh_mom = generate_unit_vector(refresh_key, state.position)
        refreshed = IntegratorState(
            state.position, fresh_mom, state.logdensity, state.logdensity_grad
        )

        def body(carry, _):
            s, ke_acc = carry
            (pos, mom, ld, ldg), ke = step(s, step_size)
            return (IntegratorState(pos, mom, ld, ldg), ke_acc + ke), None

        (proposed, kinetic_change), _ = jax.lax.scan(
            body, (refreshed, 0.0), None, length=n_steps
        )
        energy_error = kinetic_change - proposed.logdensity + refreshed.logdensity
        accept_prob  = jnp.minimum(1.0, jnp.exp(-energy_error))
        accepted     = jax.random.uniform(unif_key) < accept_prob
        new_state    = jax.lax.cond(accepted, lambda: proposed, lambda: refreshed)
        return new_state, accept_prob

    return kernel


# =====================================================================
# 3. MULTI-CHAIN INIT
# =====================================================================

def _single_init(position, logdensity_fn, rng_key):
    l, g = jax.value_and_grad(logdensity_fn)(position)
    flat_g, unravel = ravel_pytree(g)
    gnorm = jnp.linalg.norm(flat_g)
    # Paper laps_burn_in.py: init along gradient direction (toward high density)
    mom_grad = unravel(flat_g / jnp.where(gnorm > 0, gnorm, 1.0))
    mom_rand = generate_unit_vector(rng_key, position)
    momentum = jax.lax.cond(gnorm > 0, lambda: mom_grad, lambda: mom_rand)
    return IntegratorState(position=position, momentum=momentum,
                           logdensity=l, logdensity_grad=g)


def _init_multi(positions, logdensity_fn, rng_key):
    positions = jnp.asarray(positions)
    rng_keys = jax.random.split(rng_key, positions.shape[0])
    return jax.jit(jax.vmap(lambda p, k: _single_init(p, logdensity_fn, k)))(
        positions, rng_keys
    )


# =====================================================================
# 4. PHASE 1: UNADJUSTED WARMUP
# =====================================================================

def laps_find_hyperparams(log_prob, num_steps, state, rng_key,
                          num_chains, init_step_size, init_L,
                          mass_matrix_adapt=True):
    """
    Unadjusted MCLMC warmup -- paper Algorithm 1 / laps_burn_in.Adaptation.

      Phase 1 kernel: diagonal identity mass matrix (always, per laps_burn_in.py)
      L    = 1.9 * sqrt(mean_i Var_chains[x_i]) * sqrt(d)          Eq. 9
      EEVPD = Var_chains(dH) / d                                    variance not mean-sq
      bias  = mean_i (1 - E_ii)^2,  E_ii = -mean_chains(x_i g_i)  diag equipartition
      EEVPD_wanted = 0.1 * bias^(3/8)                               laps_burn_in.py
      eps  *= clamp((EEVPD_wanted / EEVPD)^(1/6), 0.3, 3.0)        Eq. 7
      diag mass matrix from second-half position variance            paper §3
    """
    dim           = state.position.shape[-1]
    step_size_cap = jnp.sqrt(dim) * 5.0
    identity_diag = jnp.ones(dim)   # Phase 1 always uses identity (laps_burn_in.py)

    kernel_v = jax.jit(jax.vmap(
        _unadjusted_kernel(log_prob, identity_diag, isokinetic_mclachlan_diag),
        in_axes=(0, 0, None, None),
    ))

    def scan_step(carry, rng_keys):
        state, L, step_size = carry
        next_state, energy_changes = kernel_v(rng_keys, state, L, step_size)

        x = next_state.position
        g = next_state.logdensity_grad

        # Mask non-finite chains
        ok   = jnp.all(jnp.isfinite(x), axis=-1) & jnp.all(jnp.isfinite(g), axis=-1)
        n_ok = jnp.maximum(jnp.sum(ok.astype(x.dtype)), 1.0)
        mask = ok[:, None].astype(x.dtype)

        # L from ensemble variance -- Eq. 9
        x_mean = jnp.sum(x * mask, axis=0) / n_ok
        var_x  = jnp.sum((x - x_mean) ** 2 * mask, axis=0) / n_ok
        new_L  = jnp.maximum(1.9 * jnp.sqrt(jnp.mean(var_x)) * jnp.sqrt(dim), 1e-2)

        # EEVPD = Var_chains(dH) / d
        e_mean = jnp.mean(energy_changes)
        eevpd  = jnp.maximum((jnp.mean(energy_changes ** 2) - e_mean ** 2) / dim, 1e-10)

        # Diagonal equipartition bias: mean_i (1 - E_ii)^2, E_ii = -mean_chains(x_i g_i)
        equi_diag    = -jnp.sum(x * g * mask, axis=0) / n_ok   # (dim,)
        bias         = jnp.mean((1.0 - equi_diag) ** 2)
        bias         = jnp.where(jnp.isfinite(bias), bias, 1.0)

        # EEVPD_wanted and step size update
        eevpd_wanted = jnp.maximum(0.1 * bias ** (3.0 / 8.0), 1e-8)
        eps_factor   = jnp.power(eevpd_wanted / eevpd, 1.0 / 6.0)
        eps_factor   = jnp.clip(
            jnp.where(jnp.isfinite(eps_factor), eps_factor, 0.5), 0.3, 3.0
        )
        new_step = jnp.clip(step_size * eps_factor, 1e-4, step_size_cap)

        return (next_state, new_L, new_step), next_state.position

    keys = jax.random.split(rng_key, (num_steps, num_chains))
    (state, L, step_size), positions = jax.lax.scan(
        scan_step, (state, init_L, init_step_size), keys
    )

    inv_mass_diag = identity_diag
    if mass_matrix_adapt:
        flat      = positions[num_steps // 2:].reshape(-1, dim)
        ok_rows   = jnp.all(jnp.isfinite(flat), axis=-1)
        n_ok      = jnp.maximum(jnp.sum(ok_rows.astype(flat.dtype)), 1.0)
        mask2d    = ok_rows[:, None].astype(flat.dtype)
        x_mean    = jnp.sum(flat * mask2d, axis=0) / n_ok
        diag_var  = jnp.sum((flat - x_mean) ** 2 * mask2d, axis=0) / n_ok
        inv_mass_diag = jnp.maximum(diag_var, 1e-6)

    return state, L, step_size, inv_mass_diag


# =====================================================================
# 5. BISECTION FOR ADJUSTED STEP SIZE  (paper §3)
# =====================================================================

def _bisect_step_size(kernel_v, state, L, eps_init, rng_key,
                      target_accept=0.70, pilot_steps=100, tol=0.03, max_iter=20):
    n_local = state.position.shape[0]

    @jax.jit
    def estimate_accept(eps, key):
        chain_keys = jax.random.split(key, (pilot_steps, n_local))
        def scan_fn(s, ks):
            s, ap = kernel_v(ks, s, L, eps)
            return s, ap
        _, accept_probs = jax.lax.scan(scan_fn, state, chain_keys)
        return jnp.mean(accept_probs)

    def acc(e):
        nonlocal rng_key
        rng_key, k = jax.random.split(rng_key)
        return float(estimate_accept(jnp.array(e), k))

    eps = float(eps_init)
    a   = acc(eps)

    if a > target_accept + tol:
        lo, hi = eps, eps * 2.0
        while acc(hi) > target_accept and hi < 1e3:
            lo, hi = hi, hi * 2.0
    elif a < target_accept - tol:
        lo, hi = eps / 2.0, eps
        while acc(lo) < target_accept and lo > 1e-6:
            lo, hi = lo / 2.0, lo
    else:
        return jnp.array(eps)

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        a   = acc(mid)
        if abs(a - target_accept) < tol:
            break
        if a > target_accept:
            lo = mid
        else:
            hi = mid

    return jnp.array(mid)


# =====================================================================
# 6. PHASE 2: ADJUSTED MULTI-CHAIN SAMPLER
# =====================================================================

def _laps_adjusted_multi(logdensity_fn, L, step_size, num_chains,
                         inv_mass_diag) -> SamplingAlgorithm:
    n_steps       = max(1, round(float(L) / float(step_size)))
    kernel_single = _adjusted_kernel(logdensity_fn, inv_mass_diag,
                                     isokinetic_mclachlan_diag, n_steps)
    kernel_v      = jax.vmap(kernel_single, in_axes=(0, 0, None, None))

    def init_fn(positions, rng_key):
        return _init_multi(positions, logdensity_fn, rng_key)

    def update_fn(rng_key, state):
        keys = jax.random.split(rng_key, num_chains)
        return kernel_v(keys, state, L, step_size)

    return SamplingAlgorithm(init_fn, update_fn)


# =====================================================================
# 7. HIGH-LEVEL WRAPPER
# =====================================================================

def LAPS(model_seq, qz=None, n_chains=16, num_burnin_steps=1000, num_results=2000,
         mass_matrix_adapt=True, init_L=None, init_step_size=None,
         progress_bar=False, print_adapt_params=False, seed=0):
    """
    Late Adjusted Parallel Sampler for GIGALens.
    Returns samples of shape (num_results, n_local, dim).
    """
    lens_sim = sim.LensSimulator(model_seq.phys_model, model_seq.sim_config, bs=1)

    def log_prob(z):
        return model_seq.prob_model.log_prob(lens_sim, z)[0]

    n_procs  = jax.process_count()
    proc_idx = jax.process_index()
    n_local  = n_chains // n_procs
    assert n_chains % n_procs == 0, \
        f"n_chains ({n_chains}) must be divisible by process_count ({n_procs})"

    rng_key  = jax.random.fold_in(jax.random.key(seed), proc_idx)
    init_key, tune_key, run_key = jax.random.split(rng_key, 3)

    # --- Initialize chains ---
    if qz is not None:
        init_positions = qz.sample((n_local,), seed=init_key)
    else:
        raw = model_seq.prob_model.prior.sample((n_local,), seed=init_key)
        init_positions = jnp.stack(model_seq.prob_model.bij.inverse(raw)).T

    state = _init_multi(init_positions, log_prob, init_key)
    dim   = state.position.shape[-1]

    # Retry chains stuck at -inf (common with constrained priors on cold start)
    for _ in range(20):
        bad = ~jnp.isfinite(state.logdensity)
        if not bool(jnp.any(bad)):
            break
        init_key, retry_key = jax.random.split(init_key)
        raw_pos   = model_seq.prob_model.prior.sample((n_local,), seed=retry_key)
        new_pos   = model_seq.prob_model.bij.inverse(raw_pos)
        new_state = _init_multi(new_pos, log_prob, retry_key)
        state = jax.tree.map(
            lambda old, new: jnp.where(bad if old.ndim == 1 else bad[:, None], new, old),
            state, new_state,
        )

    L0   = float(jnp.sqrt(dim)) if init_L         is None else float(init_L)
    eps0 = float(jnp.sqrt(dim)) * 0.25 if init_step_size is None else float(init_step_size)

    # --- Phase 1: unadjusted warmup ---
    t0 = time.perf_counter()
    state, L, step_size_warmup, inv_mass_diag = laps_find_hyperparams(
        log_prob          = log_prob,
        num_steps         = num_burnin_steps,
        state             = state,
        rng_key           = tune_key,
        num_chains        = n_local,
        init_step_size    = jnp.array(eps0),
        init_L            = jnp.array(L0),
        mass_matrix_adapt = mass_matrix_adapt,
    )
    L              = jnp.where(jnp.isfinite(L),               L,               L0)
    step_size_warmup = jnp.where(jnp.isfinite(step_size_warmup), step_size_warmup, eps0)
    inv_mass_diag  = jnp.where(jnp.isfinite(inv_mass_diag),   inv_mass_diag,   jnp.ones(dim))

    if proc_idx == 0:
        print(f"Burnin time: {time.perf_counter() - t0:.2f}s")
    if print_adapt_params and proc_idx == 0:
        print(f"WARMUP  L={float(L):.4f}  eps={float(step_size_warmup):.6f}"
              f"  L/eps={float(L)/float(step_size_warmup):.1f}")

    # --- Bisect step size for Phase 2 (paper §3) ---
    adj_kernel_v = jax.jit(jax.vmap(
        _adjusted_kernel(log_prob, inv_mass_diag, isokinetic_mclachlan_diag),
        in_axes=(0, 0, None, None),
    ))
    bis_key, run_key = jax.random.split(run_key)
    step_size = _bisect_step_size(adj_kernel_v, state, L, step_size_warmup, bis_key)

    if print_adapt_params and proc_idx == 0:
        print(f"BISECT  eps={float(step_size):.6f}  L/eps={float(L)/float(step_size):.1f}")

    # --- Phase 2: adjusted sampling ---
    sampling_alg = _laps_adjusted_multi(
        logdensity_fn = log_prob,
        L             = L,
        step_size     = step_size,
        num_chains    = n_local,
        inv_mass_diag = inv_mass_diag,
    )

    t0 = time.perf_counter()
    _, samples = blackjax.util.run_inference_algorithm(
        rng_key             = run_key,
        initial_state       = state,
        inference_algorithm = sampling_alg,
        num_steps           = num_results,
        transform           = lambda s, info: s.position,
        progress_bar        = progress_bar,
    )
    if proc_idx == 0:
        print(f"Sampling time: {time.perf_counter() - t0:.2f}s")

    return samples
