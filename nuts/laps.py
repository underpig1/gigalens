"""
Late Adjusted Parallel Sampler (LAPS)
Robnik & Seljak, arXiv:2601.16696

Two-phase parallel sampler:
  Phase 1 (unadjusted): MCLMC dynamics, no MH, ensemble step-size adaptation via energy variance.
  Phase 2 (adjusted):   HMC-style dynamics with MH correction. Step sizes are already tuned
                        so acceptance ≈ 1, matching the "best served warm" intuition.

Returns samples of shape (num_results, num_chains, dim), consistent with MCLMC and MAMS.
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


# =====================================================================
# 1. TYPE DEFINITIONS
# =====================================================================

class LAPSAdaptationState(NamedTuple):
    L: float
    step_size: float
    inverse_mass_matrix: ArrayLike


# =====================================================================
# 2. NON-DIAGONAL MASS MATRIX INTEGRATOR (from mclmc_alt.py)
# =====================================================================

def _esh_momentum_update_smart(inverse_mass_matrix):
    """Isokinetic ESH momentum update with full (non-diagonal) mass matrix."""
    if len(inverse_mass_matrix.shape) != 2:
        raise ValueError("inverse_mass_matrix must be 2D for the smart integrator.")
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
# 3. KERNELS
# =====================================================================

def _unadjusted_kernel(logdensity_fn, inverse_mass_matrix, integrator):
    """MCLMC step with isokinetic Maruyama momentum refreshment, NO MH correction.
    Reverts position on divergence (|ΔH| > 1000 or non-finite) so bad cold-start
    chains don't drive the step-size adaptation to zero."""
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


def _adjusted_kernel(logdensity_fn, inverse_mass_matrix, integrator, n_steps=1):
    """MAMS: full velocity resample → n_steps integration steps → MH accept/reject.

    Paper §3: velocity is fully resampled from the unit sphere before each
    trajectory (no partial refresh with L). n_steps is a static Python int
    (= max(1, round(L/step_size))) baked in at kernel-creation time so that
    jax.lax.scan gets a static length.
    """
    step = integrator(logdensity_fn=logdensity_fn, inverse_mass_matrix=inverse_mass_matrix)

    def kernel(rng_key, state, _L, step_size):
        del _L  # L only used in unadjusted phase; n_steps is baked in at creation
        # Full velocity resample from uniform unit sphere (paper MAMS step 1)
        refresh_key, unif_key = jax.random.split(rng_key)
        fresh_mom = generate_unit_vector(refresh_key, state.position)
        refreshed_state = IntegratorState(
            state.position, fresh_mom, state.logdensity, state.logdensity_grad
        )

        # n_steps integration steps accumulating energy error (paper MAMS step 2-3)
        def body(carry, _):
            s, ke_acc = carry
            (pos, mom, ld, ldg), ke = step(s, step_size)
            return (IntegratorState(pos, mom, ld, ldg), ke_acc + ke), None

        (proposed, kinetic_change), _ = jax.lax.scan(
            body, (refreshed_state, 0.0), None, length=n_steps
        )

        energy_error = kinetic_change - proposed.logdensity + refreshed_state.logdensity
        accept_prob = jnp.minimum(1.0, jnp.exp(-energy_error))
        accepted = jax.random.uniform(unif_key) < accept_prob
        new_state = jax.lax.cond(
            accepted,
            lambda: proposed,
            lambda: refreshed_state,
        )
        return new_state, accept_prob

    return kernel


# =====================================================================
# 4. MULTI-CHAIN INIT
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
# 5. LAPS ADAPTATION (Phase 1: unadjusted warmup)
# =====================================================================

def laps_find_hyperparams(log_prob, num_steps, state, rng_key,
                          num_chains, init_params, target_covariance,
                          mass_matrix_adapt=True,
                          desired_energy_var=5e-4):
    """
    Unadjusted warmup matching paper Algorithm 1:
      - L updated every step: L = 2*sqrt(sum_i Var_ensemble[x_i])  (Eq. 9)
      - Equipartition divergence D̃ computed from ensemble positions + gradients (Eq. 6)
      - EEVPD_wanted = F(0.025·D̃), F(x)=4x^{3/2}/(1+x^{1/2})^2  (Eq. 8)
      - Step size: ε *= (EEVPD_wanted / EEVPD_current)^{1/6}  (Eq. 7)
      - Diagonal mass matrix from second-half position variance  (§3)
    """
    dim = state.position.shape[-1]
    step_size_cap = jnp.sqrt(dim) * 5.0

    # Precompute vmapped kernel once; mass matrix is fixed during the scan and
    # updated only at the end from the collected positions.
    kernel_v = jax.jit(jax.vmap(
        _unadjusted_kernel(log_prob, init_params.inverse_mass_matrix, isokinetic_mclachlan_smart),
        in_axes=(0, 0, None, None),
    ))

    def adaptation_scan_step(carry, rng_keys):
        state, params = carry
        next_state, energy_changes = kernel_v(rng_keys, state, params.L, params.step_size)

        x = next_state.position        # (n_chains, dim)
        g = next_state.logdensity_grad # (n_chains, dim)

        # Mask chains whose position or gradient is non-finite (stuck bad inits)
        pos_ok  = jnp.all(jnp.isfinite(x), axis=-1)   # (n_chains,)
        grad_ok = jnp.all(jnp.isfinite(g), axis=-1)   # (n_chains,)
        n_pos  = jnp.maximum(jnp.sum(pos_ok),  1.0)
        n_grad = jnp.maximum(jnp.sum(grad_ok), 1.0)
        pmask  = pos_ok[:,  None].astype(x.dtype)
        gmask  = grad_ok[:, None].astype(x.dtype)

        # L from live ensemble variance over finite chains (paper Eq. 9)
        x_safe = x * pmask
        mean_x_pos = jnp.sum(x_safe, axis=0) / n_pos
        var_x = jnp.sum((x_safe - mean_x_pos) ** 2 * pmask, axis=0) / n_pos
        new_L = jnp.maximum(2.0 * jnp.sqrt(jnp.sum(var_x)), 1e-2)

        # Equipartition divergence D̃ over finite-grad chains (paper Eq. 6)
        mean_x = jnp.sum(x * gmask, axis=0) / n_grad
        dx = mean_x - x
        V = ((dx * gmask).T @ (g * gmask)) / n_grad   # (dim, dim)
        D_tilde = jnp.sum((jnp.eye(dim) - V) ** 2) / dim

        # EEVPD_wanted = F(C·D̃), C=0.025, F(x) = 4x^{3/2} / (1+x^{1/2})^2 (paper Eq. 8)
        CD = 0.025 * jnp.where(jnp.isfinite(D_tilde), D_tilde, 0.0)
        eevpd_wanted = jnp.maximum(
            4.0 * CD ** 1.5 / (1.0 + jnp.sqrt(CD)) ** 2,
            desired_energy_var,
        )

        # Step size: ε *= (EEVPD_wanted / EEVPD_current)^(1/6) (paper Eq. 7)
        eevpd = jnp.mean(energy_changes ** 2) / dim
        raw_step = params.step_size * (eevpd_wanted / (eevpd + 1e-8)) ** (1.0 / 6.0)
        new_step = jnp.clip(
            jnp.where(jnp.isfinite(raw_step), raw_step, params.step_size),
            1e-3, step_size_cap,
        )

        new_params = params._replace(step_size=new_step, L=new_L)
        return (next_state, new_params), next_state.position

    keys = jax.random.split(rng_key, (num_steps, num_chains))
    (state, params), positions = jax.lax.scan(
        adaptation_scan_step,
        (state, init_params),
        keys,
    )
    # positions: (num_steps, num_chains, dim)

    if mass_matrix_adapt:
        # Diagonal preconditioning (paper §3): rescale each coordinate by its std dev.
        flat = positions[num_steps // 2:].reshape(-1, dim)
        diag_var = jnp.maximum(jnp.var(flat, axis=0), 1e-6)
        target_diag = jnp.diag(target_covariance)   # target_covariance is always 2D
        shrink = 0.05
        new_inv_mass = jnp.diag((1.0 - shrink) * diag_var + shrink * target_diag)
        params = params._replace(inverse_mass_matrix=new_inv_mass)

    return state, params


# =====================================================================
# 6. BISECT FOR ADJUSTED-PHASE STEP SIZE (paper §3)
# =====================================================================

def _bisect_step_size(kernel_v, state, L, eps_init, rng_key,
                      target_accept=0.70, pilot_steps=100, tol=0.03, max_iter=20):
    """Find step size for the adjusted phase by bisection targeting `target_accept`.

    Paper: double/halve until bracketed, then bisect until within `tol` of target.
    Uses short pilot runs from the current warmup state to estimate acceptance.
    """
    n_local = state.position.shape[0]

    @jax.jit
    def estimate_accept(eps, key):
        chain_keys = jax.random.split(key, (pilot_steps, n_local))
        def scan_fn(s, ks):
            s, ap = kernel_v(ks, s, L, eps)
            return s, ap
        _, accept_probs = jax.lax.scan(scan_fn, state, chain_keys)
        return jnp.mean(accept_probs)

    eps = float(eps_init)

    def acc(e):
        nonlocal rng_key
        rng_key, k = jax.random.split(rng_key)
        return float(estimate_accept(jnp.array(e), k))

    a = acc(eps)

    # Find bracket by doubling or halving
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

    # Bisect
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        a = acc(mid)
        if abs(a - target_accept) < tol:
            break
        if a > target_accept:
            lo = mid
        else:
            hi = mid

    return jnp.array(mid)


# =====================================================================
# 7. MULTI-CHAIN ADJUSTED SAMPLING (Phase 2)
# =====================================================================

def _laps_adjusted_multi(logdensity_fn, L, step_size, num_chains, *,
                         integrator, inverse_mass_matrix) -> SamplingAlgorithm:
    # n_steps is a static Python int so jax.lax.scan gets a static length
    n_steps = max(1, round(float(L) / float(step_size)))
    kernel_single = _adjusted_kernel(logdensity_fn, inverse_mass_matrix, integrator, n_steps)
    kernel_v = jax.vmap(kernel_single, in_axes=(0, 0, None, None))

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
         devices=None, progress_bar=False, print_adapt_params=False, seed=0):
    """
    Late Adjusted Parallel Sampler for GIGALens.

    Replaces MAP + SVI + MCMC in a single call. When qz is provided it is used
    for warm initialization only (no SVI required).  When qz is None, chains are
    seeded from the prior.

    Returns
    -------
    samples : jnp.ndarray, shape (num_results, num_chains, dim)
    """
    lens_sim = sim.LensSimulator(model_seq.phys_model, model_seq.sim_config, bs=1)

    def log_prob(z):
        return model_seq.prob_model.log_prob(lens_sim, z)[0]

    # --- Multi-process setup ---
    # Each process handles its own chain slice independently (1 GPU per process).
    # n_chains is the total; each process runs n_chains // process_count locally.
    n_procs   = jax.process_count()
    proc_idx  = jax.process_index()
    n_local   = n_chains // n_procs
    assert n_chains % n_procs == 0, \
        f"n_chains ({n_chains}) must be divisible by process_count ({n_procs})"

    # Per-process seed so each process gets a different chain slice
    rng_key  = jax.random.fold_in(jax.random.key(seed), proc_idx)
    init_key, tune_key, run_key = jax.random.split(rng_key, 3)

    # --- Initialize local chains ---
    if qz is not None:
        init_positions = qz.sample((n_local,), seed=init_key)
        init_inv_mass  = qz.covariance()
    else:
        one     = model_seq.prob_model.prior.sample(seed=init_key)
        dim_map = len(model_seq.prob_model.bij.inverse(one))
        init_positions = jax.random.normal(init_key, shape=(n_local, dim_map))
        init_inv_mass  = jnp.eye(dim_map)

    state = _init_multi(init_positions, log_prob, init_key)
    dim   = state.position.shape[-1]

    # Re-initialize chains whose log_prob is non-finite at startup.
    # Chains stuck at NaN from the first step can never recover via the divergence guard.
    for _ in range(20):
        bad = ~jnp.isfinite(state.logdensity)   # (n_local,)
        if not bool(jnp.any(bad)):
            break
        init_key, retry_key = jax.random.split(init_key)
        new_pos = jax.random.normal(retry_key, shape=(n_local, dim))
        new_state = _init_multi(new_pos, log_prob, retry_key)
        state = jax.tree.map(
            lambda old, new: jnp.where(bad[:] if old.ndim == 1 else bad[:, None], new, old),
            state, new_state,
        )

    L0   = jnp.sqrt(dim) if init_L        is None else init_L
    eps0 = jnp.sqrt(dim) * 0.25 if init_step_size is None else init_step_size

    starting_params = LAPSAdaptationState(
        L=L0, step_size=eps0, inverse_mass_matrix=init_inv_mass
    )

    # --- Phase 1: Unadjusted warmup (vmap over local chains) ---
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

    # Fall back to initial guesses if adaptation produced NaN (e.g. all chains bad at init)
    L = jnp.where(jnp.isfinite(tuned_params.L), tuned_params.L, L0)
    step_size_warmup = jnp.where(jnp.isfinite(tuned_params.step_size), tuned_params.step_size, eps0)
    tuned_params = tuned_params._replace(L=L, step_size=step_size_warmup)
    if print_adapt_params and proc_idx == 0:
        print(f"WARMUP — L: {float(L):.4f}, step_size: {float(step_size_warmup):.6f}")

    # --- Bisect for adjusted-phase step size targeting 70% acceptance (paper §3) ---
    adj_kernel_v = jax.jit(jax.vmap(
        _adjusted_kernel(log_prob, tuned_params.inverse_mass_matrix, isokinetic_mclachlan_smart),
        in_axes=(0, 0, None, None),
    ))
    bis_key, run_key = jax.random.split(run_key)
    step_size = _bisect_step_size(adj_kernel_v, state, L, tuned_params.step_size, bis_key)
    if print_adapt_params and proc_idx == 0:
        print(f"BISECTED — step_size: {float(step_size):.6f}, L/step: {float(L)/float(step_size):.2f}")

    # --- Phase 2: Adjusted sampling (vmap over local chains, scan over steps) ---
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

    # samples: (num_results, n_local, dim) — caller gathers across processes if needed
    return samples