"""
GIGALens wrapper around the blackjax LAPS implementation.

Calls laps() from blackjax_laps.py for warmup + bisection, then collects
num_results samples by running the tuned adjusted kernel manually.
"""

import numpy as np
import jax
import jax.numpy as jnp

# blackjax_laps sets up sys.path → local clone; importing it first gives us
# the clone's blackjax for all subsequent imports.
from blackjax_laps import laps

from blackjax.mcmc.adjusted_mclmc import build_kernel as _build_adjusted
from blackjax.mcmc.integrators import (
    generate_isokinetic_integrator,
    mclachlan_coefficients,
    omelyan_coefficients,
)
from blackjax.mcmc.hmc import HMCState


def LAPS(
    model_seq,
    qz=None,
    n_chains=256,
    num_burnin_steps=4000,
    num_results=3000,
    steps_per_sample=15,
    acc_prob_target=0.7,
    diagonal_preconditioning=False,
    seed=0,
    print_adapt_params=False,
    progress_bar=False,
):
    """
    Blackjax LAPS for GIGALens.

    Parameters
    ----------
    model_seq : ModellingSequence
    qz : optional MultivariateNormalTriL — warm start. None = cold start from N(0,1).
    n_chains : int — total chains across all processes.
    num_burnin_steps : int — unadjusted phase steps.
    num_results : int — adjusted samples to return.
    steps_per_sample : int — leapfrog steps per proposal.
    acc_prob_target : float — MH acceptance target for bisection.
    diagonal_preconditioning : bool — passed to laps().
    seed : int
    print_adapt_params : bool

    Returns
    -------
    samples : jnp.ndarray, shape (num_results, n_chains, dim)
        Gathered across all processes.
    """
    import gigalens.jax.simulator as sim_mod

    lens_sim = sim_mod.LensSimulator(model_seq.phys_model, model_seq.sim_config, bs=1)

    def log_prob(z):
        return model_seq.prob_model.log_prob(lens_sim, z)[0]

    # Multi-process setup
    n_procs  = jax.process_count()
    proc_idx = jax.process_index()
    n_local  = n_chains // n_procs
    assert n_chains % n_procs == 0, (
        f"n_chains ({n_chains}) must be divisible by process_count ({n_procs})"
    )

    rng_key                      = jax.random.fold_in(jax.random.key(seed), proc_idx)
    init_key, laps_key, run_key  = jax.random.split(rng_key, 3)

    local_devices = jax.local_devices()
    mesh = jax.sharding.Mesh(np.array(local_devices), axis_names=('chains',))

    # Determine dim and build sample_init callable
    if qz is not None:
        dim = qz.sample(seed=init_key).shape[-1]
        sample_init = lambda key: qz.sample(seed=key)
    else:
        one = model_seq.prob_model.prior.sample(seed=init_key)
        dim = len(model_seq.prob_model.bij.inverse(one))
        sample_init = lambda key: jax.random.normal(key, shape=(dim,))

    if print_adapt_params and proc_idx == 0:
        print(f"LAPS (blackjax): {n_chains} chains ({n_local}/process), dim={dim}")

    # Integrator: omelyan for dim > 200, mclachlan otherwise
    high_dims        = dim > 200
    integrator_coeffs = omelyan_coefficients if high_dims else mclachlan_coefficients
    integrator        = generate_isokinetic_integrator(integrator_coeffs)
    gradient_calls_per_step = len(integrator_coeffs) // 2

    # --- Phase 1 + bisection via laps() ---
    # num_steps2 sized for ~200 adjusted samples — just enough for bisection to
    # converge.  Real samples are collected below with the tuned kernel.
    tune_steps2 = 200 * gradient_calls_per_step * steps_per_sample

    info, grad_calls, acc_used, final_state = laps(
        logdensity_fn=log_prob,
        sample_init=sample_init,
        ndims=dim,
        num_steps1=num_burnin_steps,
        num_steps2=tune_steps2,
        num_chains=n_local,
        mesh=mesh,
        rng_key=laps_key,
        steps_per_sample=steps_per_sample,
        acc_prob=acc_prob_target,
        diagonal_preconditioning=diagonal_preconditioning,
        diagnostics=True,
    )

    # Tuned parameters from the last step of the adjusted phase
    step_size = float(info["phase_2"]["step_size"][-1])
    n_steps   = int(round(float(info["phase_2"]["steps_per_sample"][-1])))

    if print_adapt_params and proc_idx == 0:
        print(
            f"Tuned: step_size={step_size:.6f}  n_steps={n_steps}"
            f"  acc_target={acc_used:.2f}  grad_calls/step={grad_calls}"
        )

    # --- Collect num_results samples with the tuned adjusted kernel ---
    _kernel = _build_adjusted(integrator=integrator)

    def single_step(rng_key, state):
        return _kernel(
            rng_key=rng_key,
            state=state,
            logdensity_fn=log_prob,
            step_size=step_size,
            integration_steps_params=(n_steps,),
            inverse_mass_matrix=1.0,
            L_proposal_factor=jnp.inf,
        )

    kernel_v = jax.jit(jax.vmap(single_step))

    def scan_step(state, rng_key):
        keys      = jax.random.split(rng_key, n_local)
        new_state, _ = kernel_v(keys, state)
        return new_state, new_state.position

    _, samples = jax.lax.scan(
        scan_step,
        final_state,                          # HMCState, leaves shape (n_local, dim)
        jax.random.split(run_key, num_results),
    )
    # samples: (num_results, n_local, dim)

    return jax.experimental.multihost_utils.process_allgather(samples, tiled=True)
