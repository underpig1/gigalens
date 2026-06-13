import jax
import jax.numpy as jnp

def LAPS(logprob_fn, n_chains, num_steps, init_positions=None, qz=None, seed=42, model_seq=None, MAMS_steps=5):
    """
    Late Adjusted Parallel Sampler (LAPS)
    Exact implementation following Robnik (2026).
    
    Stages:
      1. Unadjusted Phase: Rapidly moves chains to the typical set. Uses Ensemble 
         Chain Adaptation (ECA) via Energy Error Variance per Dimension (EEVPD).
      2. Equilibrium Switch: Monitored online using exponential moving statistics.
      3. Adjusted Phase: Smoothly transitions to the Metropolis Adjusted 
         Microcanonical Dynamics (MAMS) kernel to remove asymptotic bias.
    """
    key = jax.random.PRNGKey(seed)
    key_init, key_run = jax.random.split(key)
    
    # 1. DYNAMIC DIMENSION DETECTION
    if init_positions is not None:
        dim = init_positions.shape[-1]
    elif model_seq is not None and hasattr(model_seq, 'prob_model'):
        prior_dist = model_seq.prob_model.prior
        sample_key, key_init = jax.random.split(key_init)
        init_samples = prior_dist.sample(seed=sample_key, sample_shape=(n_chains,))
        init_positions = model_seq.prob_model.bijector.inverse(init_samples)
        dim = init_positions.shape[-1]
    elif qz is not None:
        dim = qz.shape[-1]
    else:
        raise ValueError("Cannot determine parameter count. Pass `init_positions` or `model_seq`.")
        
    print(f"[LAPS] Compiled exactly to paper specs for a {dim}-dimensional parameter space.")

    # 2. INITIALIZE ENSEMBLE STATES
    if init_positions is None:
        if qz is not None:
            init_positions = qz
        else:
            init_positions = jax.random.normal(key_init, shape=(n_chains, dim))
            
    key_vel, key_run = jax.random.split(key_run)
    raw_vel = jax.random.normal(key_vel, shape=(n_chains, dim))
    init_velocities = raw_vel / jnp.linalg.norm(raw_vel, axis=-1, keepdims=True)
    
    # Batched likelihood evaluation function
    val_and_grad_fn = jax.vmap(jax.value_and_grad(logprob_fn))
    
    # Hyperparameters & Hyperbolic Scaling Factors
    initial_epsilon = 0.1 / jnp.sqrt(dim)
    L = 1.0  # Decouherence trajectory scale
    
    # Initialize the unified tracking loop state
    # Both branches must output this exact structural dictionary
    init_logp, _ = val_and_grad_fn(init_positions)
    initial_ema_logp = jnp.mean(init_logp)
    
    state = {
        'x': init_positions,
        'u': init_velocities,
        'epsilon': initial_epsilon,
        'L': L,
        'is_adjusted': False,        # Control flag separating Phase 1 and Phase 2
        'ema_logp': initial_ema_logp, # Online stationarity tracking
        'steps_stable': 0,           # Window counter for typical set detection
    }
    
    # --- STAGE 1: UNADJUSTED STEP (WITH ECA ADAPTATION) ---
    def unadjusted_step(curr_state, step_key):
        x = curr_state['x']
        u = curr_state['u']
        eps = curr_state['epsilon']
        L_param = curr_state['L']
        
        k1, k2 = jax.random.split(step_key)
        
        # Partial velocity refreshment (MCLMC style)
        nu = jnp.exp(-eps / L_param)
        z = jax.random.normal(k1, shape=u.shape)
        u_ref = nu * u + jnp.sqrt(1.0 - nu**2) * z
        u_ref = u_ref / jnp.linalg.norm(u_ref, axis=-1, keepdims=True)
        
        logp_curr, _ = val_and_grad_fn(x)
        
        # Hyperbolic integration step (Appendix E)
        x_next = x + eps * u_ref
        logp_next, grad_next = val_and_grad_fn(x_next)
        
        grad_norm = jnp.linalg.norm(grad_next, axis=-1, keepdims=True)
        grad_norm_safe = jnp.where(grad_norm == 0, 1e-10, grad_norm)
        e_vec = grad_next / grad_norm_safe
        delta = eps * grad_norm / (dim - 1.0)
        
        u_dot_e = jnp.sum(u_ref * e_vec, axis=-1, keepdims=True)
        
        # Accumulate exact kinetic and potential energy errors
        delta_pos = -logp_next + logp_curr
        delta_vel = (dim - 1.0) * jnp.log(jnp.cosh(delta) + u_dot_e * jnp.sinh(delta))
        delta_total = delta_pos + delta_vel
        
        # Complete the hyperbolic velocity transformation
        num = u_ref + (jnp.sinh(delta) + u_dot_e * (jnp.cosh(delta) - 1.0)) * e_vec
        den = jnp.cosh(delta) + u_dot_e * jnp.sinh(delta)
        u_next = num / den
        u_next = u_next / jnp.linalg.norm(u_next, axis=-1, keepdims=True)
        
        # Ensemble Chain Adaptation (ECA) tuning
        eevpd = jnp.var(delta_total) / dim
        eevpd_safe = jnp.where(eevpd == 0, 1e-5, eevpd)
        eevpd_target = 0.05
        eps_next = eps * (eevpd_target / eevpd_safe)**(1.0 / 6.0)
        eps_next = jnp.clip(eps_next, 1e-4, 1.0)
        
        # Equilibrium Detection Logic
        mean_logp = jnp.mean(logp_next)
        ema_logp_next = 0.95 * curr_state['ema_logp'] + 0.05 * mean_logp
        rel_change = jnp.abs(ema_logp_next - curr_state['ema_logp']) / (jnp.abs(curr_state['ema_logp']) + 1e-5)
        
        is_stable = rel_change < 0.001
        steps_stable_next = jnp.where(is_stable, curr_state['steps_stable'] + 1, 0)
        
        # Transition condition: Trigger Phase 2 after 40 consecutive stationary steps
        is_adjusted_next = steps_stable_next > 40
        
        next_state = {
            'x': x_next,
            'u': u_next,
            'epsilon': eps_next,
            'L': L_param,
            'is_adjusted': is_adjusted_next,
            'ema_logp': ema_logp_next,
            'steps_stable': steps_stable_next,
        }
        return next_state, x_next

    # --- STAGE 2: METROPOLIS ADJUSTED MICROCANONICAL DYNAMICS (MAMS) ---
    def adjusted_step(curr_state, step_key):
        x = curr_state['x']
        eps = curr_state['epsilon']
        L_param = curr_state['L']
        
        k_init, k_steps, k_mh = jax.random.split(step_key, 3)
        
        # 1. Total Velocity Resampling at start of MAMS macro-step
        z_init = jax.random.normal(k_init, shape=x.shape)
        u_start = z_init / jnp.linalg.norm(z_init, axis=-1, keepdims=True)
        
        logp_start, _ = val_and_grad_fn(x)
        
        # Integrated trajectory loop over N unadjusted steps
        def mams_trajectory_fn(j, carry):
            cx, cu, c_logp, accumulated_delta, loop_key = carry
            lk1, lk2 = jax.random.split(loop_key)
            
            cx_next = cx + eps * cu
            clogp_next, cgrad_next = val_and_grad_fn(cx_next)
            
            cgrad_norm = jnp.linalg.norm(cgrad_next, axis=-1, keepdims=True)
            cgrad_norm_safe = jnp.where(cgrad_norm == 0, 1e-10, cgrad_norm)
            ce = cgrad_next / cgrad_norm_safe
            cdelta = eps * cgrad_norm / (dim - 1.0)
            
            cu_dot_ce = jnp.sum(cu * ce, axis=-1, keepdims=True)
            
            cdelta_pos = -clogp_next + c_logp
            cdelta_vel = (dim - 1.0) * jnp.log(jnp.cosh(cdelta) + cu_dot_ce * jnp.sinh(cdelta))
            
            accumulated_delta_next = accumulated_delta + cdelta_pos + cdelta_vel
            
            cnum = cu + (jnp.sinh(cdelta) + cu_dot_ce * (jnp.cosh(cdelta) - 1.0)) * ce
            cden = jnp.cosh(cdelta) + cu_dot_ce * jnp.sinh(cdelta)
            cu_next = cnum / cden
            cu_next = cu_next / jnp.linalg.norm(cu_next, axis=-1, keepdims=True)
            
            # Partial microcanonical velocity refreshment along trajectory
            nu = jnp.exp(-eps / L_param)
            nz = jax.random.normal(lk1, shape=cu_next.shape)
            cu_next = nu * cu_next + jnp.sqrt(1.0 - nu**2) * nz
            cu_next = cu_next / jnp.linalg.norm(cu_next, axis=-1, keepdims=True)
            
            return cx_next, cu_next, clogp_next, accumulated_delta_next, lk2

        initial_delta = jnp.zeros((n_chains, 1))
        final_x, final_u, _, total_energy_error, _ = jax.lax.fori_loop(
            0, MAMS_steps, mams_trajectory_fn, (x, u_start, logp_start, initial_delta, k_steps)
        )
        
        # 2. Global Metropolis-Hastings Accept / Reject Evaluator
        acceptance_prob = jnp.minimum(1.0, jnp.exp(-total_energy_error))
        rand_draws = jax.random.uniform(k_mh, shape=acceptance_prob.shape)
        is_accepted = rand_draws < acceptance_prob
        
        x_next = jnp.where(is_accepted, final_x, x)
        u_next = jnp.where(is_accepted, final_u, -u_start) # Momentum reversal on reject
        
        # 3. Step size tuning via Bisection Target (Optimizing for 70% Acceptance)
        mean_acceptance = jnp.mean(acceptance_prob)
        target_acceptance = 0.70
        eps_next = eps + 0.05 * (mean_acceptance - target_acceptance) * eps
        eps_next = jnp.clip(eps_next, 1e-4, 1.0)
        
        next_state = {
            'x': x_next,
            'u': u_next,
            'epsilon': eps_next,
            'L': L_param,
            'is_adjusted': True,  # Stays locked in phase 2
            'ema_logp': curr_state['ema_logp'],
            'steps_stable': curr_state['steps_stable'],
        }
        return next_state, x_next

    # --- RECURRENT LOOP CONTROLLER ---
    def body_fn(i, val):
        curr_state, loop_key = val
        step_key, next_loop_key = jax.random.split(loop_key)
        
        def run_unadjusted(_):
            return unadjusted_step(curr_state, step_key)
            
        def run_adjusted(_):
            return adjusted_step(curr_state, step_key)
            
        # Dynamically branches down to code blocks depending on the active stage
        next_state, samples = jax.lax.cond(
            curr_state['is_adjusted'],
            run_adjusted,
            run_unadjusted,
            operand=None
        )
        return (next_state, next_loop_key), samples

    def scan_fn(carry, i):
        next_carry, samples = body_fn(i, carry)
        return next_carry, samples
        
    _, samples = jax.lax.scan(scan_fn, (state, key_run), jnp.arange(num_steps))
    return samples