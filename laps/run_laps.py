import jax
jax.distributed.initialize(local_device_ids=None)
if jax.process_index() == 0:
    print('lap it up')
import os
import numpy as np
import jax.numpy as jnp
import blackjax
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import ForwardProbModel
from gigalens.model import PhysicalModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear

from laps_blackjax import LAPS

jax.experimental.multihost_utils.sync_global_devices("init")
if jax.process_index() == 0:
    print(f"Devices: {jax.devices()}")
    print(f"Local: {jax.local_device_count()}  Total: {jax.device_count()}")

# ---- Priors (same as run_nuts.py) ----
lens_prior = tfd.JointDistributionSequential([
    tfd.JointDistributionNamed(dict(
        theta_E=tfd.LogNormal(jnp.log(1.25), 0.25),
        gamma=tfd.TruncatedNormal(2, 0.25, 1, 3),
        e1=tfd.Normal(0, 0.1), e2=tfd.Normal(0, 0.1),
        center_x=tfd.Normal(0, 0.05), center_y=tfd.Normal(0, 0.05),
    )),
    tfd.JointDistributionNamed(dict(
        gamma1=tfd.Normal(0, 0.05), gamma2=tfd.Normal(0, 0.05),
    )),
])
lens_light_prior = tfd.JointDistributionSequential([
    tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15),
        n_sersic=tfd.Uniform(2, 6),
        e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        center_x=tfd.Normal(0, 0.05), center_y=tfd.Normal(0, 0.05),
        Ie=tfd.LogNormal(jnp.log(500.0), 0.3),
    ))
])
source_light_prior = tfd.JointDistributionSequential([
    tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15),
        n_sersic=tfd.Uniform(0.5, 4),
        e1=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
        center_x=tfd.Normal(0, 0.25), center_y=tfd.Normal(0, 0.25),
        Ie=tfd.LogNormal(jnp.log(150.0), 0.5),
    ))
])
prior = tfd.JointDistributionSequential([lens_prior, lens_light_prior, source_light_prior])

# ---- Model ----
home = os.path.expanduser("~/")
gigal_dir = os.path.join(home, 'gigalens/src/gigalens/')
kernel = np.load(gigal_dir + '/assets/psf.npy').astype(np.float32)
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=2, kernel=kernel)
phys_model = PhysicalModel(
    [epl.EPL(50), shear.Shear()],
    [sersic.SersicEllipse(use_lstsq=False)],
    [sersic.SersicEllipse(use_lstsq=False)],
)
observed_img = np.load(gigal_dir + '/assets/demo.npy')
prob_model = ForwardProbModel(prior, observed_img, background_rms=0.2, exp_time=100)
model_seq = ModellingSequence(phys_model, prob_model, sim_config)

# ---- Load qz ----
qz_data = jnp.load(os.path.join(os.path.dirname(__file__), 'qz.npz'))
qz = tfd.MultivariateNormalTriL(loc=qz_data['loc'], scale_tril=qz_data['scale_tril'])

jax.experimental.multihost_utils.sync_global_devices("model_ready")

N_CHAINS  = 2048
N_RESULTS = 2000

# ---- Run 1: LAPS cold start (no qz) ----
if jax.process_index() == 0:
    print("\n=== LAPS cold start ===")
cold_samples = LAPS(
    model_seq        = model_seq,
    qz               = None,
    n_hmc         = N_CHAINS,
    num_unadjusted_steps = 2000,
    num_adjusted_steps      = N_RESULTS,
)
cold_all = jax.experimental.multihost_utils.process_allgather(cold_samples, tiled=True)
jax.experimental.multihost_utils.sync_global_devices("cold_done")

# ---- Run 2: LAPS warm start (qz) ----
if jax.process_index() == 0:
    print("\n=== LAPS warm start (qz) ===")
warm_samples = LAPS(
    model_seq        = model_seq,
    qz               = qz,
    n_hmc         = 16,
    num_unadjusted_steps = 2000,
    num_adjusted_steps      = N_RESULTS,
)
warm_all = jax.experimental.multihost_utils.process_allgather(warm_samples, tiled=True)
jax.experimental.multihost_utils.sync_global_devices("warm_done")

# ---- Diagnostics + corner plot (process 0 only) ----
if jax.process_index() == 0:
    os.makedirs('results', exist_ok=True)

    for label, arr, fname in [
        ("LAPS cold", cold_all, "results/laps_cold_samples.npy"),
        ("LAPS warm", warm_all, "results/laps_warm_samples.npy"),
    ]:
        ESS  = blackjax.diagnostics.effective_sample_size(arr, chain_axis=1, sample_axis=0)
        Rhat = blackjax.diagnostics.potential_scale_reduction(arr, chain_axis=1, sample_axis=0)
        print(f"\n{label}  shape={arr.shape}")
        print(f"  ESS  mean={float(jnp.mean(ESS)):.1f}  min={float(jnp.min(ESS)):.1f}")
        print(f"  Rhat mean={float(jnp.mean(Rhat)):.4f}  max={float(jnp.max(Rhat)):.4f}")
        np.save(fname, np.array(arr))
        print(f"  Saved {fname}")

    import corner
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    dim = cold_all.shape[-1]
    # thin equally so both sets have the same point count
    flat_cold = np.array(cold_all).reshape(-1, dim)[::10]
    flat_warm = np.array(warm_all).reshape(-1, dim)[::10]

    fig = corner.corner(
        flat_warm,
        color="C1",
        plot_datapoints=False,
        plot_density=True,
        fill_contours=False,
    )
    corner.corner(
        flat_cold,
        color="C0",
        plot_datapoints=False,
        plot_density=True,
        fill_contours=False,
        fig=fig,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color="C1", lw=2, label="LAPS warm (qz)"),
            Line2D([0], [0], color="C0", lw=2, label="LAPS cold start"),
        ],
        loc="upper right",
        fontsize=11,
        frameon=True,
    )
    fig.savefig("results/corner.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved results/corner.png")
