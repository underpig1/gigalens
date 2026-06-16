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

from mams import MAMS

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

# ---- Run LAPS ----
N_CHAINS  = 1024   # divisible by n_devices; bump to 512/1024 if you have more GPUs
N_RESULTS = 3000

laps_samples = MAMS(
    model_seq         = model_seq,
    qz                = None,
    n_hmc          = N_CHAINS,
    num_burnin_steps  = 4000,
    num_results       = N_RESULTS,
    mass_matrix_adapt = False,
    # progress_bar      = False,
    # print_adapt_params= True,
    seed              = 0,
)

# ---- Gather samples from all processes → (num_results, n_chains, dim) ----
all_samples = jax.experimental.multihost_utils.process_allgather(laps_samples, tiled=True)
# process_allgather stacks on axis 0: (n_procs, num_results, n_local, dim)
# tiled=True gives (num_results, n_chains, dim) directly
jax.experimental.multihost_utils.sync_global_devices("done")

# ---- Diagnostics (process 0 only) ----
if jax.process_index() == 0:
    print(f"\nSamples shape: {all_samples.shape}")

    ESS  = blackjax.diagnostics.effective_sample_size(all_samples, chain_axis=1, sample_axis=0)
    Rhat = blackjax.diagnostics.potential_scale_reduction(all_samples, chain_axis=1, sample_axis=0)

    print(f"ESS  — mean: {float(jnp.mean(ESS)):.1f}  min: {float(jnp.min(ESS)):.1f}")
    print(f"Rhat — mean: {float(jnp.mean(Rhat)):.4f}  max: {float(jnp.max(Rhat)):.4f}")

    os.makedirs('results', exist_ok=True)
    np.save('results/laps_samples.npy', np.array(all_samples))
    print("Saved results/laps_samples.npy")

    import corner, matplotlib.pyplot as plt
    flat = np.array(all_samples).reshape(-1, all_samples.shape[-1])[::10]
    fig = corner.corner(flat, plot_datapoints=False)
    fig.savefig('results/corner.png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    print("Saved results/corner.png")
