import jax
jax.distributed.initialize(local_device_ids=None)
if jax.process_index() == 0:
    print('lap it up')

import time
start_time_total = time.perf_counter()
import os
import numpy as np
import jax.numpy as jnp
import blackjax
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.95"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import BackwardProbModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.model import PhysicalModel
from gigalens.jax.profiles.light import sersic, shapelets
from gigalens.jax.profiles.mass import epl, shear

import corner as corner
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
import time
import numpy as np
import optax
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm, kstest
from astropy.visualization import simple_norm
tfd = tfp.distributions
from jax.experimental import shard_map



import sys
sys.path.append('../nuts')
# from mclmc_alt import MCLMC

sys.path.append('../laps')
from laps import LAPS

jax.experimental.multihost_utils.sync_global_devices("init")
import numpy as np
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import ForwardProbModel
from gigalens.model import PhysicalModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear

from astropy.io import fits
from astropy.visualization import simple_norm

obs = np.array(np.load('./imL.npy')).astype(np.float32)
norm = simple_norm(obs, 'sqrt', percent=99.)
kernel = np.array(np.load('./psf.npy')).astype(np.float32)

data94 = fits.open("final_94_drz.fits")
image94 = data94[1].data
exp_time = data94[0].header["EXPTIME"]
data94.close()

lens_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                theta_E=tfd.LogNormal(jnp.log(1.5), 0.25),
                gamma=tfd.TruncatedNormal(2, 0.25, 1, 3),
                e1=tfd.Normal(0, 0.1),
                e2=tfd.Normal(0, 0.1),
                center_x=tfd.Normal(-0.3, 0.15),
                center_y=tfd.Normal(-0.1, 0.15),
            )
        ),
        tfd.JointDistributionNamed(
            dict(gamma1=tfd.Normal(0, 0.15), gamma2=tfd.Normal(0, 0.15))
        ),
    ]
)
lens_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15),
                n_sersic=tfd.Uniform(2, 6),
                e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
                e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
                center_x=tfd.Normal(-0.3, 0.05),
                center_y=tfd.Normal(-0.1, 0.05),
                Ie=tfd.LogNormal(jnp.log(30), 0.5),
            )
        ),
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15),
                n_sersic=tfd.Uniform(4, 8),
                e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
                e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
                center_x=tfd.Normal(-0.3, 0.05),
                center_y=tfd.Normal(-0.1, 0.05),
                Ie=tfd.LogNormal(jnp.log(30), 0.5),
            )
        ),
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15),
                n_sersic=tfd.Uniform(24, 28),
                e1=tfd.TruncatedNormal(0, 0.1, -0.5, 0.5),
                e2=tfd.TruncatedNormal(0, 0.1, -0.5, 0.5),
                center_x=tfd.Normal(-3.4, 0.25),
                center_y=tfd.Normal(-3.1, 0.25),
                Ie=tfd.LogNormal(jnp.log(30), 0.5),
            )
        )
    ]
)

source_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15),
                n_sersic=tfd.Uniform(0.5, 4),
                e1=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
                e2=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
                center_x=tfd.Normal(-0.2, 0.25),
                center_y=tfd.Normal(-0.2, 0.25),
                Ie=tfd.LogNormal(jnp.log(20.0), 0.5),
            )
        ),
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15),
                n_sersic=tfd.Uniform(0.5, 4),
                e1=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
                e2=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
                center_x=tfd.Normal(0.2, 0.25),
                center_y=tfd.Normal(0, 0.25),
                Ie=tfd.LogNormal(jnp.log(20.0), 0.5),
            )
        )
    ]
)

prior = tfd.JointDistributionSequential(
    [lens_prior, lens_light_prior, source_light_prior]
)

sim_config = SimulatorConfig(delta_pix=0.065, num_pix=100, supersample=2, kernel=kernel)
phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=False),sersic.SersicEllipse(use_lstsq=False),sersic.SersicEllipse(use_lstsq=False)], [sersic.SersicEllipse(use_lstsq=False), sersic.SersicEllipse(use_lstsq=False)])
lens_sim = LensSimulator(phys_model, sim_config, bs=1)
prob_model = ForwardProbModel(prior, obs, background_rms=0.007616, exp_time=exp_time)
model_seq = ModellingSequence(phys_model, prob_model, sim_config)
from jax.experimental import shard_map
schedule_fn = optax.polynomial_schedule(init_value=-0.02, end_value=-0.003, power=0.1, transition_steps=1000)
opt = optax.chain(
  optax.scale_by_adam(),
  optax.scale_by_schedule(schedule_fn),
)
import time

# Point A: Start tracking time
start_time = time.perf_counter()

map_estimate = model_seq.MAP(opt, seed=0, num_steps=1000)
lps = prob_model.log_prob(LensSimulator(phys_model, sim_config, bs=500), map_estimate[0])[0]
best = map_estimate[jnp.argmax(lps)][jnp.newaxis,:]
end_time = time.perf_counter()

# Calculate the difference
elapsed_time = end_time - start_time
print(f"Elapsed time for MAPS: {elapsed_time:.6f} seconds")

save = time.perf_counter()
opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
qz, loss_hist = model_seq.SVI(best, opt, n_vi=1000, num_steps=2000)
print('SVI time')
print(abs(save - time.perf_counter()))

jax.experimental.multihost_utils.sync_global_devices("model_ready")

# N_CHAINS  = 16
# N_RESULTS = 2000

# ---- Run 1: LAPS cold start (no qz) ----
# if jax.process_index() == 0:
#     print("\n=== LAPS cold start ===")
# cold_samples = LAPS(
#     model_seq        = model_seq,
#     qz               = qz,
#     n_chains         = N_CHAINS,
#     num_burnin_steps = 2000,
#     num_results      = N_RESULTS,
# )
# cold_all = jax.experimental.multihost_utils.process_allgather(cold_samples, tiled=True)
# jax.experimental.multihost_utils.sync_global_devices("cold_done")

# ESS  = blackjax.diagnostics.effective_sample_size(cold_all, chain_axis=1, sample_axis=0)
# Rhat = blackjax.diagnostics.potential_scale_reduction(cold_all, chain_axis=1, sample_axis=0)
# print(f"  ESS  mean={float(jnp.mean(ESS)):.1f}  min={float(jnp.min(ESS)):.1f}")
# print(f"  Rhat mean={float(jnp.mean(Rhat)):.4f}  max={float(jnp.max(Rhat)):.4f}")

import sys
sys.path.append('../nuts')
import mclmc_alt as mclmc

t = time.time()
samples_MCLMC = mclmc.MCLMC(model_seq, 
                              qz,
                              n_hmc=256, 
                              num_burnin_steps=500, 
                              num_results=1000, 
                              seed=1
                         )
# MCLMC(model_seq, qz, n_hmc=16, num_burnin_steps=1000, num_results=2000, mass_matrix_adapt=True,
#         init_L=None, init_step_size=None, progress_bar=False, print_adapt_params=False,seed=0)

t_sample = time.time() - t
print(f'MCLMC time: {t_sample}')
print(samples_MCLMC.shape)

print('total time')
print(abs(start_time_total - time.perf_counter()))

import corner
import matplotlib.pyplot as plt
import numpy as np

# samples_MCLMC shape: (num_results, n_chains, n_params)
print("samples shape:", samples_MCLMC.shape)

# Move to host memory and flatten chains
samples = np.asarray(samples_MCLMC)
samples_flat = samples.reshape(-1, samples.shape[-1])

# Optional: parameter labels
labels = [
    "theta_E", "gamma", "e1", "e2", "center_x", "center_y",
    "gamma1", "gamma2",
    "lens1_R", "lens1_n", "lens1_e1", "lens1_e2", "lens1_x", "lens1_y", "lens1_Ie",
    "lens2_R", "lens2_n", "lens2_e1", "lens2_e2", "lens2_x", "lens2_y", "lens2_Ie",
    "lens3_R", "lens3_n", "lens3_e1", "lens3_e2", "lens3_x", "lens3_y", "lens3_Ie",
    "src1_R", "src1_n", "src1_e1", "src1_e2", "src1_x", "src1_y", "src1_Ie",
    "src2_R", "src2_n", "src2_e1", "src2_e2", "src2_x", "src2_y", "src2_Ie",
]

# In case the dimensionality changes
if len(labels) != samples_flat.shape[-1]:
    labels = [f"p{i}" for i in range(samples_flat.shape[-1])]

fig = corner.corner(
    samples_flat,
    labels=labels,
    show_titles=True,
    title_fmt=".3f",
    quantiles=[0.16, 0.5, 0.84],
    levels=(0.68, 0.95),
)

fig.savefig("results/mclmc_corner.png", dpi=300, bbox_inches="tight")
plt.close(fig)