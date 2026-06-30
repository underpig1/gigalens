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

sys.path.append('../../laps')
from laps_blackjax import LAPS

jax.experimental.multihost_utils.sync_global_devices("init")
lens_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                theta_E=tfd.LogNormal(jnp.log(2.5), 0.25),
                gamma=tfd.TruncatedNormal(2, 0.25, 1, 2.7),
                e1=tfd.Normal(0, 0.1),
                e2=tfd.Normal(0, 0.1),
                center_x=tfd.Normal(0, 0.05),
                center_y=tfd.Normal(0, 0.05),
            )
        ),
        tfd.JointDistributionNamed(
            dict(gamma1=tfd.Normal(0, 0.05), gamma2=tfd.Normal(0, 0.05))
        ),
    ]
)
lens_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.4), 0.2),
                n_sersic=tfd.Uniform(0.5, 5),
                e1=tfd.TruncatedNormal(0, 0.15, -0.3, 0.3),
                e2=tfd.TruncatedNormal(0, 0.15, -0.3, 0.3),
                center_x=tfd.Normal(3.5, 0.05),
                center_y=tfd.Normal(0.1, 0.05),
            ) #nearby 1
        ),
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.4), 0.2),
                n_sersic=tfd.Uniform(0.5, 5),
                e1=tfd.TruncatedNormal(0, 0.15, -0.3, 0.3),
                e2=tfd.TruncatedNormal(0, 0.15, -0.3, 0.3),
                center_x=tfd.Normal(3.5, 0.05),
                center_y=tfd.Normal(0.1, 0.05),
            )#nearby 2
        ),
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15),
                n_sersic=tfd.Uniform(0.5, 10),
                e1=tfd.Normal(0, 0.15),
                e2=tfd.Normal(0, 0.15),
                center_x=tfd.Normal(0, 0.05),
                center_y=tfd.Normal(0, 0.05),
            )#lens 1
        ),
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15),
                n_sersic=tfd.Uniform(0.5, 10),
                e1=tfd.Normal(0, 0.15),
                e2=tfd.Normal(0, 0.15),
                center_x=tfd.Normal(0, 0.1),
                center_y=tfd.Normal(0, 0.1),
            )#lens 2
        )
    ]
)

source_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15),
                n_sersic=tfd.Uniform(0.5, 6),
                e1=tfd.Normal(0, 0.15),
                e2=tfd.Normal(0, 0.15),
                center_x=tfd.Normal(0, 0.1),
                center_y=tfd.Normal(0, 0.1),
            ) #sersic
        ),
        tfd.JointDistributionNamed(
            dict(
                beta=tfd.LogNormal(jnp.log(0.1), 0.1),
                center_x=tfd.Normal(0, 0.05),
                center_y=tfd.Normal(0, 0.05)
            ) #shape
        )
    ]
)
prior = tfd.JointDistributionSequential(
    [lens_prior, lens_light_prior, source_light_prior]
)

from pathlib import Path
observed_img = np.load(os.path.join(Path(__file__).resolve().parent, "system65.npy"))
observed_img = observed_img[15:135, 10:130]
print(observed_img.shape)

kernel = np.load(os.path.join(Path(__file__).resolve().parent, "psf65.npy")).astype(np.float32)
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=120, supersample=1, kernel=kernel)
n_max=6
phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=True),sersic.SersicEllipse(use_lstsq=True),sersic.SersicEllipse(use_lstsq=True),sersic.SersicEllipse(use_lstsq=True)], [sersic.SersicEllipse(use_lstsq=True),shapelets.Shapelets(n_max=n_max, use_lstsq=True, interpolate=False)])
phys_model_forward = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=False),sersic.SersicEllipse(use_lstsq=False),sersic.SersicEllipse(use_lstsq=False),sersic.SersicEllipse(use_lstsq=False)], [shapelets.Shapelets(n_max=n_max, use_lstsq=False, interpolate=False)])
lens_sim = LensSimulator(phys_model, sim_config, bs=1)
background_rms = 0.008230474
exp_time = 1197.699462
prob_model = BackwardProbModel(prior, jnp.array(observed_img), background_rms=background_rms, exp_time=exp_time)
model_seq = ModellingSequence(phys_model, prob_model, sim_config)

qz = jnp.load(os.path.join(Path(__file__).resolve().parent, "qz.npz"))
qz = tfd.MultivariateNormalTriL(loc=qz['loc'], scale_tril=qz['scale_tril'])

jax.experimental.multihost_utils.sync_global_devices("model_ready")

N_CHAINS  = 1024
N_RESULTS = 2000

import optax
from jax.experimental import shard_map
schedule_fn = optax.polynomial_schedule(init_value=-1e-2, end_value=-1e-2/3, 
                                      power=0.5, transition_steps=500)
opt = optax.chain(
  optax.scale_by_adam(),
  optax.scale_by_schedule(schedule_fn),
)
map_estimate = model_seq.MAP(opt, seed=0)

# ---- Run 1: LAPS cold start (no qz) ----
if jax.process_index() == 0:
    print("\n=== LAPS cold start ===")
cold_samples = LAPS(
    model_seq        = model_seq,
    qz               = qz,
    n_hmc         = N_CHAINS,
    num_unadjusted_steps = 2000,
    num_adjusted_steps      = N_RESULTS,
)
cold_all = jax.experimental.multihost_utils.process_allgather(cold_samples, tiled=True)
jax.experimental.multihost_utils.sync_global_devices("cold_done")

ESS  = blackjax.diagnostics.effective_sample_size(cold_all, chain_axis=1, sample_axis=0)
Rhat = blackjax.diagnostics.potential_scale_reduction(cold_all, chain_axis=1, sample_axis=0)
print(f"  ESS  mean={float(jnp.mean(ESS)):.1f}  min={float(jnp.min(ESS)):.1f}")
print(f"  Rhat mean={float(jnp.mean(Rhat)):.4f}  max={float(jnp.max(Rhat)):.4f}")