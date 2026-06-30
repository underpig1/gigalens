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
map_estimate = model_seq.MAP(opt, seed=0, num_steps=1000)
lps = prob_model.log_prob(LensSimulator(phys_model, sim_config, bs=500), map_estimate[0])[0]
best = map_estimate[jnp.argmax(lps)][jnp.newaxis,:]

schedule_fn = optax.polynomial_schedule(init_value=-1e-7, end_value=-2e-3, power=2, transition_steps=300)
# opt = optax.chain(
#   optax.scale_by_adam(),
#   optax.scale_by_schedule(schedule_fn),
# )
opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
qz, loss_hist = model_seq.SVI(best, opt, n_vi=1000, num_steps=2000)

jax.experimental.multihost_utils.sync_global_devices("model_ready")

N_CHAINS  = 16
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

schedule_fn = optax.polynomial_schedule(init_value=-1e-7, end_value=-2e-3, power=2, transition_steps=300)
# opt = optax.chain(
#   optax.scale_by_adam(),
#   optax.scale_by_schedule(schedule_fn),
# )
opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
qz, loss_hist = model_seq.SVI(best, opt, n_vi=1000, num_steps=2000)

# ---- Run 1: LAPS cold start (no qz) ----
if jax.process_index() == 0:
    print("\n=== LAPS cold start ===")
cold_samples = LAPS(
    model_seq        = model_seq,
    qz               = qz,
    n_chains         = N_CHAINS,
    num_burnin_steps = 2000,
    num_results      = N_RESULTS,
)
cold_all = jax.experimental.multihost_utils.process_allgather(cold_samples, tiled=True)
jax.experimental.multihost_utils.sync_global_devices("cold_done")

ESS  = blackjax.diagnostics.effective_sample_size(cold_all, chain_axis=1, sample_axis=0)
Rhat = blackjax.diagnostics.potential_scale_reduction(cold_all, chain_axis=1, sample_axis=0)
print(f"  ESS  mean={float(jnp.mean(ESS)):.1f}  min={float(jnp.min(ESS)):.1f}")
print(f"  Rhat mean={float(jnp.mean(Rhat)):.4f}  max={float(jnp.max(Rhat)):.4f}")