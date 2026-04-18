import jax
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_slow_operation_alarm_threshold_ms=60000'
jax.distributed.initialize(local_device_ids=None)

if jax.process_index() == 0:
    # print(f"SLURM_PROCID: {os.environ.get('SLURM_PROCID')}")
    print(f"Visible JAX devices: {jax.devices()}")
    print(f"Local device count: {jax.local_device_count()}")

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import BackwardProbModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.model import PhysicalModel
from gigalens.jax.profiles.light import shapelets
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


import blackjax
import sys
import os
from pathlib import Path
sys.path.append(os.path.join(Path(__file__).resolve().parent.parent))
sys.path.insert(0, f'{os.environ['HOME']}/gigalens/src/gigalens_2/jax')
from profiles.light import sersic
jax.experimental.multihost_utils.sync_global_devices("run_start")

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
plt.imshow(observed_img, cmap="viridis", origin='lower')
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

import nuts_multi_3
from importlib import reload
reload(nuts_multi_3)
nuts_samples = nuts_multi_3.NUTS(model_seq, qz, n_chains=16, num_burnin_steps=500, num_results=1000, target_acceptance_rate=0.8)

smp = nuts_samples.reshape(-1, 41)
smp_physical = prob_model.bij.forward(list(smp.T))
tups = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
label_prefixes = ["", "", "other1_", "other2_", "lens1_", "lens2_", "src_sersic_", "src_shape_"]
labels = []
for (i, j), label_prefix in zip(tups, label_prefixes):
    labels.extend((label_prefix + key for key in smp_physical[i][j].keys()))
median_params = [
    [
        {key: np.median(value) for key, value in d.items()}
        for d in list_of_dicts
    ]
    for list_of_dicts in smp_physical
]
print(median_params)

plt.style.use('default')
plt_samples = np.vstack(
    [np.array(list(smp_physical[i][j].values())) for i, j in tups]
).T
print(plt_samples.shape)

fig = corner.corner(
    plt_samples,
    show_titles=True,
    title_fmt=".3f",
    labels=labels
)
plt.savefig("cornerplot.png")
plt.show()

print('final nuts rhat')
print(blackjax.diagnostics.potential_scale_reduction(nuts_samples, chain_axis=0, sample_axis=1))