import jax
jax.distributed.initialize()

import sys
sys.path.insert(0, f'../gigalens/src')

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import ForwardProbModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.model import PhysicalModel
from gigalens.jax.profiles.light import sersic
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

# ---------------------------
# TIMING HELPERS (ADDED)
# ---------------------------
TIMES = {}
START_TOTAL = time.time()

def tstart(key):
    TIMES[key] = -time.time()

def tend(key):
    TIMES[key] += time.time()
# ---------------------------

tfd = tfp.distributions

# Showing all available devices
total_devices = jax.device_count()
verbose = jax.process_index() == 0
print(f"{jax.process_index()}: local devices: {jax.local_devices()}")
if verbose:
    print(f"Global devices: {jax.devices()}")

# Create priors for lens mass, lens light, source light
lens_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
                gamma=tfd.TruncatedNormal(2, 0.5, 1, 3),
                e1=tfd.Normal(0, 0.2),
                e2=tfd.Normal(0, 0.2),
                center_x=tfd.Normal(0, 0.06),
                center_y=tfd.Normal(0, 0.06),
            )
        ),
        tfd.JointDistributionNamed(
            dict(gamma1=tfd.Normal(0, 0.1), gamma2=tfd.Normal(0, 0.1))
        ),
    ]
)
lens_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
                n_sersic=tfd.Uniform(0.5, 8),
                e1=tfd.TruncatedNormal(0, 0.1, -0.15, 0.15),
                e2=tfd.TruncatedNormal(0, 0.1, -0.15, 0.15),
                center_x=tfd.Normal(0, 0.02),
                center_y=tfd.Normal(0, 0.02),
                Ie=tfd.LogNormal(jnp.log(300.0), 0.5),
            )
        )
    ]
)
source_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
                n_sersic=tfd.Uniform(0.5, 8),
                e1=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
                e2=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
                center_x=tfd.Normal(0, 0.5),
                center_y=tfd.Normal(0, 0.5),
                Ie=tfd.LogNormal(jnp.log(150.0), 0.9),
            )
        )
    ]
)
prior = tfd.JointDistributionSequential(
    [lens_prior, lens_light_prior, source_light_prior]
)

# TODO: Change to point to local directory
kernel = jnp.array(np.load("../desi-238/psf.npy"), dtype=jnp.float32)
observed_img = jnp.array(np.load("../desi-238/imL.npy"), dtype=jnp.float32)

# Modeling Parameters
background_rms = 0.2
exp_time = 100
delta_pix = 0.065
num_pix = 100
supersample = 1 #2

phys_model = PhysicalModel(
    [epl.EPL(50), shear.Shear()],
    [sersic.SersicEllipse(use_lstsq=False)],
    [sersic.SersicEllipse(use_lstsq=False)],
)
prob_model = ForwardProbModel(
    prior, observed_img, background_rms=background_rms, exp_time=exp_time
)
sim_config = SimulatorConfig(
    delta_pix=delta_pix, num_pix=num_pix, supersample=supersample, kernel=kernel
)
lens_sim = LensSimulator(phys_model, sim_config, bs=1)

model_seq = ModellingSequence(phys_model, prob_model, sim_config)

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
phys_model = PhysicalModel([epl.EPL(50), shear.Shear()],
                           [sersic.SersicEllipse(use_lstsq=False),
                            sersic.SersicEllipse(use_lstsq=False),
                            sersic.SersicEllipse(use_lstsq=False)],
                           [sersic.SersicEllipse(use_lstsq=False),
                            sersic.SersicEllipse(use_lstsq=False)])
lens_sim = LensSimulator(phys_model, sim_config, bs=1)
prob_model = ForwardProbModel(prior, obs, background_rms=0.007616, exp_time=exp_time)
model_seq = ModellingSequence(phys_model, prob_model, sim_config)

schedule_fn = optax.polynomial_schedule(init_value=-0.02, end_value=-0.003, power=0.1, transition_steps=1000)
opt = optax.chain(
  optax.scale_by_adam(),
  optax.scale_by_schedule(schedule_fn),
)

# ---------------------------
# MAP TIMING
# ---------------------------
tstart("map")
map_estimate = model_seq.MAP_multi(opt, seed=0, num_steps=1000)
tend("map")
# ---------------------------

lps = prob_model.log_prob(LensSimulator(phys_model, sim_config, bs=500), map_estimate[0])[0]
best = map_estimate[jnp.argmax(lps)][jnp.newaxis,:]

physical = prob_model.bij.forward(list(best.T))
simulated = lens_sim.simulate(physical)
fig = plt.figure()
plt.imshow(simulated, norm=norm, cmap="viridis", origin='lower')
plt.title("Predicted image")
plt.plot()
plt.savefig("results/map_results.png")
plt.close(fig)

# SVI
print("Starting SVI")

schedule_fn = optax.polynomial_schedule(init_value=-1e-7, end_value=-2e-3, power=2, transition_steps=300)
opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)

# ---------------------------
# SVI TIMING
# ---------------------------
tstart("svi")
qz, loss_hist = model_seq.SVI_multi(best, opt, n_vi=1000, num_steps=2000)
tend("svi")
# ---------------------------

fig = plt.figure()
plt.plot(loss_hist)
plt.title('Loss')
plt.savefig("results/loss.png")
plt.close(fig)

# HMC
print("Starting HMC")

# ---------------------------
# HMC TIMING
# ---------------------------
tstart("hmc")
samples = model_seq.HMC_multi(qz, num_burnin_steps=500, n_hmc=75, num_results=750)
tend("hmc")
# ---------------------------

rhat= tfp.mcmc.potential_scale_reduction(jnp.transpose(samples, (1,2,0,3)), independent_chain_ndims=2)
fig = plt.figure()
plt.plot(rhat)
plt.title('rhat')
plt.savefig("results/rhat.png")
plt.close(fig)

from corner import corner

smp = jnp.transpose(samples, (1, 2, 0, 3)).reshape((-1, len(rhat)))
smp_physical = prob_model.bij.forward(list(smp.T))

def get_corner_samples(physical, key):
    """
    Key tells you lens_mass (1), lens_light (2), or source_light (3)
    """
    return_list = []

    if key == 1:
        for i in physical[0][0].keys():
            return_list.append(physical[0][0][i])
        for j in physical[0][1].keys():
            return_list.append(physical[0][1][j])

    if key == 2:
        for i in physical[1][0].keys():
            return_list.append(physical[1][0][i])

    if key == 3:
        for i in physical[2][0].keys():
            return_list.append(physical[2][0][i])
            
    return np.stack(return_list).T

mass_samps = get_corner_samples(smp_physical, 1)

plt.style.use('default')

labels=[r'$\theta_E$', 
        r'$\gamma$', 
        r'$\epsilon_1$', 
        r'$\epsilon_2$', 
        r'$x$', r'$y$', 
        r'$\gamma_{1,ext}$', 
        r'$\gamma_{2,ext}$']

fig = corner(mass_samps, show_titles=True, title_fmt='.3f', labels=labels)
_ = fig.suptitle("Lens Parameters")

plt.title('rhat')
plt.savefig("results/rhat.png")
plt.close(fig)

# ---------------------------
# WRITE TIMINGS TO FILE (RANK 0 ONLY)
# ---------------------------
TOTAL_TIME = time.time() - START_TOTAL

if jax.process_index() == 0:
    with open("results/timings.log", "a") as f:
        f.write(
            f"{jax.device_count()} GPUs | "
            f"MAP: {TIMES['map']:.2f}s | "
            f"SVI: {TIMES['svi']:.2f}s | "
            f"HMC: {TIMES['hmc']:.2f}s | "
            f"TOTAL: {TOTAL_TIME:.2f}s\n"
        )
# ---------------------------