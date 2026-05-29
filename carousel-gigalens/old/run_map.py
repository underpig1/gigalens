"""MAP optimization for the full Carousel Lens model — run via SLURM."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'        # single GPU — avoids shard_map/pvary issues
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
sys.path.insert(0, '../sean-carousel/src')

import numpy as np
import optax
import jax
import jax.numpy as jnp
from astropy.io import fits
import tensorflow_probability.substrates.jax as tfp

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import BackwardProbModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles import mass, light
from gigalens.jax.profiles.light import sersic_shapelets as ss_mod
# Note: Ld uses EPL (per Sheu+2024 Table 2), so mass.piemd is not needed
from gigalens.jax.prior import Prior, make_prior_and_model

tfd = tfp.distributions
print('JAX devices:', jax.devices(), flush=True)

# --- Data ---
with fits.open('../sean-carousel/model_data/f140w_img.fits') as hdul:
    obs_native = hdul[0].data.astype(np.float32)
    hdr        = hdul[0].header
    exp_time   = float(hdr['EXPTIME'])
    bkg_rms    = float(hdr['BKGRMS'])

BIN = 2
h, w = obs_native.shape
obs = obs_native[:h//BIN*BIN, :w//BIN*BIN].reshape(h//BIN, BIN, w//BIN, BIN).mean(axis=(1, 3))
pc        = np.array([[hdr['PC1_1'], hdr['PC1_2']], [hdr['PC2_1'], hdr['PC2_2']]])
delta_pix = np.sqrt(np.sum(pc[0]**2)) * 3600 * BIN
num_pix   = obs.shape[0]
bkg_rms   = bkg_rms / BIN

kernel = np.load('../carousel-lens/psf.npy').astype(np.float64)
kernel /= kernel.sum()
print(f'Image: {obs.shape}  delta_pix={delta_pix:.5f}"/px', flush=True)

# --- Priors ---
dr_0962 = 0.4219
dr_1166 = 0.4969
dr_1432 = 0.5619
dr_4520 = 0.7527

# Priors from Sheu+2024 Table 2: La EPL θ_E=13.03" γ=1.67 q=0.87 PA=-45°
#                                  Ld EPL θ_E=0.99"  γ=2.12 q=0.69 PA=-38°
#                                  shear γ_ext=0.11 φ=9°
# e1=c·cos(2φ), e2=c·sin(2φ), c=(1−q)/(1+q)
la_prior = Prior(mass.epl.EPL(), dict(
    center_x=tfd.Normal(6.70, 0.5), center_y=tfd.Normal(4.80, 0.5),
    e1=tfd.TruncatedNormal( 0.000, 0.05, -0.3, 0.3),
    e2=tfd.TruncatedNormal(-0.069, 0.05, -0.3, 0.3),
    theta_E=tfd.Normal(13.03, 0.5), gamma=tfd.TruncatedNormal(1.67, 0.08, 1.5, 2.5),
))
ld_prior = Prior(mass.epl.EPL(), dict(
    center_x=tfd.Normal(11.81, 1.0), center_y=tfd.Normal(23.03, 1.0),
    e1=tfd.TruncatedNormal( 0.044, 0.05, -0.3, 0.3),
    e2=tfd.TruncatedNormal(-0.178, 0.05, -0.3, 0.3),
    theta_E=tfd.Normal(0.99, 0.15), gamma=tfd.TruncatedNormal(2.12, 0.10, 1.5, 2.5),
))
shear_prior = Prior(mass.shear.Shear(), dict(
    gamma1=tfd.Normal(0.105, 0.05),
    gamma2=tfd.Normal(0.034, 0.05),
))
bcg_prior = Prior(light.sersic.SersicEllipse(use_lstsq=True), dict(
    center_x=tfd.Normal(6.70, 0.3), center_y=tfd.Normal(4.80, 0.3),
    e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3), e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
    n_sersic=tfd.Uniform(2., 8.), R_sersic=tfd.LogNormal(jnp.log(3.0), 0.3),
))

def _src(dr, cx, cy, n_max=6, sig=1.5):
    return Prior(ss_mod.SersicShapelets(n_max=n_max, use_lstsq=True, cosmo_sample=False), dict(
        deflection_ratio=dr,
        center_x=tfd.Normal(cx, sig), center_y=tfd.Normal(cy, sig),
        e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3), e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic=tfd.Uniform(0.5, 6.), R_sersic=tfd.LogNormal(jnp.log(0.4), 0.3),
        beta=tfd.LogNormal(jnp.log(0.4), 0.3),
    ))

# n_max=6 (depth=28) for bright/complex arcs; n_max=4 (depth=15) for fainter ones.
# Basis total: 28+28+28+15+15+1(BCG) = 115. Fits on exclusive A100.
prior, phys_model = make_prior_and_model(
    lenses=[la_prior, ld_prior, shear_prior],
    sources=[
        _src(dr_0962, 7.67, 3.32, n_max=6),   # S1  z=0.962
        _src(dr_1166, 6.80, 7.92, n_max=6),   # S3  z=1.166, naked cusp
        _src(dr_1432, 4.63, 3.79, n_max=6),   # S4  z=1.432, Einstein cross
        _src(dr_1432, 4.78, 0.84, n_max=4),   # S5  z=1.432, fold quad
        _src(dr_4520, 6.70, 4.80, n_max=4),   # S7  z=4.52
    ],
    foreground=[bcg_prior],
)
print('Prior ready.', flush=True)

# --- Simulator ---
sim_config = SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix, supersample=1, kernel=kernel)

cx_px = num_pix / 2 + 6.70 / delta_pix
cy_px = num_pix / 2 + 4.80 / delta_pix
yy, xx = np.mgrid[:num_pix, :num_pix]
r_arcsec  = np.sqrt((xx - cx_px)**2 + (yy - cy_px)**2) * delta_pix
circ_mask = r_arcsec < 20.0

prob_model = BackwardProbModel(prior, obs, background_rms=bkg_rms,
                               exp_time=exp_time, mask=circ_mask)
model_seq  = ModellingSequence(phys_model, prob_model, sim_config)
print(f'Mask: {int(circ_mask.sum())} pixels  ({100*circ_mask.mean():.1f}%)', flush=True)

# --- MAP ---
NUM_STEPS = 8000
N_SAMPLES = 50

schedule  = optax.cosine_decay_schedule(init_value=3e-4, decay_steps=NUM_STEPS, alpha=1e-2)
optimizer = optax.adabelief(schedule, b1=0.95, b2=0.99)
start     = prior.sample(N_SAMPLES, jax.random.PRNGKey(0))

print(f'Running MAP: {N_SAMPLES} chains x {NUM_STEPS} steps ...', flush=True)
best, lps, chisq = model_seq.MAP(
    optimizer=optimizer, n_samples=N_SAMPLES, num_steps=NUM_STEPS, start=start, seed=0,
)
print(f'Done. Best reduced chi2 = {float(chisq[-1]):.4f}', flush=True)

os.makedirs('models', exist_ok=True)
np.save('models/MAP_best.npy',  np.array(best))
np.save('models/MAP_chisq.npy', np.array(chisq))
np.save('models/MAP_lps.npy',   np.array(lps))
print('Saved to models/', flush=True)
