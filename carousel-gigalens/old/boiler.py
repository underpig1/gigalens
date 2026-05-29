"""
Carousel Lens (DESJ0603-3558) — gigalens configuration
HST Program GO-16773, WFC3/IR F140W
Reference: Urcelay et al. 2026 (arXiv:2602.16077)
"""
import numpy as np
from astropy.io import fits

# Imaging parameters (derived from FITS header; do not change without re-running setup_data.ipynb)
NUM_PIX     = 800
DELTA_PIX   = 0.06983        # arcsec/pixel (from WCS PC matrix)
EXP_TIME    = 597.694383     # seconds
BKG_RMS     = 0.006958211939640636  # counts/s
EXTENT      = (
    -NUM_PIX/2 * DELTA_PIX,
     NUM_PIX/2 * DELTA_PIX,
    -NUM_PIX/2 * DELTA_PIX,
     NUM_PIX/2 * DELTA_PIX,
)

# Source redshifts from Urcelay et al. (spectroscopic MUSE)
Z_SOURCES = {
    'src1':  0.962,
    'src3':  1.166,
    'src45': 1.432,   # brightest arcs — default single-source choice
    'src8':  3.549,
    'src9':  1.506,
    'src11': 4.090,
    'src1213': 3.086,
}

# Known lens/cluster redshift (ACT-CL J0603.9-3557)
Z_LENS = 0.49


def settings():
    return NUM_PIX, DELTA_PIX, EXP_TIME, BKG_RMS, EXTENT


def observed_image(path='../sean-carousel/model_data/f140w_img.fits'):
    """Load background-subtracted HST F140W cutout."""
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float64)


def psf(path='../carousel-lens/psf.npy'):
    """Load and normalize the PSF kernel."""
    kernel = np.load(path).astype(np.float64)
    return kernel / kernel.sum()
