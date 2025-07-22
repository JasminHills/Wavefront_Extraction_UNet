import os
import sys
from copy import deepcopy

from lasy.laser import Laser
from lasy import profiles
from lasy.utils.zernike import zernike
from lasy.optical_elements import ParabolicMirror, ZernikeAberrations

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy 

import PIL
from PIL import Image

import time
import aotools
from scipy import fftpack
from astropy.io import fits
global wavelength
wavelength =800*1e-9

import warnings

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    
def showxy(laser, **kw):
    """
    Show a 2D image of the laser amplitude.

    Parameters
    ----------
    **kw: additional arguments to be passed to matplotlib's imshow command
    """
    temporal_field = laser.grid.get_temporal_field()
    i_slice = int(temporal_field.shape[-1] // 2)
    E = temporal_field[:,:,  i_slice]
    extent = [
            laser.grid.lo[1],
            laser.grid.hi[1],
            laser.grid.lo[0],
            laser.grid.hi[0],
        ]

    plt.imshow(abs(E), extent=extent, aspect="auto", origin="lower", **kw)
    cb = plt.colorbar()
    cb.set_label("$|E_{envelope}|$ (V/m) ", fontsize=12)
    plt.xlabel("y (m)", fontsize=12)
    plt.ylabel("x (m)", fontsize=12)
    plt.show()

def ZernikeGen(n_psfs, n_zernike):
    np.random.seed(seed=1)
    i_zernike = np.arange(2, n_zernike + 2)      # Zernike polynomial indices (piston excluded)
    o_zernike= []             # Zernike polynomial radial Order, see J. Noll paper :
    
    for i in range(1,n_zernike):      # "Zernike polynomials and atmospheric turbulence", 1975
        for j in range(i+1):
            if len(o_zernike) < n_zernike:
                o_zernike.append(i)

    # Generate randomly Zernike coefficient. By dividing the value

    # by its radial order we produce a distribution following
    # the expected 1/f^-2 law.
    scales=[]
    c_zernikeFinal=np.zeros((n_psfs, n_zernike))
    for j in range(n_psfs):
        c_zernike = np.random.random(n_zernike)-0.5
        terms=np.random.randint(80, n_zernike)
        
        for i in range(terms):#n_zernike):
            c_zernikeFinal[j, i] = c_zernike[i] / o_zernike[i]
            
    c_zernikeFinal= np.array([c_zernikeFinal[k, :] / np.abs(c_zernikeFinal[k, :]).sum()* wavelength*(10**9)for k in range(1*n_psfs)])
    return c_zernikeFinal
    
def LaserSetUp(f0):
    wavelength = 800e-9  # Laser wavelength in meters
    polarization = (1, 0)  # Linearly polarized in the x direction
    energy = 0.4  # Energy of the laser pulse in joules
    spot_size = 40e-6  # Waist of the laser pulse in meters
    WndowSz=400e-6
    tau = 42e-15  # Pulse duration of the laser in seconds
    t_peak = 0.0  # Location of the peak of the laser pulse in time
    
    ls_profile=profiles.GaussianProfile(w0=spot_size, wavelength=wavelength, tau=tau, t_peak=t_peak, laser_energy=0.4, pol=polarization )
    lo = (- WndowSz,-WndowSz,  -2 * tau)  # Lower bounds of the simulation box
    hi = (WndowSz,WndowSz, 2 * tau)  # Upper bounds of the simulation box
    num_points = (700, 700, 10)  # Number of points in each dimension
    pupil_coords=(0, 0, 400*1e-6)
    laser = Laser('xyt', lo, hi, num_points, ls_profile)
    laser.propagate(-f0*1e-3)
    return laser

def LaserProp(i, c, laser, f0, outFile):
    ls=deepcopy(laser) #Copy
    if f0==9:
        pupil_coords=(0, 0, 200*1e-6)
    else:
        pupil_coords=(0, 0, 400*1e-6)      
    WFE=np.random.randint(10, 30)
    cc=c*WFE*1e-3 
    zernike_amplitudes = {index: value for index, value in enumerate(cc)}
    phs=aotools.functions.zernike.phaseFromZernikes(c*WFE*1e-3, 128, norm='rms')
    aberrations_in=phs
    # aberrations_in = np.squeeze(np.sum(c_z[i, :, None, None]* WFE*1e-3*zernike_basis[1:, :, :], axis=0))
    znk=ZernikeAberrations(pupil_coords, zernike_amplitudes)
    ls.apply_optics(znk)
    ls.propagate(f0*1e-3)
    
    temporal_field = ls.grid.get_temporal_field()
    i_slice = int(temporal_field.shape[-1] // 2)
    E = temporal_field[:,:,  i_slice]
    norm=abs(E)**2
    norm=scipy.ndimage.zoom(norm, 128/np.shape(norm)[0], order=0)
    outfile = outFile+"/psf_" + str(i) + ".fits"
    hdu_primary = fits.PrimaryHDU(cc.astype(np.float32))
    hdu_phase = fits.ImageHDU(aberrations_in.astype(np.float32), name='PHASE')
    hdu_In = fits.ImageHDU(norm.astype(np.float32), name='INFOCUS')
    hdu_Out = fits.ImageHDU(norm.astype(np.float32), name='OUTFOCUS')
    hdu = fits.HDUList([hdu_primary, hdu_phase, hdu_In, hdu_Out])
    hdu.writeto(outfile, overwrite=True)
    

if __name__ == "__main__":
    # total arguments
    print(sys.argv)
    nm, filnm, n_psfs, n_zernike, f0 = sys.argv
    os.system('rmdir '+filnm)
    os.system('mkdir -p '+filnm)
    c_z=ZernikeGen(int(n_psfs), int(n_zernike))
    laser=LaserSetUp(int(f0))
    for i, c in enumerate(c_z):
        LaserProp(i, c, laser, int(f0), filnm)
        
    
    

