from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import healpy as hp
from optparse import OptionParser
import os
from cr import cr_1
import tidal as tid

def opt_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
    
parser = OptionParser()
# Options
parser.add_option('--coords', dest='coords', default='EQ', type=str,
                  help='Coordinates (GAL or EQ)')
parser.add_option('--catalog', dest='fname_cat', default='data/2MASS_XSC.fits', type=str,
                  help='Path to catalog')
parser.add_option('--k-threshold', dest='k_threshold', default=13.9, type=float,
                  help='K-band threshold')
parser.add_option('--rotate', dest='rotate', default=False, action='store_true',
                  help='Rotate to leave galaxy at the pole?')
parser.add_option('--plot-stuff', dest='plot_stuff', default=False, action='store_true',
                  help='Plot stuff?')
parser.add_option('--nside',dest='nside', default=64, type=int,
                  help='HEALPix resolution parameter')
parser.add_option('--output-prefix',dest='outdir', default='output', type=str,
                  help='Output directory')
parser.add_option('--smooth-scale',dest='theta_sm', default=1., type=float,
                  help='Smoothing scale')
(o, args) = parser.parse_args()

os.system('mkdir -p '+o.outdir)
npix=hp.nside2npix(o.nside)

#Read catalog
cat=fits.open(o.fname_cat)[1].data
#Remove faint sources
if 'KCORR' in cat.names :
    kmask=cat['KCORR']<o.k_threshold
elif 'kcorr_schlegel' in cat.names :
    kmask=cat['kcorr_schlegel']<o.k_threshold
else :
    raise KeyError("Can't find K-band magnitude")
cat=cat[kmask]

if 'SUPDEC' in cat.names :
    dec=cat['SUPDEC']
elif 'sup_dec' in cat.names :
    dec=cat['sup_dec']
else :
    raise KeyError("Can't find declination")
if 'SUPRA' in cat.names :
    ra=cat['SUPRA']
elif 'sup_ra' in cat.names :
    ra=cat['sup_ra']
else :
    raise KeyError("Can't find right ascension")
theta,phi=tid.get_theta_phi(dec,ra,coords=o.coords,rotate=o.rotate)
mask=tid.get_mask(theta,phi,o.nside,coords=o.coords,rotate=o.rotate)
dmap_sm,pot_sm,t11,t21,t22=tid.get_tidal_maps(theta,phi,mask,theta_sm=o.theta_sm,
                                              verbose=True,return_density_and_potential=True)
hp.write_map(o.outdir+'/tidal_sm%.1lf.fits'%o.theta_sm,[dmap_sm,pot_sm,t11,t21,t22],overwrite=True)

if o.plot_stuff :
    hp.mollview(pot_sm)
    hp.mollview(t11)
    hp.mollview(t21)
    hp.mollview(t22)
    
if o.plot_stuff :
    plt.show()
