import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.io import fits
from pixell import enmap
from astroquery.skyview import SkyView
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


def load_files(tidal_filename, galaxy_cat_filename, mask_filename, verbose=True):
    """Loads relevant tidal files and galaxy data.

    ellipticity cut > 0.5, kcorr_schlegel < 13.9

    Args:
        tidal_filename: path to processed tide maps in healpix format
        galaxy_cat_filename: path to galaxy catalog
        mask_filename: path to math in healpix format
        
    Returns:
        tuple: consisting of 
            0. dict containing fields t11, t12, t22, den, pot
            3. masked and filtered galaxy catalog
            4. mask
            
        
    """

    den, pot, t11, t12, t22 = hp.read_map(tidal_filename, field=[0,1,2,3,4], verbose=verbose)
    tide_dict = {"den":den, "pot":pot, "t11":t11, "t12":t12, "t22":t22}
    npix = len(t11)
    nside = hp.get_nside(t11)

    # load in galaxy
    hdu = fits.open(galaxy_cat_filename)
    galaxy_data = hdu[1].data

    # masking and cuts
    galaxy_ind = hp.ang2pix( 
        nside=nside, 
        phi=galaxy_data['dec'], theta=galaxy_data['ra'], lonlat=True )
    mask = hp.read_map(mask_filename, verbose=verbose)
    ellip = (1-galaxy_data['j_ba'])
    cut = np.logical_and(galaxy_data['kcorr_schlegel'] < 13.9, mask[galaxy_ind] > 0.0)
    cut = np.logical_and(cut, ellip > 0.5)
    galaxy_data = galaxy_data[ cut ]
    
    return tide_dict, galaxy_data, mask


def compute_principal_eig( t11, t12, t22 ):
    """Computes principal eigenvalues/vectors from tidal tensor components.
    
    Args:
        t11 (ndarray): t_11 of tidal tensor
        t12 (ndarray): t_12 of tidal tensor
        t22 (ndarray): t_22 of tidal tensor
    
    Returns:
        tuple: eigenvalues, eigenvectors
    """

    npix = len(t11)
    # generate big stack of arrays
    array_stack = np.zeros( (npix, 2, 2) ) # long axis is first
    array_stack[:,0,0] = t11
    array_stack[:,1,0] = t12
    array_stack[:,0,1] = t12
    array_stack[:,1,1] = t22
    
    # solve eigenvalues
    w,v = np.linalg.eig(a=array_stack)
    princ_eig_ind = np.argmax( np.abs(w), axis=1 )
    princ_eigenvectors = np.array( [v[i,:,princ_eig_ind[i]] for i in range(npix)] )
    princ_eigenvalues = np.array( [w[i,princ_eig_ind[i]] for i in range(npix)] )
    
    return princ_eigenvalues, princ_eigenvectors

    
def get_cutout_from_hp(field, bbox,
                       kernel_width=1e-4, 
                       xp=50, yp=50, debug=False, 
                       rot=(0,0,0), hold=False):
    """Get a rectangular cutout of a healpix map
        
        kernel_width: gotta sample the healpix map with a gaussian kernel
            with this width
        bbox: is phi_min, phi_max, theta_min, theta_max
        rot: rotation for mollweide projection when debug=True
    """
    smoothed_cutout = np.zeros((xp,yp))
    ph_grid = np.zeros((xp,yp))
    th_grid = np.zeros((xp,yp))

    npix = len(field)
    nside = hp.get_nside(field)
    th,ph = hp.pix2ang(nside,np.arange(npix))
    
    width = bbox[1] - bbox[0]
    height = bbox[3] - bbox[2]
    center_theta, center_phi = bbox[2] + height/2, bbox[0] + width/2
    for i in range(xp):
        for j in range(yp):
            ph0 = (center_phi - width/2 + i * width/xp) 
            th0 = (center_theta - height/2 + j * height/xp) 
            ph_grid[i,j] = ph0
            th_grid[i,j] = th0
            smoothed_cutout[i,j] = np.sum( 
                np.exp( -( (th - th0)**2 + (ph - ph0)**2) / (2 * kernel_width) ) * field )

    norm = np.sum( np.exp( -( (th - center_theta)**2 + (ph - center_phi)**2) / (2 * kernel_width) ) * 1 )
    smoothed_cutout /= norm
    
    if debug:
        den_copy_highlighted = field.copy()
        region = np.logical_and( 
            np.logical_and( th < center_theta + height/2,  th > center_theta - height/2 ),
            np.logical_and( ph < center_phi + width/2, ph > center_phi - width/2 )
            )
        den_copy_highlighted[~region] *= 0.1
        hp.mollview( den_copy_highlighted, rot=rot, hold=hold)
    
    return smoothed_cutout, ph_grid, th_grid


def plot_vectorfield(ax, bbox, nside, healpix_vectorfield, arrowscaling=0.01):
    """Plot a vectorfield onto an axis within a bbox
    """
    width = bbox[1] - bbox[0]
    height = bbox[3] - bbox[2]
    npix = healpix_vectorfield.shape[0]
    center_theta, center_phi = bbox[2] + height/2, bbox[0] + width/2
    th, ph = hp.pix2ang(nside,np.arange(npix))
    # now add in the vector field of the principal vectors
    for i in range(npix):
        th0, ph0 = th[i], ph[i]
        if ((ph0 > center_phi - width/2) & (ph0 < center_phi + width/2) & \
            (th0 > center_theta - height/2) & (th0 < center_theta + height/2)):
            # plot vector
            vec = healpix_vectorfield[i,:] 
            dx, dy = vec[0] * arrowscaling, vec[1] * arrowscaling
            ax.arrow( ph0 - dx/2, th0 - dy/2, dx, dy, lw=0.5, head_width=0, color="k" )
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    
    
def plot_row( three_indices, data, phi_x, phi_y ):
    """Plot three galaxies from a galaxy catalog with ellipticity
    """
    
    fig = plt.figure(figsize=(15,5))

    for fig_i, i in enumerate(three_indices):

        ra0, dec0 = data['ra'][i], data['dec'][i]
        c = SkyCoord(ra=ra0*u.degree, dec=dec0*u.degree, frame='icrs')
        # Retrieve M83 2MASS K-band image:
        images = SkyView.get_images(position=c, survey=['2MASS-J'],
                                        pixels=50, scaling="Log")

        hdu_im = images[0][0]
        im = hdu_im.data.copy()
        im[im < 0] = 0
        wcs = WCS(hdu_im.header)
        ax = fig.add_subplot(1, 3, fig_i+1, projection=wcs)  # create an axes object in the figure
        ax.imshow( im , origin='lower', vmax=np.mean(im) + 1*np.std(im))
        amp = 5 * (1-data['j_ba'][i])  
        ax.arrow(49 / 2, 49 / 2, dx=amp*phi_x[i], dy=amp*phi_y[i], color="black", lw=3, head_width=1 )

    plt.tight_layout()
    