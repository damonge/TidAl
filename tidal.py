from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import healpy as hp
from cr import cr_1

def get_theta_phi(dec,ra,coords='EQ',rotate=False):
    """
    Returns the spherical coordinates for a set of declination and right ascensions

    :param dec: array of declinations
    :param ra: array of right ascensions
    :param coords: output coordinates. Pass 'EQ' for equatorial, 'GAL' for galactic.
    :param rotate: if `coords='GAL'` and you want to rotate the galaxy into the pole, pass `rotate=True`.
    :return: 2 arrays with spherical coordinates, theta, phi
    """
    if coords=='EQ':
        theta=np.radians(90-dec)
        phi=np.radians(ra)
    else:
        rot=hp.Rotator(coord=['C','G'])
        theta,phi=rot(np.radians(90-dec),np.radians(ra))
        if rotate:
            coord=hp.ang2vec(theta,phi)
            rot=np.array([[0,1,0],[0,0,1],[1,0,0]])
            coord=(np.dot(rot,coord.T)).T
            theta,phi=hp.vec2ang(coord)
    return theta,phi

def get_mask(theta,phi,nside,coords='EQ',rotate=False):
    """
    Returns the mask used for the 2MASS analysis

    :param theta,phi: arrays of spherical coordinates for each object
    :param nside: HEALPix resolution parameter for the output mask
    :param coords: output coordinates. Pass 'EQ' for equatorial, 'GAL' for galactic.
    :param rotate: if `coords='GAL'` and you want to rotate the galaxy into the pole, pass `rotate=True`.
    :return: array containing the mask.
    """
    #Make Ngal map
    npix=hp.nside2npix(nside)
    pix=hp.ang2pix(nside,theta,phi)
    nmap=np.bincount(pix,minlength=npix)

    #Need to read off star density and dust in high resolution and rotate
    sd_hi=hp.ud_grade(hp.read_map("data/sd_map_full.fits",verbose=False),nside_out=4*nside)
    ak_hi=0.367*hp.ud_grade(hp.read_map("data/lambda_sfd_ebv.fits",verbose=False),nside_out=4*nside)
    if coords=='EQ':
        r=hp.Rotator(coord=['G','C'])
        th_g,phi_g=hp.pix2ang(4*nside,np.arange(hp.nside2npix(4*nside)))
        th_e,phi_e=r(th_g,phi_g)
        ipix1=hp.ang2pix(4*nside,th_e,phi_e)
    else:
        if rotate:
            coord=np.array(hp.pix2vec(4*nside,np.arange(hp.nside2npix(4*nside))))
            rot=np.array([[0,1,0],[0,0,1],[1,0,0]])
            coord=(np.dot(rot,coord))
            ipix1=hp.vec2pix(4*nside,coord[0],coord[1],coord[2])
        else:
            ipix1=np.arange(hp.nside2npix(4*nside))
    npix_count=hp.ud_grade(np.bincount(ipix1,minlength=hp.nside2npix(4*nside))+0.,
                           nside_out=nside,power=-2)
    sd_count=hp.ud_grade(np.bincount(ipix1,weights=sd_hi,minlength=hp.nside2npix(4*nside)),
                         nside_out=nside,power=-2)
    ak_count=hp.ud_grade(np.bincount(ipix1,weights=ak_hi,minlength=hp.nside2npix(4*nside)),
                         nside_out=nside,power=-2)
    not0=npix_count>0
    sd_lo=np.zeros(npix); sd_lo[not0]=sd_count[not0]/npix_count[not0]
    ak_lo=np.zeros(npix); ak_lo[not0]=ak_count[not0]/npix_count[not0]

    msk_sd=np.zeros(npix); msk_sd[sd_lo<3.5]=1.
    msk_ak=np.zeros(npix); msk_ak[ak_lo<0.06]=1.
    msk_ng=np.zeros(npix); msk_ng[nmap>0]=1.
    mask=msk_sd*msk_ak*msk_ng

    return mask

def get_tidal_maps(theta,phi,mask,theta_sm=0.5,verbose=False,return_density_and_potential=False):
    """
    Returns the mask used for the 2MASS analysis

    :param theta,phi: arrays of spherical coordinates for each object
    :param mask: mask to be used in the calculation
    :param theta_sm: smoothing scale (in degrees)
    :param verbose: set to True if you wanna follow the progress of the calculation
    :param return_density_and_potential: set to True if you want to calculate the smoothed density and potential fields (besides the tidal tensor components).
    :return: If `return_density_and_potential==True`, 5 arrays containing density, potential, t_theta_theta, t_theta_phi and t_phi_phi. Otherwise, just the last 3 arrays.
    """
    if verbose:
        print("Making delta map")
    #Make Ngal map
    npix=len(mask)
    nside=hp.npix2nside(npix)
    pix=hp.ang2pix(nside,theta,phi)
    nmap=np.bincount(pix,minlength=npix)

    #Make delta map
    ng_mean=np.sum(nmap*mask)/np.sum(mask)
    dmap=mask*(nmap/ng_mean-1)

    #Make Gaussianized overdensity map
    dlogmap=np.zeros_like(dmap)
    dlogmap[mask>0]=np.log((1+dmap[mask>0])*np.sqrt(1+np.std(dmap[mask>0])**2))

    #Compute Gaussianized power spectrum and polynomial fit
    if verbose:
        print("Computing power spectra")
    cls=hp.anafast(dlogmap,pol=False)/np.mean(mask)
    ls=np.arange(3*nside)
    logls=np.log(ls[1:]); logcls=np.log(cls[1:])
    err=np.sqrt(2./(2*ls[1:]+1))/np.log(10.)
    poly=np.poly1d(np.polyfit(logls,logcls,5,w=1/err))
    ls_fit=ls.copy(); ls_fit[0]=1; cls_fit=np.exp(poly(np.log(ls_fit)))

    #Get constrained realization
    if verbose:
        print("Generating constrained realization")
    noise_level=0.05*np.sqrt(4*np.pi*np.mean(mask)/np.sum(nmap*mask))
    dlogmap_cr=cr_1(dlogmap*mask,mask,noise_level,cls_fit,m_true=dlogmap,smooth_scale_mask=0.05,
                    include_random=False,n_iter_max=150,plot_stuff=False,info_output_rate=-1,seed=-1,
                    verbose=verbose)
    sigmasq=np.std(dmap[mask>0])**2
    dlogmap_cr=(1-mask)*dlogmap_cr+mask*dlogmap
    dmap_cr=np.exp(dlogmap_cr)/np.sqrt(1+sigmasq)-1

    #Get tidal field
    if verbose:
        print("Computing tidal field")
    dmap_lm=hp.map2alm(dmap_cr,pol=False)

    #Projected potential
    llplus=np.zeros(len(ls)); llplus[1:]=-1./((ls*(ls+1))[1:])
    pot_sm=hp.alm2map(hp.almxfl(dmap_lm,llplus),nside,pol=False,
                      sigma=np.radians(theta_sm),verbose=False)

    #Projected tidal field
    dmap_sm,q,u=hp.alm2map([dmap_lm,dmap_lm,0*dmap_lm],nside,pol=True,sigma=np.radians(theta_sm),verbose=False)
    t11=0.5*(dmap_sm+q)
    t21=0.5*u
    t22=0.5*(dmap_sm-q)

    if return_density_and_potential :
        return dmap,dmap_sm,pot_sm,t11,t21,t22
    else:        
        return t11,t21,t22

'''
Example:
ns=64; crd='EQ'; rt=False; thsm=0.5
cat=fits.open("data/2MASS_XSC.fits")[1].data
cat=cat[cat['kcorr_schlegel']<13.9]
th,ph=get_theta_phi(cat['sup_dec'],cat['sup_ra'],coords=crd,rotate=rt)
mask=get_mask(th,ph,ns,coords=crd,rotate=rt)
tid=get_tidal_maps(th,ph,mask,theta_sm=thsm)
'''
