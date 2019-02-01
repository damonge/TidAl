from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import healpy as hp
from optparse import OptionParser
import os
from cr import cr_1

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

#Produce ngal map
if os.path.isfile(o.outdir+'/ngals.fits') :
    nmap=hp.read_map(o.outdir+'/ngals.fits',verbose=False)
else :
    if o.coords=='EQ' :
        if 'SUPDEC' in cat.names :
            theta=np.radians(90-cat['SUPDEC'])
        elif 'sup_dec' in cat.names :
            theta=np.radians(90-cat['sup_dec'])
        else :
            raise KeyError("Can't find declination")
        if 'SUPRA' in cat.names :
            phi=np.radians(cat['SUPRA'])
        elif 'sup_ra' in cat.names :
            phi=np.radians(cat['sup_ra'])
        else :
            raise KeyError("Can't find right ascension")
    else :
        if 'B' in cat.names :
            theta=np.radians(90-cat['B'])
        elif 'b' in cat.names :
            theta=np.radians(90-cat['b'])
        else :
            raise KeyError("Can't find Galactic latitude")
        if 'L' in cat.names :
            phi=np.radians(cat['L'])
        elif 'l' in cat.names :
            phi=np.radians(cat['l'])
        else :
            raise KeyError("Can't find Galactic longitude")
        if o.rotate :
            coord=hp.ang2vec(theta,phi)
            rot=np.array([[0,1,0],[0,0,1],[1,0,0]])
            coord=(np.dot(rot,coord.T)).T
            theta,phi=hp.vec2ang(coord)
    pix=hp.ang2pix(o.nside,theta,phi)
    nmap=np.bincount(pix,minlength=npix)
    hp.write_map(o.outdir+'/ngals.fits',nmap+0.,overwrite=True)
if o.plot_stuff :
    hp.mollview(nmap);

#Create mask
if os.path.isfile(o.outdir+'/mask.fits') :
    mask=hp.read_map(o.outdir+'/mask.fits',verbose=False)
else :
    #Need to read off star density and dust in high resolution and rotate
    sd_hi=hp.ud_grade(hp.read_map("data/sd_map_full.fits",verbose=False),nside_out=4*o.nside)
    ak_hi=0.367*hp.ud_grade(hp.read_map("data/lambda_sfd_ebv.fits",verbose=False),nside_out=4*o.nside)
    if o.coords=='EQ' :
        r=hp.Rotator(coord=['G','C'])
        th_g,phi_g=hp.pix2ang(4*o.nside,np.arange(hp.nside2npix(4*o.nside)))
        th_e,phi_e=r(th_g,phi_g)
        ipix1=hp.ang2pix(4*o.nside,th_e,phi_e)
    else :
        if o.rotate :
            coord=np.array(hp.pix2vec(4*o.nside,np.arange(hp.nside2npix(4*o.nside))))
            rot=np.array([[0,1,0],[0,0,1],[1,0,0]])
            coord=(np.dot(rot,coord))
            ipix1=hp.vec2pix(4*o.nside,coord[0],coord[1],coord[2])
        else :
            ipix1=np.arange(hp.nside2npix(4*o.nside))
    npix_count=hp.ud_grade(np.bincount(ipix1,minlength=hp.nside2npix(4*o.nside))+0.,nside_out=o.nside,power=-2)
    sd_count=hp.ud_grade(np.bincount(ipix1,weights=sd_hi,minlength=hp.nside2npix(4*o.nside)),nside_out=o.nside,power=-2)
    ak_count=hp.ud_grade(np.bincount(ipix1,weights=ak_hi,minlength=hp.nside2npix(4*o.nside)),nside_out=o.nside,power=-2)
    not0=npix_count>0
    sd_lo=np.zeros(npix); sd_lo[not0]=sd_count[not0]/npix_count[not0]
    ak_lo=np.zeros(npix); ak_lo[not0]=ak_count[not0]/npix_count[not0]

    msk_sd=np.zeros(npix); msk_sd[sd_lo<3.5]=1.
    msk_ak=np.zeros(npix); msk_ak[ak_lo<0.06]=1.
    msk_ng=np.zeros(npix); msk_ng[nmap>0]=1.
    mask=msk_sd*msk_ak*msk_ng
    hp.write_map(o.outdir+'/mask.fits',mask+0.,overwrite=True)
if o.plot_stuff :
    hp.mollview(mask)

#Create overdensity map
if os.path.isfile(o.outdir+'/dmap.fits') :
    dmap=hp.read_map(o.outdir+'/dmap.fits',verbose=False)
else :
    ng_mean=np.sum(nmap*mask)/np.sum(mask)
    dmap=mask*(nmap/ng_mean-1)
    hp.write_map(o.outdir+'/dmap.fits',dmap,overwrite=True)
if o.plot_stuff :
    hp.mollview(mask*dmap)

#Create gaussianized overdensity map
if os.path.isfile(o.outdir+'/dlogmap.fits') :
    dlogmap=hp.read_map(o.outdir+'/dlogmap.fits',verbose=False)
else :
    dlogmap=np.zeros_like(dmap);
    dlogmap[mask>0]=np.log((1+dmap[mask>0])*np.sqrt(1+np.std(dmap[mask>0])**2))
    hp.write_map(o.outdir+'/dlogmap.fits',dlogmap,overwrite=True)
if o.plot_stuff :
    hp.mollview(mask*dlogmap)

#Compute Gaussianized power spectrum and polynomial fit
if os.path.isfile(o.outdir+'/cls.npz') :
    d=np.load(o.outdir+'/cls.npz')
    ls=d['ls']; cls=d['cls']; cls_fit=d['cls_fit'];
else :
    cls=hp.anafast(dlogmap,pol=False)/np.mean(mask)
    ls=np.arange(3*o.nside)
    logls=np.log(ls[1:]); logcls=np.log(cls[1:])
    err=np.sqrt(2./(2*ls[1:]+1))/np.log(10.)
    poly=np.poly1d(np.polyfit(logls, logcls, 5,w=1/err))
    ls_fit=ls.copy(); ls_fit[0]=1; cls_fit=np.exp(poly(np.log(ls_fit)))
    np.savez(o.outdir+'/cls',ls=ls,cls=cls,cls_fit=cls_fit)
if o.plot_stuff :
    plt.figure()
    plt.plot(ls,cls,'k-')
    plt.plot(ls,cls_fit,'r-')
    plt.loglog()

#Get constrained realization
if os.path.isfile(o.outdir+'/dmap_wf.fits') and os.path.isfile(o.outdir+'/dlogmap_wf.fits') :
    dmap_cr=hp.read_map(o.outdir+'/dmap_wf.fits',verbose=False)
    dlogmap_cr=hp.read_map(o.outdir+'/dlogmap_wf.fits',verbose=False)
else :
    noise_level=0.05*np.sqrt(4*np.pi*np.mean(mask)/np.sum(nmap*mask))
    dlogmap_cr=cr_1(dlogmap*mask,mask,noise_level,cls_fit,m_true=dlogmap,smooth_scale_mask=0.05,
                    include_random=False,n_iter_max=150,plot_stuff=False,info_output_rate=-1,seed=-1)
    sigmasq=np.std(dmap[mask>0])**2
    dlogmap_cr=(1-mask)*dlogmap_cr+mask*dlogmap
    dmap_cr = np.exp(dlogmap_cr)/np.sqrt(1+sigmasq)-1
    hp.write_map(o.outdir+'/dmap_wf.fits',dmap_cr,overwrite=True)
    hp.write_map(o.outdir+'/dlogmap_wf.fits',dlogmap_cr,overwrite=True)
if o.plot_stuff :
    hp.mollview(dmap_cr-dmap);
    hp.mollview(dlogmap_cr-dmap);

if os.path.isfile(o.outdir+'/tidal_sm%.1lf.fits'%o.theta_sm) :
    dmap_sm,pot_sm,t11,t21,t22=hp.read_map(o.outdir+'/tidal_sm%.1lf.fits'%o.theta_sm,
                                           field=[0,1,2,3,4],verbose=False)
else :
    #Get tidal maps
    th,ph=hp.pix2ang(o.nside,np.arange(npix))
    cotth=np.cos(th)/np.sin(th)
    dmap_sm=hp.smoothing(dmap_cr,sigma=np.radians(o.theta_sm),verbose=False)
    
    #Projected potential
    ls=np.arange(3*o.nside)
    # avoid division by 0
    ls[0]=1; llplus = -1./(ls*(ls+1)); llplus[0] = 0
    # removing the infinity element
    pot_sm_lm=hp.almxfl(hp.map2alm(dmap_sm,pol=False),llplus)

    #tidal tensor components
    pot_sm,dpot_dth,dpot_dsph=hp.alm2map_der1(pot_sm_lm,o.nside)
    dpot_dth_lm=hp.map2alm(dpot_dth,pol=False)
    dpot_dsph_lm=hp.map2alm(dpot_dsph,pol=False)
    #T11 is dpot/dtheta^2
    _,t11,dpot_dsph_dtheta=hp.alm2map_der1(dpot_dth_lm,o.nside)
    dpot_dsph_dtheta_lm=hp.map2alm(dpot_dsph_dtheta,pol=False)
    #T21 is d/dth [dpot/dph/sin(th)]
    _,t21,dpot_dsph_dsph=hp.alm2map_der1(dpot_dsph_lm,o.nside)
    #T22 is 1/sth*d/dph [1/sth * dpot/dph]
    t22=cotth*dpot_dth+dpot_dsph_dsph

    hp.write_map(o.outdir+'/tidal_sm%.1lf.fits'%o.theta_sm,[dmap_sm,pot_sm,t11,t21,t22],overwrite=True)
t12=t21.copy()
if o.plot_stuff :
    hp.mollview(pot_sm)
    hp.mollview(t11)
    hp.mollview(t12)
    hp.mollview(t22)
    
if o.plot_stuff :
    plt.show()
