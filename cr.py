import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg

glb_niter=0
def cr_1(m_data,                 #Map with the data
         m_mask,                 #Map with the mask
         noise_level,            #Noise level (std per sterad)
         cl_signal,              #Power spectrum of the signal
         m_true=None,            #Map with the true signal (only needed for comparison plots)
         smooth_scale_mask=0.05, #Apodization scale for the mask (in rad)
         include_random=True,       #Add random fluctuations to the Wiener-filtered map
         n_iter_max=500,         #Maximum number of CG iterations
         plot_stuff=False,           #Set to 1 to plot informative maps
         info_output_rate=100,   #Output CG information every 100 iterations
         seed=-1) :              #Random seed
    global glb_niter
    glb_niter=0
    npix=len(m_data)
    nside=hp.npix2nside(npix)
    pix_area=4*np.pi/npix

    #Initialize RNG
    if seed>=0 :
        np.random.seed(1234)

    #Apodize mask
    if smooth_scale_mask>0 :
        mask_smoothed=m_mask*hp.smoothing(m_mask,fwhm=smooth_scale_mask,iter=5,verbose=False)
    else :
        mask_smoothed=m_mask
    if plot_stuff :
        hp.mollview(mask_smoothed); plt.show()

    #Compute shorthand quantities
    fsky=np.sum(mask_smoothed)/len(mask_smoothed)
    noise_per_pixel=noise_level/np.sqrt(pix_area)
    cl_noise=noise_level**2*np.ones(3*nside)
    map_inoise=mask_smoothed/noise_per_pixel**2
    ipix_area=1./pix_area
    icl_signal=pix_area**2/cl_signal

    #Diagonal product in harmonic space
    def prod_fourier(m,cl) :
        return hp.alm2map(hp.almxfl(hp.map2alm(m,iter=3),cl),nside,verbose=False)*ipix_area
    
    #Product with the noise matrix
    def prod_inoise(m,power) :
        return m*map_inoise**power

    #Matrix to invert (S^-1+N^-1)
    def linop_A(m) :
        return prod_fourier(m,icl_signal)+prod_inoise(m,1.0)

    #Preconditioner (S^-1+mean(N^-1))^-1
    def linop_M(m) :
#        return prod_fourier(m,cl_signal)
        return prod_fourier(m,cl_signal*cl_noise/(cl_signal+fsky*cl_noise))

    #Output information about CG iteration
    def callback(m) :
        global glb_niter
        glb_niter=glb_niter+1
        if (info_output_rate>0) and (glb_niter%info_output_rate==0) :
            print("%d iters"%glb_niter+" res=%lE"%(np.std(m_mask*(m-m_data))))
            if plot_stuff :
                if m_true!=None :
                    lr=np.arange(3*nside)
                    plt.plot(lr,hp.anafast(m_true))
                    plt.plot(lr,hp.anafast(m))
                    plt.plot(lr,cl_signal)
                    plt.plot(lr,cl_noise)
                    plt.gca().set_yscale('log')
                    plt.gca().set_xscale('log')
                    plt.show()
                hp.mollview(m_data,title="Data",max=1.5)#,min=np.amin(m_true),max=np.amax(m_true));
                if m_true!=None :
                    hp.mollview(m_true,title="True",max=1.5)#,min=np.amin(m_true),max=np.amax(m_true));
                hp.mollview(m,title="%d iters"%glb_niter,max=1.5)#,min=np.amin(m_true),max=np.amax(m_true));
                plt.show()

    #Choose the goodnes of the choice of preconditioner
    if plot_stuff :
        hp.mollview(linop_M(linop_A(m_data))-m_data,title="Preconditioner choice")
        plt.show()

    #Set up CG system
    A=LinearOperator((npix,npix),matvec=linop_A)
    M=LinearOperator((npix,npix),matvec=linop_M)
    b=prod_inoise(m_data,1.)
    if include_random :
        rs=np.random.randn(npix)
        rn=np.random.randn(npix)
        b+=prod_fourier(rs,np.sqrt(pix_area*icl_signal))+prod_inoise(rn,0.5)

    print("Solving")
    map_solve,info=cg(A,b,x0=m_data,maxiter=n_iter_max,M=M,callback=callback)
    if plot_stuff :
        hp.mollview(map_solve,title="Solved %d"%info); plt.show()

    return map_solve
