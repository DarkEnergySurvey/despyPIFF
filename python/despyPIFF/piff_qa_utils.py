#!/usr/bin/env python3

"""
Analyze PIFF model to provide QA feedback
"""

#from __future__ import print_function
import time
import sys
import numpy as np
import fitsio
from scipy.optimize import curve_fit

import piff
import galsim
from ngmix import priors, joint_prior
import ngmix 

# flag values for psf catalog

MAX_CENTROID_SHIFT = 1.0
NOT_USED = 1
BAD_MEASUREMENT = 2
CENTROID_SHIFT = 4
OUTLIER = 8
FAILURE = 32
RESERVED = 64
NOT_STAR = 128
BLACK_FLAG_FACTOR = 512 # blacklist flags are this times the original exposure blacklist flag
                        # blacklist flags go up to 64,
pixel_scale = 0.263

###########################################
def do_ngmix_fit(im,wt,x,y,fwhm,icnt=0,rng=None,ftype='star',verbose=0):

    flag = 0
    dx, dy, g1, g2, flux = 0., 0., 0., 0., 0.
    T_guess = (fwhm/ 2.35482)**2 * 2.
    T = T_guess

#
#   Try to get a fit to the object using ngmix
#
    try:
        if (galsim.__version__ >= '1.5.1'):
            wcs=im.wcs.local(im.center)
            cen=im.true_center-im.origin
        else:
            wcs=im.wcs.local(im.center())
            cen=im.trueCenter()-im.origin()
#
#       Setting up priors
#
        cen_prior=ngmix.priors.CenPrior(0.0,0.0,pixel_scale,pixel_scale,rng=rng)
        gprior=ngmix.priors.GPriorBA(0.1,rng=rng)
        Tprior=ngmix.priors.LogNormal(T,0.2,rng=rng)
        Fprior=ngmix.priors.FlatPrior(-10.,1.e10,rng=rng)
        prior=ngmix.joint_prior.PriorSimpleSep(cen_prior,gprior,Tprior,Fprior)
#
#       Putting data in context for NGMIX
#
        jac=ngmix.Jacobian(wcs=wcs, x=cen.x + x - int(x+0.5), y=cen.y + y -int(y+0.5))
        obs=ngmix.Observation(image=im.array,weight=wt.array,jacobian=jac)

        psf_fitter = ngmix.fitting.Fitter(model='gauss', prior=prior)
        psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng, guess_from_moms=False)
        psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=3)    

        res = psf_runner.go(obs=obs)
        ngmix_flag=res['flags']
        if (ngmix_flag != 0):
            flag |= BAD_MEASUREMENT

        dy = res['pars'][0]
        dx = res['pars'][1]
        T = res['T']
        g1, g2 = res['g']
        flux = res['flux']

        if (verbose > 2):
            print(" {:5d} {:5s} {:7.1f} {:7.1f} {:7.3f} {:7.3f} {:10.7f} {:10.7f} {:7.3f} {:3d} {:12.3f} ".format(icnt,ftype,x,y,dx,dy,g1,g2,T,flag,flux))

    except Exception as e:
        print("Exception: ",e)
        flag |= BAD_MEASUREMENT

    return dx, dy, g1,g2, T, flux, flag


########################################
def get_piff_size(psf,xpos,ypos,cnum=0,color=None,verbose=0):
 
    fwhm=np.zeros((xpos.size,ypos.size),dtype=np.float64)
    g2_amp=np.zeros((xpos.size,ypos.size),dtype=np.float64)
    g2_x0=np.zeros((xpos.size,ypos.size),dtype=np.float64)
    g2_y0=np.zeros((xpos.size,ypos.size),dtype=np.float64)
    g2_sx=np.zeros((xpos.size,ypos.size),dtype=np.float64)
    g2_sy=np.zeros((xpos.size,ypos.size),dtype=np.float64)
    g2_the=np.zeros((xpos.size,ypos.size),dtype=np.float64)
    g2_off=np.zeros((xpos.size,ypos.size),dtype=np.float64)

    if (verbose > 2):
        print("---------------------------------------------------------------------")

    for ix in range(xpos.size):
        for iy in range(ypos.size):
#            print(psf.interp_property_names)
            if (color is None):
                im=psf.draw(x=xpos[ix],y=ypos[iy],chipnum=cnum,flux=1.0)
            else:
                if (color == 'GI_COLOR'):
                    im=psf.draw(x=xpos[ix],y=ypos[iy],chipnum=cnum,flux=1.0,GI_COLOR=0.6)
                elif (color == 'IZ_COLOR'):
                    im=psf.draw(x=xpos[ix],y=ypos[iy],chipnum=cnum,flux=1.0,IZ_COLOR=0.34)
#            psf_im=im.array 
            nrow,ncol = im.array.shape
            px0=nrow/2.
            py0=ncol/2.
            py, px = np.indices(im.array.shape)
            pr = np.sqrt((px - px0)**2 + (py - py0)**2)

            radbin=np.arange(0.,20.,0.5)
            r0=pr.reshape(pr.size)
            i0=im.array.reshape(im.array.size)
            s0=np.zeros(i0.size)
            s0+=1.0
#            print(s0.shape)

            r0sind=np.argsort(r0)
            r0s=r0[r0sind]
            i0s=i0[r0sind]
            maxval=np.amax(im.array)
            i=0
            while(((i0s[i])>0.5*maxval)and(i<r0.size-1)):
                i=i+1
            if (i == 0):
                sig_guess=2.0/2.35482
                print("Warning: encountered non-centered PSF model? at {:.1f},{:.1f} ".format(xpos[ix],ypos[iy]))
            else:
                sig_guess=r0s[i]/2.35482
            amp_guess=maxval/(sig_guess*2.35482)
#
#           Circularly symmetric Gaussian (with fixed center)
#
#            r_guess=[amp_guess,sig_guess,0.0]
#            popt,pcov=curve_fit(rgauss,r0,i0,p0=r_guess,sigma=s0,absolute_sigma=False)
#            perr = np.sqrt(np.diag(pcov))
#            if (verbose > 1):
#                print("                 Simple radially symmetric Gauss FIT results (amp, sig, bkg): {:.3f}               {:.3f} {:.3f} ".format(
#                    popt[0],popt[1],popt[2]))
#                print("                 Simple radially symmetric Gauss FIT results     perr(covar): {:.3f}               {:.3f} {:.3f} ".format(
#                    perr[0],perr[1],perr[2]))
#            i_resid=rgauss(r0,*popt)-i0s
#            g_fitval=rgauss(radbin,*popt)
#            fwhm[ix,iy]=popt[1]*2.35482

#
#           Circularly symmetric Gaussian (with floating center)
#
#            r_guess2=[amp_guess,px0,py0,sig_guess,0.0]
#            z0=im.array.reshape(im.array.size)
#            px=px.reshape(px.size)
#            py=py.reshape(py.size)
#            xdata_tuple=(px, py) 
#            popt2,pcov2=curve_fit(rgauss2,xdata_tuple,z0,p0=r_guess2,sigma=s0,absolute_sigma=False)
#            perr2 = np.sqrt(np.diag(pcov2))
#            if (verbose > 1):
#                print("      Floating radially symmetric Gauss FIT results (amp, x0, y0,  sig, bkg): {:.3f} {:6.3f} {:6.3f} {:.3f} {:.3f} ".
#                    format(popt2[0],popt2[1]-px0,popt2[2]-py0,popt2[3],popt2[4]))
#                print("      Floating radially symmetric Gauss FIT results              perr(covar): {:.3f} {:6.3f} {:6.3f} {:.3f} {:.3f} ".
#                    format(perr2[0],perr2[1],perr2[2],perr2[3],perr2[4]))
#            fwhm2[ix,iy]=popt2[3]*2.35482

#
#           Elliptical Gaussian (with floating center)
#
            z0=im.array.reshape(im.array.size)
            e_guess=[amp_guess,px0,py0,sig_guess,sig_guess,0.0,0.0]
            xdata_tuple=(px, py)
            try:
                popt3,pcov3=curve_fit(twoD_Gaussian,xdata_tuple, z0, p0=e_guess, sigma=s0, absolute_sigma=False)
                perr3 = np.sqrt(np.diag(pcov3))
#               Force position angle fall in the range +/-pi
                angle=popt3[5];
                n2pi=np.rint(angle/(np.pi))
                angle=angle-(n2pi*np.pi)

                if (verbose > 2):
                    print("ix,iy={:d},{:d}".format(ix,iy))
                    print(" Floating elliptical Gauss FIT results (amp, x0, y0, sigx, sigy, theta, bkg): {:.3f} {:6.3f} {:6.3f} {:.3f} {:.3f} {:6.3f} {:.3f} ".format(
                        popt3[0],popt3[1]-px0,popt3[2]-py0,popt3[3],popt3[4],angle,popt3[6]))
                    print(" Floating elliptical Gauss FIT results                           perr(covar): {:.3f} {:6.3f} {:6.3f} {:.3f} {:.3f} {:6.3f} {:.3f} ".format(
                        perr3[0],perr3[1],perr3[2],perr3[3],perr3[4],perr3[5],perr3[6]))

                g2_amp[ix,iy]=popt3[0]
                g2_x0[ix,iy]=popt3[1]-px0
                g2_y0[ix,iy]=popt3[2]-py0
                g2_sx[ix,iy]=popt3[3]*2.35482
                g2_sy[ix,iy]=popt3[4]*2.35482
                g2_the[ix,iy]=angle
                g2_off[ix,iy]=popt3[6]
                fwhm[ix,iy]=2.35482*0.5*(np.abs(popt3[3])+np.abs(popt3[4]))
            except Exception as e:
                print("Exception: ",e)
                g2_amp[ix,iy]=-1.0
                g2_x0[ix,iy]=-99.0
                g2_y0[ix,iy]=-99.0
                g2_sx[ix,iy]=-99.0
                g2_sy[ix,iy]=-99.0
                g2_the[ix,iy]=-99.0
                g2_off[ix,iy]=-99.0
                fwhm[ix,iy]=-99.0

#   Done, package fit results so that they can be preserved for QA plots and outlier examination

    g2out={'amp':g2_amp,'x0':g2_x0,'y0':g2_y0,'sx':g2_sx,'sy':g2_sy,'theta':g2_the,'off':g2_off}
    if (verbose > 2):
        print("---------------------------------------------------------------------")

    return fwhm,g2out

#
#   RAG desprecated 
#   circularly symmetric, fixed center Gaussian case
#########################################
#def rgauss(r,a,b,c):
#    return a*np.exp(-(r*r)/(2.*b*b))+c

#   RAG desprecated 
#   circularly symmetric, fitted center Gaussian case
#########################################
#def rgauss2(xdata_tuple,a,x0,y0,b,c):
#    (x,y)=xdata_tuple
#    r2=((x-x0)**2)+((y-y0)**2)
#    return a*np.exp(-(r2)/(2.*b*b))+c

#   twoD-elliptical Gaussian with fitted center
#########################################
def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x,y)=xdata_tuple
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()



###########################################
def medclip(data,clipsig=4.0,maxiter=10,converge_num=0.0001,verbose=0):
    ct = data.size
    iter = 0; c1 = 1.0 ; c2 = 0.0

    avgval = np.mean(data)
    medval = np.median(data)
    sig = np.std(data)
    wsm = np.where( abs(data-medval) < clipsig*sig )
    if ((verbose > 0)and(verbose < 4)):
        print("iter,avgval,medval,sig")
    if ((verbose > 2)and(verbose < 4)):
        print(0,avgval,medval,sig)
    if (verbose > 3):
        print("iter,avgval,medval,sig,ct,c1,c2")
        print(0,avgval,medval,sig,ct,c1,c2)

    while (c1 >= c2) and (iter < maxiter):
        iter += 1
        lastct = ct
        avgval = np.mean(data[wsm])
        medval = np.median(data[wsm])
        sig = np.std(data[wsm])
        wsm = np.where( abs(data-medval) < clipsig*sig )
        ct = len(wsm[0])
        if ct > 0:
            c1 = abs(ct - lastct)
            c2 = converge_num * lastct
        if ((verbose > 2)and(verbose < 4)):
            print(iter,avgval,medval,sig)
#        print ct,c1,c2
        if (verbose > 3):
            print(iter,avgval,medval,sig,ct,c1,c2)
#   End of while loop
    if (iter >= maxiter):
        print("Warning: medclip had not yet converged after {:d} iterations".format(iter))

    medval = np.median(data[wsm])
    avgval = np.mean(data[wsm])
    stdval = np.std(data[wsm])
    if (verbose > 0):
        print(iter+1,avgval,medval,sig)

    return avgval,medval,stdval

