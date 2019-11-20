#!/usr/bin/env python

# $Id$
# $Rev::                                  $:  # Revision of last commit.
# $LastChangedBy::                        $:  # Author of last commit.
# $LastChangedDate::                      $:  # Date of last commit.

"""Analyze PIFF model to provide QA feedback
"""

from __future__ import print_function
import time
#import datetime
import sys
import numpy as np
import fitsio
from scipy.optimize import curve_fit

#import despydb.desdbi
#import despyPIFF.DECam_focal_plane as DFP

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from matplotlib.collections import PolyCollection
#from matplotlib.patches import Polygon

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
def do_ngmix_fit(im,wt,x,y,fwhm,icnt=0,ftype='star',verbose=0):

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
        cen_prior=priors.CenPrior(0.0,0.0,pixel_scale,pixel_scale)
        gprior=priors.GPriorBA(0.1)
        Tprior=priors.LogNormal(T,0.2)
        Fprior=priors.FlatPrior(-10.,1.e10)
        prior=joint_prior.PriorSimpleSep(cen_prior,gprior,Tprior,Fprior)
#
#       Putting data in context for NGMIX
#
        jac=ngmix.Jacobian(wcs=wcs, x=cen.x + x - int(x+0.5), y=cen.y + y -int(y+0.5))
        obs=ngmix.Observation(image=im.array,weight=wt.array,jacobian=jac)
        lm_pars={'maxfev':4000}
        runner=ngmix.bootstrap.PSFRunner(obs,'gauss',T,lm_pars,prior=prior)
        runner.go(ntry=3)
#
#       FIT
#
        ngmix_flag=runner.fitter.get_result()['flags']
        gmix=runner.fitter.get_gmix()

        if (ngmix_flag != 0):
            flag |= BAD_MEASUREMENT
        dx,dy=gmix.get_cen()
        g1,g2,T=gmix.get_g1g2T()
        flux=gmix.get_flux()/wcs.pixelArea()

        if (verbose > 2):
            print(" {:5d} {:5s} {:7.3f} {:7.3f} {:10.7f} {:10.7f} {:7.3f} {:3d} {:12.3f} ".format(icnt,ftype,dx,dy,g1,g2,T,flag,flux))

    except Exception as e:
        print("Exception: ",e)
        flag |= BAD_MEASUREMENT
            
    return dx, dy, g1,g2, T, flux, flag


########################################
def get_piff_size(psf,xpos,ypos,verbose=0):
 
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
            im=psf.draw(x=xpos[ix],y=ypos[iy],flux=1.0)
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
            popt3,pcov3=curve_fit(twoD_Gaussian,xdata_tuple, z0, p0=e_guess, sigma=s0, absolute_sigma=False)
            perr3 = np.sqrt(np.diag(pcov3))
#           Force position angle fall in the range +/-pi
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

#   Done, package fit results so that they can be preserved for QA plots and outlier examination

    g2out={'amp':g2_amp,'x0':g2_x0,'y0':g2_y0,'sx':g2_sx,'sy':g2_sy,'theta':g2_the,'off':g2_off}
    if (verbose > 2):
        print("---------------------------------------------------------------------")

    return fwhm,g2out


#########################################
def rgauss(r,a,b,c):
    return a*np.exp(-(r*r)/(2.*b*b))+c

#########################################
def rgauss2(xdata_tuple,a,x0,y0,b,c):
    (x,y)=xdata_tuple
    r2=((x-x0)**2)+((y-y0)**2)
    return a*np.exp(-(r2)/(2.*b*b))+c

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



#########################################
#def plot_FP_QA(fname,qa_result,verbose=0):
#
#    """ Takes a set of QA results (one per ccd) and makes a per FP plot
#
#        Inputs:
#            fname:      Output plot file name.
#            qa_result:  Dict (per ccd/Cat) with munged QA data (the combination of which can populate a FP image)
#
#        Returns:
#            RetCode:    
#    """
#   
##
##   Arrays to hold corners of boxes that will provide polygons to be rendered
##
#    tmp_x1=[]
#    tmp_x2=[]
#    tmp_x3=[]
#    tmp_x4=[]
#
#    tmp_y1=[]
#    tmp_y2=[]
#    tmp_y3=[]
#    tmp_y4=[]
#
#    tmp_z1=[]
#    g2_list=['g2_amp','g2_x0','g2_y0','g2_sx','g2_sy','g2_the','g2_off']
#    g2_dict={'g2_amp':1,'g2_x0':2,'g2_y0':3,'g2_sx':4,'g2_sy':5,'g2_the':6,'g2_off':7}
#    g2_norm={'g2_amp':True,'g2_x0':True,'g2_y0':True,'g2_sx':True,'g2_sy':True,'g2_the':False,'g2_off':True}
#    tmp_z2={}
#    for g2par in g2_list:
#        tmp_z2[g2par]=[]
#
##
##   Form the arrays that control rendering
##
#    cat_list=[]
#    for Cat in qa_result:
#        if (Cat not in ['outland']):
#            cat_list.append(Cat)
#
#    for Cat in cat_list:
#        ccd=qa_result[Cat]['ccdnum']
#
#        x0=DFP.DECam_FP_layout[ccd]['x1']
#        y0=DFP.DECam_FP_layout[ccd]['y1']
#
#        bs=qa_result[Cat]['fwhm_map']['bs']
#        hbs=bs/2.
#        for ix in range(qa_result[Cat]['fwhm_map']['xpos'].size):
#            xp=qa_result[Cat]['fwhm_map']['xpos'][ix]
#            for iy in range(qa_result[Cat]['fwhm_map']['ypos'].size):
#                yp=qa_result[Cat]['fwhm_map']['ypos'][iy]
#
#                tmp_x1.append(x0+xp-hbs)
#                tmp_x2.append(x0+xp+hbs)
#                tmp_x3.append(x0+xp+hbs)
#                tmp_x4.append(x0+xp-hbs)
#
#                tmp_y1.append(y0+yp-hbs)
#                tmp_y2.append(y0+yp-hbs)
#                tmp_y3.append(y0+yp+hbs)
#                tmp_y4.append(y0+yp+hbs)
#
#                tmp_z1.append(qa_result[Cat]['fwhm_map']['fwhm'][ix,iy])
#                for g2par in g2_list:
#                    tmp_z2[g2par].append(qa_result[Cat]['fwhm_map'][g2par][ix,iy])
#
#    tmp_x1=np.array(tmp_x1)
#    tmp_x2=np.array(tmp_x2)
#    tmp_x3=np.array(tmp_x3)
#    tmp_x4=np.array(tmp_x4)
#    tmp_y1=np.array(tmp_y1)
#    tmp_y2=np.array(tmp_y2)
#    tmp_y3=np.array(tmp_y3)
#    tmp_y4=np.array(tmp_y4)
#
#    tmp_z1=np.array(tmp_z1)
#    for g2par in g2_list:
#        tmp_z2[g2par]=np.array(tmp_z2[g2par])
#
##   Note the following would have been for the simple (FWHM plot that has been remove below)
##   Clip extent of values to those within +/- 5 sigma 
##
##    z1_a,z1_m,z1_s=medclip(tmp_z1,clipsig=5.0,verbose=0)
##    z1_min=z1_a-(5.0*z1_s)
##    z1_max=z1_a+(5.0*z1_s)
##    print(z1_a,z1_m,z1_s,z1_min,z1_max)
##    wsm=np.where(tmp_z1>z1_max)
##    tmp_z1[wsm]=z1_max
##    wsm=np.where(tmp_z1<z1_min)
##    tmp_z1[wsm]=z1_min
#
##
##   Form polygons
##
#    y_box=np.array([tmp_x1,tmp_x2,tmp_x3,tmp_x4],'f4')
#    x_box=np.array([tmp_y1,tmp_y2,tmp_y3,tmp_y4],'f4')
#    pols=zip(x_box,y_box)
#    pols=np.swapaxes(pols,0,2)
#    pols=np.swapaxes(pols,1,2)
###
###   Plot simple version... just the FWHM across the focal plane
###
##    plt.figure(figsize=(6,9),dpi=90)
##    plt.rc('font',size=8)
##    cm=plt.get_cmap("rainbow")
##
##    ax=plt.subplot(2,1,1)
##    ax.axis([28672,1,28762,1])
##    ax.set_aspect(1.0)
##    coll=PolyCollection(pols,array=tmp_z1,cmap=cm,edgecolor='none',zorder=2)
##    plt.gca().add_collection(coll)
##    plt.colorbar(coll,ax=ax)
##    plt.title('Gauss Fit(PIFF_PSF)')
##
##    plt.savefig('junk_FP_width.png')
##    plt.close()
#
#    plt.figure(figsize=(12,9),dpi=90)
#    plt.rc('font',size=8)
#    cm=plt.get_cmap("rainbow")
#
##
##   Set outliers to max and min values....
##
#    for g2par in g2_list:
#        if (g2_norm[g2par]):
#            try:
#                z2_min=qa_result['outland'][g2par]['min_out']
#                z2_max=qa_result['outland'][g2par]['max_out']
##                print("Using pre-calc range")
#            except:
#                z2_a,z2_m,z2_s=medclip(tmp_z2[g2par],clipsig=5.0,verbose=0)
#                z2_min=z2_a-(5.0*z2_s)
#                z2_max=z2_a+(5.0*z2_s)
##                print("Calculated plotting range on fly")
#            wsm=np.where(tmp_z2[g2par]>z2_max)
#            tmp_z2[g2par][wsm]=z2_max
#            wsm=np.where(tmp_z2[g2par]<z2_min)
#            tmp_z2[g2par][wsm]=z2_min
#        ax=plt.subplot(3,3,g2_dict[g2par])
#        ax.axis([28672,1,28762,1])
#        ax.set_aspect(1.0)
#        coll2=PolyCollection(pols,array=tmp_z2[g2par],cmap=cm,edgecolor='none',zorder=2)
#        plt.gca().add_collection(coll2)
#        plt.colorbar(coll2,ax=ax)
#        plt.title(g2par)
##
##   Stellar distribution plot
##
#    nstar=0
#    for Cat in cat_list:
#        nstar+=qa_result[Cat]['star_data']['x'].size
#
##    print("Nstar = ",nstar)
#
#    xstar=np.zeros(nstar,dtype='f8')
#    ystar=np.zeros(nstar,dtype='f8')
#
#    ctr=0
#    for Cat in cat_list:
#        ccd=qa_result[Cat]['ccdnum']
#
#        x0=DFP.DECam_FP_layout[ccd]['x1']
#        y0=DFP.DECam_FP_layout[ccd]['y1']
#
#        ka=qa_result[Cat]['star_data']['x']+x0
#        ks=ka.size
#        xstar[ctr:ctr+ks]=ka
#        ka=qa_result[Cat]['star_data']['y']+y0
#        ks=ka.size
#        ystar[ctr:ctr+ks]=ka
#        ctr+=ks
#
#    ax=plt.subplot(3,3,8)
#    ax.axis([28672,1,28762,1])
#    ax.set_aspect(1.0)
#    plt.scatter(ystar,xstar,1,marker='.',color='blue')
#
#    plt.savefig(fname)
#    plt.close()
#
###    return 0
