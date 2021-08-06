#!/usr/bin/env python3

# $Id$
# $Rev::                                  $:  # Revision of last commit.
# $LastChangedBy::                        $:  # Author of last commit.
# $LastChangedDate::                      $:  # Date of last commit.

"""Utilities for making QA plots when doing subselection for PIFF inputs.
"""

#from __future__ import print_function
import numpy as np
import despyPIFF.DECam_focal_plane as DFP

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon

########################################
def quick_hess(xdata,ydata,axrange=None,nxbin=250,nybin=100,logx=False,logy=False):

    if (axrange is None):
        xmin=np.amin(xdata)
        xmax=np.amax(xdata)
        ymin=np.amin(ydata)
        ymax=np.amax(ydata)
    else:
        xmin=axrange[0]
        xmax=axrange[1]
        ymin=axrange[2]
        ymax=axrange[3]
    if (logx):
        xspace=np.logspace(np.log10(xmin),np.log10(xmax),nxbin)
    else:
        xspace=np.linspace(xmin,xmax,nxbin)
    if (logy):
        yspace=np.logspace(np.log10(ymin),np.log10(ymax),nybin)
        print(yspace)
    else:
        yspace=np.linspace(ymin,ymax,nybin)

    H,xbins,ybins=np.histogram2d(xdata,ydata,bins=(xspace,yspace))
    return H,xbins,ybins


########################################
def plot_selection(fname,data,dataCut,dataCutVHS,verbose=0):

    plt.figure(figsize=(8,8),dpi=90)
    plt.rc('font',size=9)
    plt.subplot()

    plt.subplot(2,2,1)
    plt.scatter(data['spread_model'],data['sn'],1,marker='.',color='blue')
    plt.scatter(dataCut['spread_model'],dataCut['sn'],1,marker='.',color='red')
    plt.yscale('log')
    plt.axis([-0.125,.125,1.0,3.e4])
    plt.xlabel('SPREAD_MODEL')
    plt.ylabel('S/N')
    plt.title('GAIA DR2')

    plt.subplot(2,2,2)
    plt.scatter(data['flux_radius'],data['sn'],1,marker='.',color='blue')
    plt.scatter(dataCut['flux_radius'],dataCut['sn'],1,marker='.',color='red')
    plt.yscale('log')
    plt.axis([-0.125,15.0,1.0,3.e4])
    plt.xlabel('FLUX_RADIUS [pix]')
    plt.ylabel('S/N')
    plt.title('GAIA DR2')

    plt.subplot(2,2,3)
    plt.scatter(data['spread_model'],data['sn'],1,marker='.',color='blue')
    plt.scatter(dataCutVHS['spread_model'],dataCutVHS['sn'],1,marker='.',color='red')
    plt.yscale('log')
    plt.axis([-0.125,.125,1.0,3.e4])
    plt.xlabel('SPREAD_MODEL')
    plt.ylabel('S/N')
    plt.title('VHS')

    plt.subplot(2,2,4)
    plt.scatter(data['flux_radius'],data['sn'],1,marker='.',color='blue')
    plt.scatter(dataCutVHS['flux_radius'],dataCutVHS['sn'],1,marker='.',color='red')
    plt.yscale('log')
    plt.axis([-0.125,15.0,1.0,3.e4])
    plt.xlabel('FLUX_RADIUS [pix]')
    plt.ylabel('S/N')
    plt.title('VHS')

    plt.savefig(fname)
    plt.close()

    return 0


########################################
def plot_selection2(fname,data,dataCut,dataCutVHS,verbose=0):

#
#   Attempts too create a scalable 2-d histogram (i.e. forms an image so it can be rescaled)
#   Current version does not functio (properly) after matplotlib upgrades.. :(
#

    plt.figure(figsize=(8,8),dpi=90)
    plt.rc('font',size=9)
    plt.subplot()

    plt.subplot(2,2,1)
    axrange=[-0.05,0.05,1.0,3.e4]
    H,xbins,ybins=quick_hess(data['spread_model'],data['sn'],axrange=axrange,logx=False,logy=True)
    wsm=np.where(H<0.1)
    H[wsm]=0.1
    plt.imshow(np.log10(H).T,origin='lower',extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],cmap="Greys",aspect='auto')
    plt.scatter(dataCut['spread_model'],dataCut['sn'],1,marker='.',color='red')
    plt.yscale('log')
    plt.axis(axrange)
    plt.title('GAIA_DR2 Selection')
    plt.xlabel('Spread_Model')
    plt.ylabel('S/N')

    plt.subplot(2,2,2)
    axrange=[0.0,12.0,1.0,3.e4]
    H,xbins,ybins=quick_hess(data['flux_radius'],data['sn'],axrange=axrange,logx=False,logy=True)
    plt.yscale('log')
    wsm=np.where(H<0.1)
    H[wsm]=0.1
    plt.imshow(np.sqrt(H).T,origin='lower',extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],cmap="Greys",aspect='auto')
    plt.scatter(dataCut['flux_radius'],dataCut['sn'],1,marker='.',color='red')
    plt.axis(axrange)
    plt.title('GAIA_DR2 Selection')
    plt.xlabel('Flux_Radius[pix]')
    plt.ylabel('S/N')

    plt.subplot(2,2,3)
    axrange=[-0.05,0.05,1.0,3.e4]
    H,xbins,ybins=quick_hess(data['spread_model'],data['sn'],axrange=axrange,logx=False,logy=True)
    plt.yscale('log')
    wsm=np.where(H<0.1)
    H[wsm]=0.1
    plt.imshow(np.sqrt(H).T,origin='lower',extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],cmap="Greys",aspect='auto')
    if (dataCutVHS['sn'].size > 0):
        plt.scatter(dataCutVHS['spread_model'],dataCutVHS['sn'],1,marker='.',color='red')
    else:
        xtxt=axrange[0]+(0.025*(axrange[1]-axrange[0]))
        ytxt=10.0**(np.log10(axrange[3])-(0.100*(np.log10(axrange[3])-np.log10(axrange[2]))))
        plt.text(xtxt,ytxt,'No VHS matches',color='red')
    plt.axis(axrange)
    plt.title('VHS Selection')
    plt.xlabel('Spread_Model')
    plt.ylabel('S/N')

    plt.subplot(2,2,4)
    axrange=[0.0,12.0,1.0,3.e4]
    H,xbins,ybins=quick_hess(data['flux_radius'],data['sn'],axrange=axrange,logx=False,logy=True)
    wsm=np.where(H<0.1)
    H[wsm]=0.1
    plt.yscale('log')
    plt.imshow(np.sqrt(H).T,origin='lower',extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],cmap="Greys",aspect='auto')
    if (dataCutVHS['sn'].size > 0):
        plt.scatter(dataCutVHS['flux_radius'],dataCutVHS['sn'],1,marker='.',color='red')
    else:
        xtxt=axrange[0]+(0.025*(axrange[1]-axrange[0]))
        ytxt=10.0**(np.log10(axrange[3])-(0.100*(np.log10(axrange[3])-np.log10(axrange[2]))))
        plt.text(xtxt,ytxt,'No VHS matches',color='red')
    plt.axis(axrange)
    plt.title('VHS Selection')
    plt.xlabel('Flux_Radius[pix]')
    plt.ylabel('S/N')

    plt.savefig(fname)
    plt.close()

    return 0


########################################
def plot_selection3(fname,data,dataCut,dataCutVHS,verbose=0):

#
#   version uses 2d-hist to get a proper plot but is no longer as facile with respect to scaling the grey levels
#
    plt.figure(figsize=(8,8),dpi=90)
    plt.rc('font',size=9)
    plt.subplot()

    plt.subplot(2,2,1)
    axrange=[-0.05,0.05,1.0,3.e4]
    xspace=np.linspace(axrange[0],axrange[1],250)
    yspace=np.logspace(np.log10(axrange[2]),np.log10(axrange[3]),100)
    plt.hist2d(data['spread_model'],data['sn'],bins=(xspace,yspace),cmap="Greys")
    plt.scatter(dataCut['spread_model'],dataCut['sn'],1,marker='.',color='red')
    plt.yscale('log')
    plt.axis(axrange)
    plt.title('GAIA_DR2 Selection')
    plt.xlabel('Spread_Model')
    plt.ylabel('S/N')

    plt.subplot(2,2,2)
    axrange=[0.0,12.0,1.0,3.e4]
    xspace=np.linspace(axrange[0],axrange[1],250)
    yspace=np.logspace(np.log10(axrange[2]),np.log10(axrange[3]),100)
    plt.hist2d(data['flux_radius'],data['sn'],bins=(xspace,yspace),cmap="Greys")
    plt.scatter(dataCut['flux_radius'],dataCut['sn'],1,marker='.',color='red')
    plt.yscale('log')
    plt.axis(axrange)
    plt.title('GAIA_DR2 Selection')
    plt.xlabel('Flux_Radius[pix]')
    plt.ylabel('S/N')

    plt.subplot(2,2,3)
    axrange=[-0.05,0.05,1.0,3.e4]
    xspace=np.linspace(axrange[0],axrange[1],250)
    yspace=np.logspace(np.log10(axrange[2]),np.log10(axrange[3]),100)
    plt.hist2d(data['spread_model'],data['sn'],bins=(xspace,yspace),cmap="Greys")
    if (dataCutVHS['sn'].size > 0):
        plt.scatter(dataCutVHS['spread_model'],dataCutVHS['sn'],1,marker='.',color='red')
    else:
        xtxt=axrange[0]+(0.025*(axrange[1]-axrange[0]))
        ytxt=10.0**(np.log10(axrange[3])-(0.100*(np.log10(axrange[3])-np.log10(axrange[2]))))
        plt.text(xtxt,ytxt,'No VHS matches',color='red')
    plt.yscale('log')
    plt.axis(axrange)
    plt.title('VHS Selection')
    plt.xlabel('Spread_Model')
    plt.ylabel('S/N')

    plt.subplot(2,2,4)
    axrange=[0.0,12.0,1.0,3.e4]
    xspace=np.linspace(axrange[0],axrange[1],250)
    yspace=np.logspace(np.log10(axrange[2]),np.log10(axrange[3]),100)
    plt.hist2d(data['flux_radius'],data['sn'],bins=(xspace,yspace),cmap="Greys")
    if (dataCutVHS['sn'].size > 0):
        plt.scatter(dataCutVHS['flux_radius'],dataCutVHS['sn'],1,marker='.',color='red')
    else:
        xtxt=axrange[0]+(0.025*(axrange[1]-axrange[0]))
        ytxt=10.0**(np.log10(axrange[3])-(0.100*(np.log10(axrange[3])-np.log10(axrange[2]))))
        plt.text(xtxt,ytxt,'No VHS matches',color='red')
    plt.yscale('log')
    plt.axis(axrange)
    plt.title('VHS Selection')
    plt.xlabel('Flux_Radius[pix]')
    plt.ylabel('S/N')

    plt.savefig(fname)
    plt.close()

    return 0


########################################
def plot_FP_quant(fname,data,verbose=0):

    tmp_x1=np.array([DFP.DECam_FP_layout[ccd]['x1'] for ccd in range(1,63) if (ccd in data['m_gaia'])])
    tmp_x2=np.array([DFP.DECam_FP_layout[ccd]['x2'] for ccd in range(1,63) if (ccd in data['m_gaia'])])
    tmp_x3=np.array([DFP.DECam_FP_layout[ccd]['x2'] for ccd in range(1,63) if (ccd in data['m_gaia'])])
    tmp_x4=np.array([DFP.DECam_FP_layout[ccd]['x1'] for ccd in range(1,63) if (ccd in data['m_gaia'])])
    tmp_y1=np.array([DFP.DECam_FP_layout[ccd]['y1'] for ccd in range(1,63) if (ccd in data['m_gaia'])])
    tmp_y2=np.array([DFP.DECam_FP_layout[ccd]['y1'] for ccd in range(1,63) if (ccd in data['m_gaia'])])
    tmp_y3=np.array([DFP.DECam_FP_layout[ccd]['y2'] for ccd in range(1,63) if (ccd in data['m_gaia'])])
    tmp_y4=np.array([DFP.DECam_FP_layout[ccd]['y2'] for ccd in range(1,63) if (ccd in data['m_gaia'])])
    y_box=np.array([tmp_x1,tmp_x2,tmp_x3,tmp_x4],'f4')
    x_box=np.array([tmp_y1,tmp_y2,tmp_y3,tmp_y4],'f4')
    pols=np.array(list(zip(x_box,y_box)))
    pols=np.swapaxes(pols,0,2)
    pols=np.swapaxes(pols,1,2)

    plt.figure(figsize=(12,6),dpi=90)
    plt.rc('font',size=8)
    cm=plt.get_cmap("rainbow")

    tmp_zval=np.array([data['m_gaia'][ccd] for ccd in range(1,63) if (ccd in data['m_gaia'])])
    ax=plt.subplot(2,3,1)
    ax.axis([28672,1,28762,1])
#    ax.axis([1,28672,1,28762])
    coll=PolyCollection(pols,array=tmp_zval,cmap=cm,edgecolor='none',zorder=2)
    plt.gca().add_collection(coll)
    plt.colorbar(coll,ax=ax)
    plt.title('GAIA Median(FLUX_RADIUS)')

    tmp_zval2=np.array([data['s_gaia'][ccd] for ccd in range(1,63) if (ccd in data['s_gaia'])])
    ax=plt.subplot(2,3,2)
    ax.axis([28672,1,28762,1])
    coll2=PolyCollection(pols,array=tmp_zval2,cmap=cm,edgecolor='none',zorder=2)
    plt.gca().add_collection(coll2)
    plt.colorbar(coll2,ax=ax)
    if ('EXPNUM' in data):
        plt.title('Exp={:d}\nGAIA StdDev(FLUX_RADIUS)'.format(data['EXPNUM']))
    else:
        plt.title('GAIA StdDev(FLUX_RADIUS)')

    tmp_zval3=np.array([data['n_gaia'][ccd] for ccd in range(1,63) if (ccd in data['n_gaia'])])
    ax=plt.subplot(2,3,3)
    ax.axis([28672,1,28762,1])
    coll2=PolyCollection(pols,array=tmp_zval3,cmap=cm,edgecolor='none',zorder=2)
    plt.gca().add_collection(coll2)
    plt.colorbar(coll2,ax=ax)
    plt.title('# GAIA Stars selected')


    tmp_x1=np.array([DFP.DECam_FP_layout[ccd]['x1'] for ccd in range(1,63) if (ccd in data['m_vhs'])])
    tmp_x2=np.array([DFP.DECam_FP_layout[ccd]['x2'] for ccd in range(1,63) if (ccd in data['m_vhs'])])
    tmp_x3=np.array([DFP.DECam_FP_layout[ccd]['x2'] for ccd in range(1,63) if (ccd in data['m_vhs'])])
    tmp_x4=np.array([DFP.DECam_FP_layout[ccd]['x1'] for ccd in range(1,63) if (ccd in data['m_vhs'])])
    tmp_y1=np.array([DFP.DECam_FP_layout[ccd]['y1'] for ccd in range(1,63) if (ccd in data['m_vhs'])])
    tmp_y2=np.array([DFP.DECam_FP_layout[ccd]['y1'] for ccd in range(1,63) if (ccd in data['m_vhs'])])
    tmp_y3=np.array([DFP.DECam_FP_layout[ccd]['y2'] for ccd in range(1,63) if (ccd in data['m_vhs'])])
    tmp_y4=np.array([DFP.DECam_FP_layout[ccd]['y2'] for ccd in range(1,63) if (ccd in data['m_vhs'])])
    y_box=np.array([tmp_x1,tmp_x2,tmp_x3,tmp_x4],'f4')
    x_box=np.array([tmp_y1,tmp_y2,tmp_y3,tmp_y4],'f4')
    pols=np.array(list(zip(x_box,y_box)))
    pols=np.swapaxes(pols,0,2)
    pols=np.swapaxes(pols,1,2)

    tmp_zval=np.array([data['m_vhs'][ccd] for ccd in range(1,63) if (ccd in data['m_vhs'])])
    if (tmp_zval.size > 0):
        ax=plt.subplot(2,3,4)
        ax.axis([28672,1,28762,1])
#        ax.axis([1,28672,1,28762])
        coll=PolyCollection(pols,array=tmp_zval,cmap=cm,edgecolor='none',zorder=2)
        plt.gca().add_collection(coll)
        plt.colorbar(coll,ax=ax)
        plt.title('VHS Median(FLUX_RADIUS)')

    tmp_zval2=np.array([data['s_vhs'][ccd] for ccd in range(1,63) if (ccd in data['s_vhs'])])
    if (tmp_zval2.size > 0):
        ax=plt.subplot(2,3,5)
        ax.axis([28672,1,28762,1])
        coll2=PolyCollection(pols,array=tmp_zval2,cmap=cm,edgecolor='none',zorder=2)
        plt.gca().add_collection(coll2)
        plt.colorbar(coll2,ax=ax)
        plt.title('VHS StdDev(FLUX_RADIUS)')

    tmp_zval3=np.array([data['n_vhs'][ccd] for ccd in range(1,63) if (ccd in data['n_vhs'])])
    if (tmp_zval3.size > 0):
        ax=plt.subplot(2,3,6)
        ax.axis([28672,1,28762,1])
        coll2=PolyCollection(pols,array=tmp_zval3,cmap=cm,edgecolor='none',zorder=2)
        plt.gca().add_collection(coll2)
        plt.colorbar(coll2,ax=ax)
        plt.title('# VHS Stars selected')

    plt.savefig(fname)
    plt.close()

    return 0
