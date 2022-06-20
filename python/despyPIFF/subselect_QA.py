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
from matplotlib.colors  import LogNorm

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
def plot_selection(fname,data,verbose=0):

#
#   version uses 2d-hist to get a proper plot but is no longer as facile with respect to scaling the grey levels
#
    plt.figure(figsize=(12,8),dpi=90)
    plt.rc('font',size=9)
    plt.subplot()

    plt.subplot(2,3,1)
    axrange=[-0.05,0.05,3.0,1.e4]
    xspace=np.linspace(axrange[0],axrange[1],250)
    yspace=np.logspace(np.log10(axrange[2]),np.log10(axrange[3]),100)
    plt.hist2d(data['SPREAD_MODEL'],data['SN'],bins=(xspace,yspace),cmap="Greys")
    wsm=np.where(data['GAIA_STAR']>0)
    plt.scatter(data['SPREAD_MODEL'][wsm],data['SN'][wsm],1,marker='.',color='red')
    wsm=np.where(data['GAIA_STAR']==1)
    plt.scatter(data['SPREAD_MODEL'][wsm],data['SN'][wsm],1,marker='.',color='blue')
    plt.yscale('log')
    plt.axis(axrange)
    plt.title('GAIA Selection')
    plt.xlabel('Spread_Model')
    plt.ylabel('S/N')

    plt.subplot(2,3,2)
    axrange=[0.0,10.0,3.0,1.e4]
    xspace=np.linspace(axrange[0],axrange[1],250)
    yspace=np.logspace(np.log10(axrange[2]),np.log10(axrange[3]),100)
    plt.hist2d(data['FLUX_RADIUS'],data['SN'],bins=(xspace,yspace),cmap="Greys")
    wsm=np.where(data['GAIA_STAR']>0)
    plt.scatter(data['FLUX_RADIUS'][wsm],data['SN'][wsm],1,marker='.',color='red')
    wsm=np.where(data['GAIA_STAR']==1)
    plt.scatter(data['FLUX_RADIUS'][wsm],data['SN'][wsm],1,marker='.',color='blue')
    plt.yscale('log')
    plt.axis(axrange)
    plt.title('GAIA Selection')
    plt.xlabel('Flux_Radius[pix]')
    plt.ylabel('S/N')

    plt.subplot(2,3,3)
    axrange=[0.0,10.0,3.0,1.e4]
    xspace=np.linspace(axrange[0],axrange[1],250)
    yspace=np.logspace(np.log10(axrange[2]),np.log10(axrange[3]),100)
    plt.hist2d(data['FLUX_RADIUS'],data['SN'],bins=(xspace,yspace),cmap="Greys")
    wsm=np.where(data['EXT_MASH']==0)
    plt.scatter(data['FLUX_RADIUS'][wsm],data['SN'][wsm],1,marker='.',color='blue')
    plt.yscale('log')
    plt.axis(axrange)
    plt.title('EXT_MASH=0')
    plt.xlabel('Flux_Radius[pix]')
    plt.ylabel('S/N')

    plt.subplot(2,3,5)
    axrange=[-0.5,3.5,-1.5,3.0]
    xspace=np.linspace(axrange[0],axrange[1],250)
    yspace=np.linspace(axrange[2],axrange[3],250)
    wsm=np.where(np.logical_and(np.logical_and(data['R_MAG']>-99.,data['Z_MAG']>-99.),data['K_MAG']>-99.))
    rz_color=data['R_MAG'][wsm]-data['Z_MAG'][wsm]
    zk_color=data['Z_MAG'][wsm]-data['K_MAG'][wsm]
    plt.hist2d(rz_color,zk_color,bins=(xspace,yspace),cmap="Greys",norm=LogNorm(vmin=0.5,vmax=10.0))

    wsm=np.where(np.logical_and(np.logical_and(data['R_MAG']>-99.,data['Z_MAG']>-99.),np.logical_and(data['K_MAG']>-99.,data['GAIA_STAR']==1)))
    rz_color=data['R_MAG'][wsm]-data['Z_MAG'][wsm]
    zk_color=data['Z_MAG'][wsm]-data['K_MAG'][wsm]
    plt.scatter(rz_color,zk_color,1,marker='.',color='blue')
##    plt.yscale('log')
    plt.axis(axrange)
    plt.title('GAIA_STAR=1 w/ VHS')
    plt.xlabel('r-z (mag)')
    plt.ylabel('z-K (mag)')

    plt.subplot(2,3,6)
    axrange=[-0.5,3.5,-1.5,3.0]
    xspace=np.linspace(axrange[0],axrange[1],250)
    yspace=np.linspace(axrange[2],axrange[3],250)
    wsm=np.where(np.logical_and(np.logical_and(data['R_MAG']>-99.,data['Z_MAG']>-99.),data['K_MAG']>-99.))
    rz_color=data['R_MAG'][wsm]-data['Z_MAG'][wsm]
    zk_color=data['Z_MAG'][wsm]-data['K_MAG'][wsm]
    plt.hist2d(rz_color,zk_color,bins=(xspace,yspace),cmap="Greys",norm=LogNorm(vmin=0.5,vmax=10.0))

    wsm=np.where(np.logical_and(np.logical_and(data['R_MAG']>-99.,data['Z_MAG']>-99.),np.logical_and(data['K_MAG']>-99.,data['EXT_MASH']==0)))
    rz_color=data['R_MAG'][wsm]-data['Z_MAG'][wsm]
    zk_color=data['Z_MAG'][wsm]-data['K_MAG'][wsm]
    plt.scatter(rz_color,zk_color,1,marker='.',color='blue')
    plt.axis(axrange)
    plt.title('EXT_MASH=0 w/ VHS')
    plt.xlabel('r-z (mag)')
    plt.ylabel('z-K (mag)')

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
