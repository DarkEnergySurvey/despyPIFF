#!/usr/bin/env python

# $Id$
# $Rev::                                  $:  # Revision of last commit.
# $LastChangedBy::                        $:  # Author of last commit.
# $LastChangedDate::                      $:  # Date of last commit.

"""Analyze PIFF model to provide QA feedback
"""

from __future__ import print_function
import time
import sys
import numpy as np

#import despyPIFF.quick_stat as qs
import despyPIFF.DECam_focal_plane as DFP
import despyPIFF.piff_qa_utils as pqu

#from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon

#import piff
#import galsim
#from ngmix import priors, joint_prior
#import ngmix 

#########################################
def plot_FP_QA(fname,qa_result,verbose=0):

    """ Takes a set of QA results (one per ccd) and makes a per FP plot

        Inputs:
            fname:      Output plot file name.
            qa_result:  Dict (per ccd/Cat) with munged QA data (the combination of which can populate a FP image)

        Returns:
            RetCode:    
    """
   
#
#   Arrays to hold corners of boxes that will provide polygons to be rendered
#
    tmp_x1=[]
    tmp_x2=[]
    tmp_x3=[]
    tmp_x4=[]

    tmp_y1=[]
    tmp_y2=[]
    tmp_y3=[]
    tmp_y4=[]

    tmp_z1=[]
    g2_list=['g2_amp','g2_x0','g2_y0','g2_sx','g2_sy','g2_the','g2_off']
    g2_dict={'g2_amp':1,'g2_x0':2,'g2_y0':3,'g2_sx':4,'g2_sy':5,'g2_the':6,'g2_off':7}
    g2_norm={'g2_amp':True,'g2_x0':True,'g2_y0':True,'g2_sx':True,'g2_sy':True,'g2_the':False,'g2_off':True}
    tmp_z2={}
    for g2par in g2_list:
        tmp_z2[g2par]=[]

#
#   Form the arrays that control rendering
#
    cat_list=[]
    for Cat in qa_result:
        if (Cat not in ['outland']):
            cat_list.append(Cat)

    for Cat in cat_list:
        ccd=qa_result[Cat]['ccdnum']

        x0=DFP.DECam_FP_layout[ccd]['x1']
        y0=DFP.DECam_FP_layout[ccd]['y1']

        bs=qa_result[Cat]['fwhm_map']['bs']
        hbs=bs/2.
        for ix in range(qa_result[Cat]['fwhm_map']['xpos'].size):
            xp=qa_result[Cat]['fwhm_map']['xpos'][ix]
            for iy in range(qa_result[Cat]['fwhm_map']['ypos'].size):
                yp=qa_result[Cat]['fwhm_map']['ypos'][iy]

                tmp_x1.append(x0+xp-hbs)
                tmp_x2.append(x0+xp+hbs)
                tmp_x3.append(x0+xp+hbs)
                tmp_x4.append(x0+xp-hbs)

                tmp_y1.append(y0+yp-hbs)
                tmp_y2.append(y0+yp-hbs)
                tmp_y3.append(y0+yp+hbs)
                tmp_y4.append(y0+yp+hbs)

                tmp_z1.append(qa_result[Cat]['fwhm_map']['fwhm'][ix,iy])
                for g2par in g2_list:
                    tmp_z2[g2par].append(qa_result[Cat]['fwhm_map'][g2par][ix,iy])

    tmp_x1=np.array(tmp_x1)
    tmp_x2=np.array(tmp_x2)
    tmp_x3=np.array(tmp_x3)
    tmp_x4=np.array(tmp_x4)
    tmp_y1=np.array(tmp_y1)
    tmp_y2=np.array(tmp_y2)
    tmp_y3=np.array(tmp_y3)
    tmp_y4=np.array(tmp_y4)

    tmp_z1=np.array(tmp_z1)
    for g2par in g2_list:
        tmp_z2[g2par]=np.array(tmp_z2[g2par])
#
#   Note the following would have been for the simple (FWHM plot that has been remove below)
#   Clip extent of values to those within +/- 5 sigma 
#
    z1_a,z1_m,z1_s=pqu.medclip(tmp_z1,clipsig=5.0,verbose=0)
    z1_min=z1_a-(5.0*z1_s)
    z1_max=z1_a+(5.0*z1_s)
    print(z1_a,z1_m,z1_s,z1_min,z1_max)
    wsm=np.where(tmp_z1>z1_max)
    tmp_z1[wsm]=z1_max
    wsm=np.where(tmp_z1<z1_min)
    tmp_z1[wsm]=z1_min
#
#   Form polygons
#
    y_box=np.array([tmp_x1,tmp_x2,tmp_x3,tmp_x4],'f4')
    x_box=np.array([tmp_y1,tmp_y2,tmp_y3,tmp_y4],'f4')
    pols=zip(x_box,y_box)
    pols=np.swapaxes(pols,0,2)
    pols=np.swapaxes(pols,1,2)

    plt.figure(figsize=(12,9),dpi=90)
    plt.rc('font',size=8)
    cm=plt.get_cmap("rainbow")

    for g2par in g2_list:
#
#       Set outliers to max and min values....
#
        if (g2_norm[g2par]):
            try:
                z2_min=qa_result['outland'][g2par]['min_out']
                z2_max=qa_result['outland'][g2par]['max_out']
#                print("Using pre-calc range")
            except:
                z2_a,z2_m,z2_s=pqu.medclip(tmp_z2[g2par],clipsig=5.0,verbose=0)
                z2_min=z2_a-(5.0*z2_s)
                z2_max=z2_a+(5.0*z2_s)
#                print("Calculated plotting range on fly")
            wsm=np.where(tmp_z2[g2par]>z2_max)
            tmp_z2[g2par][wsm]=z2_max
            wsm=np.where(tmp_z2[g2par]<z2_min)
            tmp_z2[g2par][wsm]=z2_min
        else:
            if (g2par == "g2_the"):
                z2_min=-np.pi/2.0
                z2_max=np.pi/2.0
                wsm=np.where(tmp_z2[g2par]>z2_max)
                tmp_z2[g2par][wsm]=z2_max
                wsm=np.where(tmp_z2[g2par]<z2_min)
                tmp_z2[g2par][wsm]=z2_min
#
#       Go ahead and plot
#
        ax=plt.subplot(3,3,g2_dict[g2par])
        ax.axis([28672,1,28762,1])
        ax.set_aspect(1.0)
        coll2=PolyCollection(pols,array=tmp_z2[g2par],cmap=cm,edgecolor='none',zorder=2)
        plt.gca().add_collection(coll2)
        plt.colorbar(coll2,ax=ax)
        plt.title(g2par)
#
#   plot of FWHM
#
    ax=plt.subplot(3,3,8)
    ax.axis([28672,1,28762,1])
    ax.set_aspect(1.0)
    coll=PolyCollection(pols,array=tmp_z1,cmap=cm,edgecolor='none',zorder=2)
    plt.gca().add_collection(coll)
    plt.colorbar(coll,ax=ax)
    plt.title('FWHM')
#
#
#   Stellar distribution plot
#
    nstar=0
    for Cat in cat_list:
        nstar+=qa_result[Cat]['star_data']['x'].size

#    print("Nstar = ",nstar)

    xstar=np.zeros(nstar,dtype='f8')
    ystar=np.zeros(nstar,dtype='f8')

    ctr=0
    for Cat in cat_list:
        ccd=qa_result[Cat]['ccdnum']

        x0=DFP.DECam_FP_layout[ccd]['x1']
        y0=DFP.DECam_FP_layout[ccd]['y1']

        ka=qa_result[Cat]['star_data']['x']+x0
        ks=ka.size
        xstar[ctr:ctr+ks]=ka
        ka=qa_result[Cat]['star_data']['y']+y0
        ks=ka.size
        ystar[ctr:ctr+ks]=ka
        ctr+=ks

    ax=plt.subplot(3,3,9)
    ax.axis([28672,1,28762,1])
    ax.set_aspect(1.0)
    plt.scatter(ystar,xstar,1,marker='.',color='blue')

    plt.savefig(fname)
    plt.close()

    return 0
