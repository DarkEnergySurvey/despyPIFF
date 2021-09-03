#!/usr/bin/env python3
"""
Analyze PSF model result from PIFF, obtain and ingest QA metrics
to provide feedback to downstream utilities.

Original code from DESDM SVN by Robert Gruendl
Ported to git by Felipe Menanteau
Updated to python 3 version by Robert Gruendl
"""

#from __future__ import print_function
import argparse
import os
import re
import time
import sys
import numpy as np
import fitsio

import despydb.desdbi
#import despyPIFF.DECam_focal_plane as DFP
import despyPIFF.piff_qa_plot   as pqp
import despyPIFF.piff_qa_utils  as pqu
import despyPIFF.piff_qa_ingest as pqi

#from scipy.optimize import curve_fit

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from matplotlib.collections import PolyCollection
#from matplotlib.patches import Polygon

import piff
import galsim
from ngmix import priors, joint_prior
import ngmix 
import healpy

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
def check_PIFF_data(fname,img,seed=None,nsides=[64,16384,65536],blocksize=128,verbose=0):

    """Perform a series of checks and analyses on PSF model from PIFF
        Inputs:
            fname:      PIFF model file
            img:        Corresponding image (from which PIFF model was generated)
            blocksize:  Sampling size used when looking at generic model results (should be a factor of 2)
            verbose:    Controls the amount of information logged

        Returns:
            piff_result: Dict containing information garnered from the PIFF model.
                Low-level info includes: expnum, ccdnum
                
    """

    piff_failed=False
    piff_flag=0
    piff_result={}
    rfits=fitsio.FITS(fname,'r')
    h0=rfits[0].read_header()
    if ('expnum' in h0):
        piff_result['expnum']=int(h0['EXPNUM'])
    else:
        print("Warning! EXPNUM not present in primary HDU.  Using a value of -1 to continue will likely fail to ingest")
        piff_result['expnum']=-1
    if ('ccdnum' in h0):
        piff_result['ccdnum']=int(h0['CCDNUM'])
    else:
        print("Warning! CCDNUM not present in primary HDU.  Will attempt too poll psfstars HDU.")
        piff_result['ccdnum']=-1
#
#   Test 1:  Number of stars used
#       FLAG = 2
#
    star_cols=rfits['psf_stars'].get_colnames()
    star_data=rfits['psf_stars'].read()

    piff_result['nstar']=len(star_data)
    if (verbose > 0):
        print("  nstar used: {:d}".format(piff_result['nstar']))
    if (piff_result['nstar']<20):
        print("  Failed: too few stars")
        piff_failed=True
        piff_flag+=2
    rfits.close()

    if (piff_result['ccdnum'] == -1):
        chipnums=np.unique(star_data['chipnum'])
        if (chipnums.size > 1):
            print("Warning!  More than one CCDNUM present.  Will use first but expect anomalous results/plots.")
        else:
            print("CCDNUM quandry successfully resolved using value of: {:d}".format(int(chipnums[0])))
        piff_result['ccdnum']=int(chipnums[0])

#
#   Test 2: Readable PSF (by PIFF)
#       FLAG = 4
#
    try:
        psf=piff.read(fname)
    except Exception as e:
        print("Catastrophic PIFF failure: unable to read PSF")
        piff_failed=True
        piff_flag+=4

#
#   Test 3:  Chisq vs DOF
#       FLAG = 1
#
    piff_result['nremoved']=int(psf.nremoved)
    piff_result['chisq']=psf.chisq
    piff_result['dof']=int(psf.dof)
    if (verbose > 0):
        print("  nstar removed: {:d} ".format(psf.nremoved))
    if (psf.chisq > 1.5*psf.dof):
        piff_failed=True
        piff_flag+=1
        if (verbose > 0):
            print("  Bad PIFF solution: chisq={:.3f} for DOF={:.1f} ".format(psf.chisq,psf.dof))
    else:
        if (verbose > 0):
            print("  Acceptable PIFF solution: chisq={:.3f} for DOF={:.1f} ".format(psf.chisq,psf.dof))

#
#   Test 4 (map out PSF size variation)
#       First just the size at the chip center
#       Then, sampled across the CCD
#
    xcen=np.array([1024.])
    ycen=np.array([2048.])
    fwhm_cen,g2_cen=pqu.get_piff_size(psf,xcen,ycen,cnum=piff_result['ccdnum'],verbose=verbose)
    piff_result['fwhm']=fwhm_cen[0,0]*pixel_scale
    print('Central model FWHM : {:.3f} '.format(piff_result['fwhm']))
#
#   Coarse map to examine distribution across the focal plane (and make QA plots)
#
    print("Sampling PIFF model (with stepsize = {:d} [pixels])".format(blocksize))
    t0=time.time()
    xpos=np.arange(blocksize/2.0,2048,blocksize)
    ypos=np.arange(blocksize/2.0,4096,blocksize)
    fwhm_im,g2_im=pqu.get_piff_size(psf,xpos,ypos,cnum=piff_result['ccdnum'],verbose=verbose)
    piff_result['fwhm_map']={'bs':blocksize,'xpos':xpos,'ypos':ypos,'fwhm':fwhm_im*pixel_scale,
        'g2_amp':g2_im['amp'],'g2_x0':g2_im['x0'],'g2_y0':g2_im['y0'],'g2_sx':g2_im['sx'],'g2_sy':g2_im['sy'],'g2_the':g2_im['theta'],'g2_off':g2_im['off']}
    print("Elapsed time to sample PIFF model across CCD: {:.2f}".format(time.time()-t0))

#
#   Test 5:  residuals (not working), ngmix star and model statistics (and stellar distribution)
#
    t0=time.time()
    nstar=star_data['x'].size
    print("Performing NGMIX fits for stars and PIFF models for {:d} stars".format(nstar))
    s_dx=np.zeros(nstar,dtype=np.float64)
    s_dy=np.zeros(nstar,dtype=np.float64)
    s_e1=np.zeros(nstar,dtype=np.float64)
    s_e2=np.zeros(nstar,dtype=np.float64)
    s_T=np.zeros(nstar,dtype=np.float64)
    s_flux=np.zeros(nstar,dtype=np.float64)
    s_flag=np.zeros(nstar,dtype=np.int16)

    m_dx=np.zeros(nstar,dtype=np.float64)
    m_dy=np.zeros(nstar,dtype=np.float64)
    m_e1=np.zeros(nstar,dtype=np.float64)
    m_e2=np.zeros(nstar,dtype=np.float64)
    m_T=np.zeros(nstar,dtype=np.float64)
    m_flux=np.zeros(nstar,dtype=np.float64)
    m_flag=np.zeros(nstar,dtype=np.int16)

    print("Using image: {:s} for stellar fits".format(img))
    if (img[-2:] == "fz"):
        full_img=galsim.fits.read(img,hdu=1)
        full_wgt=galsim.fits.read(img,hdu=3)
    else:
        full_img=galsim.fits.read(img,hdu=0)
        full_wgt=galsim.fits.read(img,hdu=2)

    fwhm_guess=piff_result['fwhm']
    print("Using FWHM_guess={:.3f} corresponding to T={:.3f}".format(fwhm_guess, (fwhm_guess/ 2.35482)**2 * 2. ))

    if (seed is None):
        rng=None
    else:
        rng=np.random.RandomState(seed*piff_result['ccdnum']+piff_result['expnum'])

    if (verbose > 2):
        print("---------------------------------------------------------------------")
        print(" Star and model fits ")
        print(" {:5s} {:7s} {:7s} {:10s} {:10s} {:7s} {:3s} {:12s} ".format("#","dx","dy","e1","e2","T","flg","flux"))
    for i in range(star_data['x'].size):
        stamp_size=24

        x0=star_data['x'][i]
        y0=star_data['y'][i]
        b=galsim.BoundsI(int(x0)-stamp_size/2, int(x0)+stamp_size/2,
                         int(y0)-stamp_size/2, int(y0)+stamp_size/2)
        b= b & full_img.bounds

        img=full_img[b]
        wgt=full_wgt[b]
        mod=psf.draw(x=x0,y=y0,chipnum=star_data['chipnum'][i],image=img.copy())
#        print("FLUX by sum on model: ",np.sum(mod.array))
        mod*=star_data['flux'][i]
        mwgt=wgt.copy()

        s_dx[i],s_dy[i],s_e1[i],s_e2[i],s_T[i],s_flux[i],s_flag[i]=pqu.do_ngmix_fit(img,wgt,x0,y0,fwhm_guess,icnt=i,rng=rng,ftype='star ',verbose=verbose)
        m_dx[i],m_dy[i],m_e1[i],m_e2[i],m_T[i],m_flux[i],m_flag[i]=pqu.do_ngmix_fit(mod,mwgt,x0,y0,fwhm_guess,icnt=i,rng=rng,ftype='model',verbose=verbose)

#
#   NOTE: RA was coming from piffify in units of hours
#
    if (verbose > 2):
        print("---------------------------------------------------------------------")
    piff_result['star_data']={'x':star_data['x'],'y':star_data['y'],'ra':15.0*star_data['ra'],'dec':star_data['dec'],'flux':star_data['flux'],'snr':star_data['snr'],
                                's_e1':s_e1,'s_e2':s_e2,'s_T':s_T,'s_flag':s_flag,'m_e1':m_e1,'m_e2':m_e2,'m_T':m_T,'m_flag':m_flag}

    print("Rolling up NGMIX fit statistics")
    wsm=np.where(s_flag == 0)
    piff_result['star_nfit']=s_e1[wsm].size
    if (s_e1[wsm].size > 1):
        piff_result['star_e1_mean']=np.mean(s_e1[wsm])
        piff_result['star_e1_std']=np.std(s_e1[wsm])
        piff_result['star_e2_mean']=np.mean(s_e2[wsm])
        piff_result['star_e2_std']=np.std(s_e2[wsm])
        piff_result['star_t_mean']=np.mean(s_T[wsm])
        piff_result['star_t_std']=np.std(s_T[wsm])

        if (verbose > 0):
            print("  Stars (n,mean/stddev(e1,e2,T)): {nfit:7d}  {e1_m:9.6f} {e1_s:9.6f}  {e2_m:9.6f} {e2_s:9.6f}  {T_m:7.3f} {T_s:7.3f} ".format(
                nfit=piff_result['star_nfit'],
                e1_m=piff_result['star_e1_mean'],
                e2_m=piff_result['star_e2_mean'],
                T_m=piff_result['star_t_mean'],
                e1_s=piff_result['star_e1_std'],
                e2_s=piff_result['star_e2_std'],
                T_s=piff_result['star_t_std']))
    else:
        print("Insufficient stars with unflagged fits for statistics")
        piff_result['star_e1_mean']=-9999.
        piff_result['star_e1_std']=-9999.
        piff_result['star_e2_mean']=-9999.
        piff_result['star_e2_std']=-9999.
        piff_result['star_t_mean']=-9999.
        piff_result['star_t_std']=-9999.
    
    wsm=np.where(m_flag == 0)
    piff_result['model_nfit']=m_e1[wsm].size
    if (m_e1[wsm].size > 1):
        piff_result['model_e1_mean']=np.mean(m_e1[wsm])
        piff_result['model_e1_std']=np.std(m_e1[wsm])
        piff_result['model_e2_mean']=np.mean(m_e2[wsm])
        piff_result['model_e2_std']=np.std(m_e2[wsm])
        piff_result['model_t_mean']=np.mean(m_T[wsm])
        piff_result['model_t_std']=np.std(m_T[wsm])

        if (verbose > 0):
            print(" Models (n.mean/stddev(e1,d2,T)): {nfit:7d}  {e1_m:9.6f} {e1_s:9.6f}  {e2_m:9.6f} {e2_s:9.6f}  {T_m:7.3f} {T_s:7.3f} ".format(
                nfit=piff_result['model_nfit'],
                e1_m=piff_result['model_e1_mean'],
                e2_m=piff_result['model_e2_mean'],
                T_m=piff_result['model_t_mean'],
                e1_s=piff_result['model_e1_std'],
                e2_s=piff_result['model_e2_std'],
                T_s=piff_result['model_t_std']))
    else:
        print("Insufficient number of unflagged PIFF model fits for statistics")
        piff_result['model_e1_mean']=-9999.
        piff_result['model_e1_std']=-9999.
        piff_result['model_e2_mean']=-9999.
        piff_result['model_e2_std']=-9999.
        piff_result['model_t_mean']=-9999.
        piff_result['model_t_std']=-9999.

    print("Elapsed time to perform NGMIX star and model fits: {:.2f}".format(time.time()-t0))
#
#   Add HPIX indices
#   NOTE RA was coming from piffify in units of hours.
#

    print("Adding Healpix Indices w/ nsides={:s}".format(','.join(str(nside) for nside in nsides)))
    hpix_cols = ["hpix_{:d}".format(nside) for nside in nsides]
    phi = 15.0*star_data['ra']/180.*np.pi
    theta = (90. - star_data['dec'])/180.*np.pi
    for k in range(len(nsides)):
        nside = nsides[k]
        hpix_col = hpix_cols[k]
        pixs = healpy.ang2pix(nside, theta, phi, nest=True)
        piff_result['star_data'][hpix_col]=pixs

    piff_result['flag']=piff_flag

    return piff_result


#########################################
def examine_fit_outliers(data,sigout=5.0,verbose=0):
    """ Find the RMS of fit quantities (exposure based) and determine number of locations with significant outliers

        Inputs:
            data:   Dict (one per CCD) holding the fitting results
            sigout: Threshold for outlier significance to be considered for flagging"
        Output:
            data:   Dict with added information about outliers per CCD 
                    Note new dict entry 'outland' preserves outlier information for re-use (eg. when plotting)
    """
#
    g2_list=['g2_amp','g2_x0','g2_y0','g2_sx','g2_sy','g2_the','g2_off']

    print(" ")
    print("Examining model distribution across focal plane to look for outliers")
#
#   Loop over CCDs and accumulate statistics
#   Form the arrays that control rendering
#       Work out size needed
#       Create arrays
#       Fill
#
    AccSize=0    
    for Cat in data:
        AccSize+=data[Cat]['fwhm_map'][g2_list[0]].size
    AccData={}
    for Val in g2_list:
        AccData[Val]=np.zeros(AccSize,dtype='f8')
    ctr=0
    for Cat in data:
        ks=data[Cat]['fwhm_map'][g2_list[0]].size
        for Val in g2_list:
            key_array=data[Cat]['fwhm_map'][Val].reshape(ks)
            AccData[Val][ctr:ctr+ks]=key_array
        ctr+=ks
#
#   Use sigma-clipping to identify average and stddev for each quantity/value
#
    out_land={}
    nout={}
    for Val in g2_list:
        avg_val,med_val,std_val=pqu.medclip(AccData[Val],clipsig=5.0,verbose=0)
        srange=sigout*std_val
        out_land[Val]={'avg':avg_val,'med':med_val,'std':std_val,'min_out':avg_val-srange,'max_out':avg_val+srange}
        nout[Val]=np.zeros(62,dtype='i2')

    numtot=0
    for Cat in data:
        ccd=data[Cat]['ccdnum']
        num=data[Cat]['fwhm_map'][g2_list[0]].size
        numtot+=num
        for Val in g2_list:
            wsm=np.where(np.logical_or(data[Cat]['fwhm_map'][Val]>out_land[Val]['max_out'],data[Cat]['fwhm_map'][Val]<out_land[Val]['min_out']))
            nout[Val][ccd-1]=data[Cat]['fwhm_map'][Val][wsm].size
#
#       Save the number of outliers for centroids and widths
#
        data[Cat]['frac_cen_out']=(nout['g2_x0'][ccd-1]+nout['g2_y0'][ccd-1])/(2.*num)
        data[Cat]['frac_width_out']=(nout['g2_sx'][ccd-1]+nout['g2_sy'][ccd-1])/(2.*num)

#
#   Show a breakdown by CCD (if verbosity is turned on)
#
    if (verbose > 1):
        str_val=' ccd '
        for Val in g2_list:
            str_val=str_val+'{:s} '.format(Val)
        print(" {:s} ".format(str_val))
        print("---------------------------------------------------------")
        for i in range(62):
            str_val=' {:5d} '.format(i+1)
            for Val in g2_list:
                str_val=str_val+'{:5d} '.format(nout[Val][i])
            print(" {:s} ".format(str_val))
        str_val=' {:7d} '.format(numtot)
        print("---------------------------------------------------------")
        for Val in g2_list:
            str_val=str_val+'{:5d} '.format(np.sum(nout[Val]))
        print(" {:s} ".format(str_val))

    data['outland']=out_land

        
    return data


############################################################
############################################################

if __name__ == "__main__":
    t00=time.time()
    parser = argparse.ArgumentParser(description='Compare a set of DES refcats to VISTA')

    parser.add_argument('--cat',          action='store', type=str, default=None, required=True, help='Input catalog (list) to be examined')
    parser.add_argument('--img',          action='store', type=str, default=None, required=True, help='Input image (list) to be examined')
#    parser.add_argument('--debug'       , action='store_true', default=False,               help='Debug mode resticts code to work on a handful of objects')

    parser.add_argument('--out_thresh',   action='store', type=float, default=5.0,           help='Threshold (in sigma) to determine number of outlier (default=5.0)')
    parser.add_argument('--updateDB',     action='store_true', default=False,                help='Flag to allow DB update of QA information')
    parser.add_argument('--qa_table',     action='store', type=str, default='PIFF_MODEL_QA', help='DB table to update with general QA (default=PIFF_MODEL_QA)')
    parser.add_argument('--qa_star_table',action='store', type=str, default='PIFF_STAR_QA',  help='DB table to update with stellar/model measurements (default=PIFF_STAR_QA)')
    parser.add_argument('--qa_plot',      action='store', type=str, default=None,            help='Filename for QA plot showing model across the focal plane (default: no plot)')
    parser.add_argument('--seed',         action='store', type=int, default=None,            help='Seed for randoms in NGMIX (note uses SEED*CCDNUM+EXPNUM) default=None -> NGMIX chooses its own random... ie. non-reproducible results')
    parser.add_argument('-v','--verbose', action='store', type=int, default=0, help='Verbosity (defualt:0; currently values up to 2)')
    parser.add_argument('-T','--Timing',  action='store_true', default=False, help='If set timing information accompanies output')
    parser.add_argument('-s', '--section', action='store', type=str, default=None, help='section of .desservices file with connection info')
    parser.add_argument('-S', '--Schema', action='store', type=str, default=None, help='Schema')

    t00=time.time()
    args = parser.parse_args()
    if (args.verbose > 0):
        print("Args: {:}".format(args))

##########################################################
#   Handle simple args (verbose, Schema, bandlist)
#
    verbose=args.verbose
#
#   Obtain Schema (if user specified).
#
    if (args.Schema is None):
        dbSchema=""
    else:
        dbSchema="%s." % (args.Schema)

##########################################################
#   constants
#
#    pi=3.141592654
#    halfpi=pi/2.0
#    deg2rad=pi/180.0

    nsides=[64, 16384, 65536]
    t0=time.time()

#
#   Form list of PIFF data
#
    cat_list=[]
    if (os.path.isfile(args.cat)):
        if (args.cat[-4:] == "fits"):
#
#           File is a single FITS table
#
            cat_list.append(args.cat)
        else:
#
#           File is a list/group of tables
#
            f1=open(args.cat,'r')
            for line in f1:
                line=line.strip()
                columns=line.split(',')
                if (columns[0] != "#"):
                    cat_list.append(columns[0].strip())
            f1.close()
    else:
        print("Input catalog/list {:s}, not found.  Abort!".format(args.cat))
        exit(1)

#
#   Form list of IMG data
#
    img_list=[]
    if (os.path.isfile(args.img)):
        if ((args.img[-4:] == "fits")or(args.img[-7:] == "fits.fz")):
#
#           File is a single FITS table
#
            img_list.append(args.img)
        else:
#
#           File is a list/group of tables
#
            f1=open(args.img,'r')
            for line in f1:
                line=line.strip()
                columns=line.split(',')
                if (columns[0] != "#"):
                    img_list.append(columns[0].strip())
            f1.close()
    else:
        print("Input image/list {:s}, not found.  Abort!".format(args.img))
        exit(1)
#
#   Form paired dict: where dict[cat]=img:
#   Note this is currently assuming that lists are ordered.. could add checking by reading headers and pairing appropriately
#
    img_dict={}
    for i in range(len(cat_list)):
        img_dict[cat_list[i]]=img_list[i]
#        print(i,cat_list[i],img_list[i])
#
#    ExpCat={}
#    MetaCat={}
    qa_result={}
    for Cat in cat_list:
        print("##########################################################")
        print("Working on catalog: {:s} ".format(Cat))
        qa_result[Cat]=check_PIFF_data(Cat,img_dict[Cat],seed=args.seed,nsides=nsides,verbose=args.verbose)

    qa_result=examine_fit_outliers(qa_result,sigout=args.out_thresh,verbose=args.verbose)

#
#   New rollup values mean and standard deviation of T aggregated over all CCDs excepting CCD=31.
#
    exp_star_t_val=[]
    for Cat in cat_list:
        if ('ccdnum' in qa_result[Cat]):
            if (qa_result[Cat]['ccdnum']!=31):
                exp_star_t_val.append(qa_result[Cat]['star_t_mean'])
    est_val=np.array(exp_star_t_val)
    qa_result['exp_star_t_mean']=est_val.mean()
    qa_result['exp_star_t_std']=est_val.std()


    if (args.updateDB):
        try:
            desdmfile = os.environ["des_services"]
        except KeyError:
            desdmfile = None
        dbh = despydb.desdbi.DesDbi(desdmfile,args.section,retry=True)

        nval1,nval2=pqi.ingest_piff_qa(qa_result,args.qa_table,args.qa_star_table,dbh,dbSchema,verbose=verbose)
        dbh.close()
#
    if (args.qa_plot is not None):
        print("QA plot requested... generating it...")
        recode=pqp.plot_FP_QA(args.qa_plot,qa_result,verbose=0)

    print("Total elapsed time = {:.2f}".format(time.time()-t00))

    exit(0)
