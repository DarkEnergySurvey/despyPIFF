#!/usr/bin/env python3
"""
Pull and ingest the PIFF HSM stats

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
#import despyPIFF.piff_qa_plot   as pqp
#import despyPIFF.piff_qa_utils  as pqu
#import despyPIFF.piff_qa_ingest as pqi

#from scipy.optimize import curve_fit

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from matplotlib.collections import PolyCollection
#from matplotlib.patches import Polygon

import piff
#import galsim
#from ngmix import priors, joint_prior
#import ngmix 
#import healpy

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
def get_PIFF_data(fname,verbose=0):

    Timing=True
    t0=time.time()
#
    piff_failed=False
    piff_flag=0
    piff_result={}
    if (verbose > 0):
        print("Reading PIFF model: {:s}".format(fname))
    rfits=fitsio.FITS(fname,'r')
    h0=rfits[0].read_header()

#    if ('band' in h0):
#        piff_result['band']=h0['BAND']
#    else:
#        print('Warning! BAND not found in primary HDU.  Will check for color in psfstars.')
#        piff_result['band']=None
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

    if ('pfw_attempt_id' in h0):
        piff_result['pfw_attempt_id']=int(h0['PFW_ATTEMPT_ID'])
    else:
        print("Warning! PFW_ATTEMPT_ID not present in primary HDU.  Using value of None.")
        piff_result['pfw_attempt_id']=None

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
#
#   NOTE: RA was coming from piffify in units of hours (rewriting as degrees)
#   Then preserve star_data in piff_result structure
#
    star_data['ra']*=15.

#
#   Backup search to arrive at a CCDNUM
#
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

    piff_result['flag']=piff_flag


    return piff_result,star_data,star_cols



###########################################
def get_HSM_data(fname,checkCCD=None,verbose=0):

    """Perform a series of checks and analyses on PSF model from PIFF
        Inputs:
            fname:      PIFF model file
            checkCCD:   Optional check that CCDNUM in header matches previously specified value
            verbose:    Controls the amount of information logged

        Returns:
            piff_result: Dict containing information garnered from the PIFF model.
                Low-level info includes: expnum, ccdnum
                
    """

    if (verbose > 0):
        print("Reading HSMstat file: {:s}".format(fname))
    rfits=fitsio.FITS(fname,'r')
    h0=rfits[0].read_header()
#
#   If requested atttempt to verify that CCDNUM matches that expected.
#
    if (not(checkCCD is None)):
        if ('CCDNUM' in h0):
            if (h0['CCDNUM']!=checkCCD):
                print("Warning: CCDNUM in HSM file primary HDU was: {:d}.  Code is attempting to match to {:}".format(h0['CCDNUM'],checkCCD))
        else:
            print("Warning: CCDNUM keyword not present in primary HDU.  Requested check not completed.")
#
#       FLAG = 2
#
    hsm_cols=rfits[1].get_colnames()
    hsm_data=rfits[1].read()
    nstar=len(hsm_data)
    if (verbose > 0):
        print("nstar {:d}: {:s}".format(nstar,fname))
    rfits.close()
#
    return hsm_data,hsm_cols


############################################################
############################################################

if __name__ == "__main__":
    t00=time.time()
    parser = argparse.ArgumentParser(description='Compare a set of DES refcats to VISTA')

    parser.add_argument('--model',        action='store', type=str, default=None, required=True, help='Input PIFF model file (list) to be examined')
    parser.add_argument('--stat',         action='store', type=str, default=None, required=True, help='Input PIFF HSM stat file (list) to be examined')

    parser.add_argument('--pfw_attempt_id', action='store', type=int, default=None,            help='PFW_ATTEMPT_ID (optional, default=None)')
    parser.add_argument('--updateDB',     action='store_true', default=False,                help='Flag to allow DB update of QA information')

    parser.add_argument('--qa_table',     action='store', type=str, default='PIFF_HSM_MODEL_QA', help='DB table to update with general model QA          (default=PIFF_HSM_MODEL_QA)')
    parser.add_argument('--qa_star_table',action='store', type=str, default='PIFF_HSM_STAR_QA',  help='DB table to update with object/model measurements (default=PIFF_HSM_STAR_QA)')

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

    try:
        desdmfile = os.environ["des_services"]
    except KeyError:
        desdmfile = None
    dbh = despydb.desdbi.DesDbi(desdmfile,args.section,retry=True)

##########################################################
#   constants
#
#    pi=3.141592654
#    halfpi=pi/2.0
#    deg2rad=pi/180.0

    nsides=[64, 16384, 65536]
    t0=time.time()

    DBorder_MODEL_QA=['FILENAME','PFW_ATTEMPT_ID','EXPNUM','CCDNUM','NSTAR','NREMOVED','CHISQ','DOF','FLAG',
                     'STAR_T_MEAN','STAR_T_STD','FWHM_CEN','EXP_STAR_T_MEAN','EXP_STAR_T_STD']

    DBorder_STAR_QA=['MODEL_FILENAME','HSMSTAT_FILENAME','STAR_NUMBER','PFW_ATTEMPT_ID','EXPNUM','CCDNUM',
                     'U','V','X','Y','RA','DEC','FLUX','SNR','IS_RESERVE','FLAG_MODEL','FLAG_TRUTH',
                     'T_DATA','G1_DATA','G2_DATA','T_MODEL','G1_MODEL','G2_MODEL',
                     'GAIA_STAR','GAIA_SOURCE_ID','PHOT_OBJ','COADD_OBJECT_ID','VHS_OBJ','VHS_SOURCEID',
                     'BDF_T','EXT_MASH','G_MAG','R_MAG','I_MAG','Z_MAG','K_MAG','GI_COLOR','IZ_COLOR','FLAG_COLOR']

#   Form a list of potential PIFF models to ingest QA from:

    model_list=[]
    if (os.path.isfile(args.model)):
        if (args.model[-4:] == "fits"):
#
#           File is a single FITS table
#
            model_list.append(args.model)
        else:
#
#           File is a list/group of tables
#
            f1=open(args.model,'r')
            for line in f1:
                line=line.strip()
                columns=line.split(',')
                if (columns[0] != "#"):
                    model_list.append(columns[0].strip())
            f1.close()
    else:
        print("Input model/list {:s}, not found.  Abort!".format(args.model))
        exit(1)

#
#   Form list of PIFF HSM stat data
#

    stat_list=[]
    if (os.path.isfile(args.stat)):
        if (args.stat[-4:] == "fits"):
#
#           File is a single FITS table
#
            stat_list.append(args.stat)
        else:
#
#           File is a list/group of tables
#
            f1=open(args.stat,'r')
            for line in f1:
                line=line.strip()
                columns=line.split(',')
                if (columns[0] != "#"):
                    stat_list.append(columns[0].strip())
            f1.close()
    else:
        print("Input HSM stat file/list {:s}, not found.  Abort!".format(args.stat))
        exit(1)

    if (len(model_list)!=len(stat_list)):
        print("Found {:d} models but {:d} HSM stat files!  Aborting!".format(len(model_list),len(stat_list)))
        exit(1)
#
    qa_result={}    
    for icat, CatMod in enumerate(model_list):
        print("###################")
        print("Working on entry #{:d}:  model={:s}   HSMstat={:s}  ".format(icat,CatMod,stat_list[icat]))
        qa_mod,star_data,star_cols=get_PIFF_data(CatMod,verbose=verbose)
        hsmData,hsmCols=get_HSM_data(stat_list[icat],checkCCD=qa_mod['ccdnum'],verbose=verbose)
#
#       Check that # of objects found in PIFF star catalog and HSM stat catalog match
#
        nstar_data=star_data[star_cols[0]].size
        nhsm_stats=hsmData[hsmCols[0]].size
        if (nstar_data != nhsm_stats):
            print("Number of PIFF stars ({:d}) and HSMstat objects ({:d}) do not match!".format(nstar_data,nhsm_stats))
            print("Aborting")
            exit(1)
#
#       Make sanity check that any difference between RA-Dec positions in Model and HSMstat catalogs can be attributed to machine accuracy
#           rather than inconsistent catalog ordering.
#       Current limit of 7.5e-5 deg is eqivalent to one DECam pixel.
#
        max_dra=np.max(np.abs(star_data['ra']-hsmData['ra']))
        max_ddec=np.max(np.abs(star_data['dec']-hsmData['dec']))
        if (verbose >0):
            print("Check of maximum difference in RA,Dec between PIFF(stars) and HSMstat catalogs found:")
            print("   max(dra),max(ddec) = {:},{:} ".format(max_dra,max_ddec))
        if ((max_dra > 7.5e-5)or(max_ddec > 7.5e-5)):
            print("Warning: Catalog(s) PIFF-stars and HSMstat may not be given in the same order!")
            exit(1)
#
#       Populate the results section
#
        qa_result[icat]={'gen':{},'model':{},'stars':{},'starcol':[]}
        qa_result[icat]['gen']['FILENAME']=CatMod
        qa_result[icat]['gen']['MODEL_FILENAME']=CatMod
        qa_result[icat]['gen']['HSMSTAT_FILENAME']=stat_list[icat]
        qa_result[icat]['gen']['EXPNUM']=qa_mod['expnum']
        qa_result[icat]['gen']['CCDNUM']=qa_mod['ccdnum']
#       Some logic to pull PFW_ATTEMPT_ID from header if it exists or from command line if specified
#       Note command line trumps header...
        if (args.pfw_attempt_id is None):
            if ('pfw_attempt_id' in qa_mod):
                qa_result[icat]['gen']['PFW_ATTEMPT_ID']=qa_mod['pfw_attempt_id']
            else:
                qa_result[icat]['gen']['PFW_ATTEMPT_ID']=None
        else:
            qa_result[icat]['gen']['PFW_ATTEMPT_ID']=args.pfw_attempt_id
        qa_result[icat]['model']['NSTAR']=qa_mod['nstar']
        qa_result[icat]['model']['NREMOVED']=qa_mod['nremoved']
        qa_result[icat]['model']['CHISQ']=qa_mod['chisq']
        qa_result[icat]['model']['DOF']=qa_mod['dof']
        qa_result[icat]['model']['FLAG']=qa_mod['flag']

        for col in DBorder_STAR_QA:
            if (col.lower() in star_cols):
                qa_result[icat]['stars'][col]=star_data[col.lower()]
                qa_result[icat]['starcol'].append(col)
            elif (col in star_cols):
                qa_result[icat]['stars'][col]=star_data[col]
                qa_result[icat]['starcol'].append(col)
            else:
                if (col.lower() in hsmCols):
                    qa_result[icat]['stars'][col]=hsmData[col.lower()]
                    qa_result[icat]['starcol'].append(col)
                elif (col in hsmCols):
                    qa_result[icat]['stars'][col]=hsmData[col]
                    qa_result[icat]['starcol'].append(col)
                elif (col == "T_DATA"):
                    qa_result[icat]['stars'][col]=hsmData['T_data']
                    qa_result[icat]['starcol'].append(col)
                elif (col == "T_MODEL"):
                    qa_result[icat]['stars'][col]=hsmData['T_model']
                    qa_result[icat]['starcol'].append(col)

#        for k1 in qa_result:
#            for k2 in qa_result[k1]:
#                if (k2 == "starcol"):
#                    print(k1,k2,qa_result[k1][k2])
#                else:
#                    for k3 in qa_result[k1][k2]:
#                        print(k1,k2,k3,qa_result[k1][k2][k3])

#
#   Calculate roll-up statistics
#

    if (verbose > 0):
        print("###################")
        print("Calculating rollup quantities for the exposure.")
    t_mean_list=[]
    for icat in qa_result:
        if ('T_DATA' in qa_result[icat]['stars']):
#           Remove objects that are supect prior to calculating rollup values
            wsm=np.where(np.logical_and(qa_result[icat]['stars']['FLAG_TRUTH']==0,qa_result[icat]['stars']['FLAG_MODEL']==0))
            if (qa_result[icat]['stars']['T_DATA'][wsm].size > 2):
                qa_result[icat]['model']['STAR_T_MEAN']=np.mean(qa_result[icat]['stars']['T_DATA'][wsm])
                qa_result[icat]['model']['STAR_T_STD']=np.std(qa_result[icat]['stars']['T_DATA'][wsm])
                qa_result[icat]['model']['FWHM_CEN']=2.3548200450309493*np.sqrt(qa_result[icat]['model']['STAR_T_MEAN']/2.)
#               Do NOT include CCD=31 in stats
                if (qa_result[icat]['gen']['CCDNUM']!=31):
                    t_mean_list.append(qa_result[icat]['model']['STAR_T_MEAN'])
            else:
                qa_result[icat]['model']['STAR_T_MEAN']=None
                qa_result[icat]['model']['STAR_T_STD']=None
    
    if (len(t_mean_list) > 2):
        exp_star_t_mean=np.mean(np.array(t_mean_list))
        exp_star_t_std=np.std(np.array(t_mean_list))
    else:
        exp_star_t_mean=None
        exp_star_t_std=None

    for icat in qa_result:
        qa_result[icat]['model']['EXP_STAR_T_MEAN']=exp_star_t_mean
        qa_result[icat]['model']['EXP_STAR_T_STD']=exp_star_t_std

#
#   Data typing from numpy to Oracle was fragile but seems to be more bullet proof if ndarray's are switched for lists
#
    print("Swapping out np.ndarrays for ingestion ease")
    for icat in qa_result:
        for col in qa_result[icat]['stars']:
            if (np.issubdtype(qa_result[icat]['stars'][col].dtype,np.integer)):
                qa_result[icat]['stars'][col]=qa_result[icat]['stars'][col].tolist()
            elif (np.issubdtype(qa_result[icat]['stars'][col].dtype,np.floating)):
                qa_result[icat]['stars'][col]=qa_result[icat]['stars'][col].tolist()
            elif (np.issubdtype(qa_result[icat]['stars'][col].dtype,np.bool)):
                tmp_arr=np.zeros(qa_result[icat]['stars'][col].size,dtype=np.dtype('>i4'))
                wsm=np.where(qa_result[icat]['stars'][col])
                tmp_arr[wsm]=1
                qa_result[icat]['stars'][col]=tmp_arr.tolist()
            else:
                print(col,qa_result[icat]['stars'][col].dtype)

#
#   Prepare to ingest
#   Columns are taken from the 'gen' columns not in HSM data file and transfer into list of lists for ingest
#   First the STARS table
#
    if (verbose > 0):
        print("Preparing to ingest data.")
        
    new_data=[]
    for icat in qa_result:
        num_stars=len(qa_result[icat]['stars'][qa_result[icat]['starcol'][0]])
        for i in range(num_stars):
            new_row=[]
            for col in DBorder_STAR_QA:
                if (col in qa_result[icat]['gen']):
                    new_row.append(qa_result[icat]['gen'][col])
                elif (col == 'STAR_NUMBER'):
                    new_row.append(i)
                elif (col in qa_result[icat]['stars']):
                    new_row.append(qa_result[icat]['stars'][col][i])
                else:
                    new_row.append(None)

            new_data.append(new_row)

    if (verbose > 2):
        print("Sample data for ingest")
        print(new_data[0])

    n2_insert=len(new_data)
    if (args.updateDB):
        dbh.insert_many(args.qa_star_table,DBorder_STAR_QA,new_data)
        dbh.commit()
        print(" Commit {:d} rows to {:s}".format(n2_insert,args.qa_star_table))
    else:
        print("Warning! Ingest skipped. Must have set --updateDB flag")
        print(" Would have commited {:d} rows to {:s}".format(n2_insert,args.qa_star_table))

#
#   Now the MODEL table
#
    new_data=[]
    for icat in qa_result:
        new_row=[]
        for col in DBorder_MODEL_QA:
            if (col in qa_result[icat]['gen']):
                new_row.append(qa_result[icat]['gen'][col])
            elif (col in qa_result[icat]['model']):
                new_row.append(qa_result[icat]['model'][col])
            else:
                new_row.append(None)

        new_data.append(new_row)

    if (verbose > 2):
        print("Sample data for ingest")
        print(new_data[0])

    n2_insert=len(new_data)
    if (args.updateDB):
        dbh.insert_many(args.qa_table,DBorder_MODEL_QA,new_data)
        dbh.commit()
        print(" Commit {:d} rows to {:s}".format(n2_insert,args.qa_table))
    else:
        print("Warning! Ingest skipped. Must have set --updateDB flag")
        print(" Would have commited {:d} rows to {:s}".format(n2_insert,args.qa_table))

#
#   Finished!!!
#

    print("Total elapsed time was: {:.2f}".format(time.time()-t00))

    exit(0)

