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

#import piff
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
def find_uningested_PIFF_HSM(tag,nexp,dbh,dbSchema,verbose=0):

    Timing=True
    t0=time.time()
#
#   Form query and obtain PIFF model and hsm file infor
#
    query="""select av.val
        from {schema:s}proctag t, {schema:s}pfw_attempt_val av
        where t.tag='{ptag:s}'
            and t.pfw_attempt_id=av.pfw_attempt_id
            and av.key='expnum'
            and not exists (select 1 from gruendl.piff_hsm_star_qa q where q.pfw_attempt_id=t.pfw_attempt_id) 
            and rownum<{numexp:d}""".format(schema=dbSchema,ptag=tag,numexp=nexp)
#
    if (verbose > 0):
        if (verbose == 1):
            QueryLines=query.split('\n')
            QueryOneLine='sql = '
            for line in QueryLines:
                QueryOneLine=QueryOneLine+" "+line.strip()
            print("{:s}".format(QueryOneLine))
        if (verbose > 1):
            print("{:s}".format(query))

    curDB = dbh.cursor()
    prefetch=100000
    curDB.arraysize=int(prefetch)
    curDB.execute(query)
    desc = [d[0].lower() for d in curDB.description]

    ExpList=[]
    for row in curDB:
        rowd = dict(zip(desc, row))
#        if (int(rowd['val']) not in [598603,794822]):
        ExpList.append(int(rowd['val']))

    curDB.close()

    if (verbose>0):
        print("# Number of Uningested sets staged is {nval:d} ".format(nval=len(ExpList)))
    if (Timing):
        t1=time.time()
        print(" Query execution time: {:.2f}".format(t1-t0))

    return ExpList




###########################################
def get_PIFF_data(expnum,tag,dbh,dbSchema,verbose=0):

    Timing=True
    t0=time.time()
#
#   Form query and obtain PIFF model and hsm file infor
#
    query="""select oa.root||'/'||fai.path as fpath,
            m.filename,
            fai.compression,
            m2.filename as model_filename,
            m.pfw_attempt_id,
            m.expnum,
            m.ccdnum
        from {schema:s}proctag t, {schema:s}miscfile m, {schema:s}miscfile m2, {schema:s}file_archive_info fai, {schema:s}ops_archive oa
        where t.tag='{ptag:s}'
            and t.pfw_attempt_id=m.pfw_attempt_id
            and m.filetype='piff_model_stats'
            and m.expnum={exp:d}
            and m.filename=fai.filename
            and m.pfw_attempt_id=m2.pfw_attempt_id
            and m2.filetype='piff_model'
            and m2.ccdnum=m.ccdnum
            and fai.archive_name=oa.name""".format(schema=dbSchema,ptag=tag,exp=expnum)
#
    if (verbose > 0):
        if (verbose == 1):
            QueryLines=query.split('\n')
            QueryOneLine='sql = '
            for line in QueryLines:
                QueryOneLine=QueryOneLine+" "+line.strip()
            print("{:s}".format(QueryOneLine))
        if (verbose > 1):
            print("{:s}".format(query))

    curDB = dbh.cursor()
    prefetch=100000
    curDB.arraysize=int(prefetch)
    curDB.execute(query)
    desc = [d[0].lower() for d in curDB.description]

    fileDict={}
    for row in curDB:
        rowd = dict(zip(desc, row))
        fileDict[rowd['ccdnum']] = rowd

    curDB.close()

    if (verbose>0):
        print("# Number of PIFF model/HSM catalog pairs found is {nval:d} ".format(nval=len(fileDict)))
    if (Timing):
        t1=time.time()
        print(" Query execution time: {:.2f}".format(t1-t0))

    return fileDict



###########################################
def get_HSM_data(fname,verbose=0):

    """Perform a series of checks and analyses on PSF model from PIFF
        Inputs:
            fname:      PIFF model file
            verbose:    Controls the amount of information logged

        Returns:
            piff_result: Dict containing information garnered from the PIFF model.
                Low-level info includes: expnum, ccdnum
                
    """

    rfits=fitsio.FITS(fname,'r')
    h0=rfits[0].read_header()
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

    parser.add_argument('-t', '--tag',     action='store', type=str, default=None, required=True, help='Proctag')

    parser.add_argument('-e', '--expnum',  action='store', type=int, default=None, required=False, help='Expnum')
    parser.add_argument('-n', '--nexp',    action='store', type=int, default=None, required=False, help='Number of exposures to draw from the tag')

#    parser.add_argument('--cat',          action='store', type=str, default=None, required=True, help='Input catalog (list) to be examined')
#    parser.add_argument('--img',          action='store', type=str, default=None, required=True, help='Input image (list) to be examined')
#    parser.add_argument('--debug'       , action='store_true', default=False,               help='Debug mode resticts code to work on a handful of objects')

    parser.add_argument('--out_thresh',   action='store', type=float, default=5.0,           help='Threshold (in sigma) to determine number of outlier (default=5.0)')
    parser.add_argument('--updateDB',     action='store_true', default=False,                help='Flag to allow DB update of QA information')
    parser.add_argument('--qa_star_table',action='store', type=str, default='GRUENDL.PIFF_HSM_STAR_QA',  help='DB table to update with stellar/model measurements (default=PIFF_STAR_QA)')
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

    if (args.expnum is None)and(args.nexp is None):
        print("Must either give an exposure number (-e) or a number of exposures (-n) or both")
        print("Aborting!")
        exit(1)
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

    DBorder_STAR_QA=['FILENAME','STAR_NUMBER','MODEL_FILENAME','PFW_ATTEMPT_ID','EXPNUM','CCDNUM','U','V','X','Y','RA','DEC','FLUX','RESERVE',
                     'T_DATA','G1_DATA','G2_DATA','T_MODEL','G1_MODEL','G2_MODEL']

#    Form a list of potential PIFF sets to run on:


    if (args.nexp is not None):
        ExpList=find_uningested_PIFF_HSM(args.tag,args.nexp,dbh,dbSchema,verbose=verbose)
    else:
        ExpList=[]
    if (args.expnum is not None):
        ExpList.append(args.expnum)
    if (verbose > 2):
        print("Explist: {:}".format(ExpList))


    nexp_com=0
    nexp_pos=len(ExpList)
    for expnum in ExpList:
#
#       Form list of PIFF data
#
        fileDict=get_PIFF_data(expnum,args.tag,dbh,dbSchema,verbose=verbose)
#
#       read data
#       prepare to ingest
#
        new_data=[]
        for ccd in fileDict:
            if (fileDict[ccd]['compression'] is None):
                fnpath=fileDict[ccd]['fpath']+'/'+fileDict[ccd]['filename']
            else:
                fnpath=fileDict[ccd]['fpath']+'/'+fileDict[ccd]['filename']+fileDict[ccd]['compression']
            
            hsmData,hsmCols=get_HSM_data(fnpath,verbose=verbose)

            hsmDict={}
            for col in hsmCols:
                hsmDict[col.lower()]=hsmData[col]
#
#           Add columns not in HSM data file and transfer into list of lists for ingest
#
            for i in range(hsmData[hsmCols[0]].size):
                new_row=[]
                for col in DBorder_STAR_QA:
                    collow=col.lower()
                    if (collow in fileDict[ccd]):
                        new_row.append(fileDict[ccd][collow])
                    else:
                        if (collow in ['reserve']):
                            new_row.append(int(hsmDict[collow][i]))
                        elif (collow == 'star_number'):
                            new_row.append(i)
                        else:
                            new_row.append(hsmDict[collow][i])
                new_data.append(new_row)

        if (verbose > 2):
            print("Sample data for ingest")
            print(new_data[0])

        n2_insert=len(new_data)
        nexp_com=nexp_com+1
        if (args.updateDB):
#        dbh.insert_many(schema+StarTab,DBorder_STAR_QA,new_data)
            dbh.insert_many(args.qa_star_table,DBorder_STAR_QA,new_data)
            dbh.commit()
#            print("Commit {:d} rows to {:s}".format(n2_insert,schema+StarTab))
            print(" {:d} of {:d}: Commit {:d} rows to {:s}".format(nexp_com,nexp_pos,n2_insert,args.qa_star_table))
        else:
            print("Warning! Ingest skipped. Must have set --updateDB flag")
            print(" {:d} of {:d}: Would have commited {:d} rows to {:s}".format(nexp_com,nexp_pos,n2_insert,args.qa_star_table))

    print("Total elapsed time was: {:.2f}".format(time.time()-t00))

    exit(0)

