#!/usr/bin/env python

# $Id$
# $Rev::                                  $:  # Revision of last commit.
# $LastChangedBy::                        $:  # Author of last commit.
# $LastChangedDate::                      $:  # Date of last commit.

"""Subselection of a DES (FINALCUT) catalog for use with PIFF
"""

from __future__ import print_function
import argparse
import os
import re
import time
#import datetime
import sys
import numpy as np
#from numpy import ma
import fitsio

import pandas as pd
import despydb
from despyPIFF import subselect_QA as qa

from astropy.coordinates import SkyCoord
from astropy import units as u

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon

###########################################
def read_data(fname,ldac=False,hdu=1,verbose=0):

    rfits=fitsio.FITS(fname,'r')
    if (ldac):
        ldachead=rfits[hdu-1].read()
#        ldachead_cols=rfits[hdu-1].get_colnames()
#        print(ldachead)
#        print(ldachead_cols)
    cols=rfits[hdu].get_colnames()
    if (verbose > 0):
        print("Cols: ",cols)
    data = rfits[hdu].read()
    rfits.close()

    if (ldac):
        return data,cols,ldachead
    else:
        return data,cols


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
  

######################################################################################
def form_ExpCat(grp_list,DESColDict,verbose=0):

    """ Reads a set of FITS catalogs and organizes data for further use

        Inputs:
            grp_list:   A list of filepaths that are reference catalogs
            DESColDict: Dictionary of columns that will be recorded

        Returns:
            ExpCat:      Dict of Cats
            MetaCat:    Dict containing MetaData for the Catalogs
            RaDecRange: Summary of area covered by the catalogs
    """

    ExpCat={}
    MetaCat={}
    rd_range={}
    rd_range['ra']=[]
    rd_range['dec']=[]
    rd_range['crossra0']=False
    for Cat in grp_list:
        if (verbose > 2):
            print("Working on catalog: {:s} ".format(Cat))

        NewCat,NewCatCols,ldhead=read_data(Cat,ldac=True,hdu=2,verbose=0)

        ImgKeyDict={'EXPNUM':'i4','CCDNUM':'i4','BAND':'a5'}
        MetaCat[Cat]={}
        for rec in ldhead['Field Header Card'][0]:
            for keywd in ImgKeyDict:
                x=re.match(keywd,rec)
                if (x is not None):
                    m1=re.search("=",rec)
                    i1=m1.end()
                    m2=re.search("/",rec)
                    if (m2 is None):
                        i2=79
                    else:
                        i2=m2.start()
                    if (ImgKeyDict[keywd] in ['i2','i4','i8']):
                        MetaCat[Cat][keywd]=int(rec[i1:i2].strip())
                    elif (ImgKeyDict[keywd] in ['f4','f8']):
                        MetaCat[Cat][keywd]=float(rec[i1:i2].strip())
                    else:
                        fstring=rec[i1:i2]
                        fstring=re.sub('\'','',fstring)
                        MetaCat[Cat][keywd]=fstring.strip()
                    

        ExpCat[Cat]={}
#       Remove things that will blow-up a signal-to-noise calculation
        wsm=np.where(NewCat['FLUXERR_AUTO']>0.0)
        for key in DESColDict:
            ExpCat[Cat][key]=NewCat[key][wsm]

#       Info later used to obtain the RA/Dec range   
        ra1=np.min(ExpCat[Cat]['ALPHAWIN_J2000'])
        ra2=np.max(ExpCat[Cat]['ALPHAWIN_J2000'])
        if (ra2-ra1 > 180.):
            rd_range['crossra0']=True
        dec1=np.min(ExpCat[Cat]['DELTAWIN_J2000'])
        dec2=np.max(ExpCat[Cat]['DELTAWIN_J2000'])

        rd_range['ra'].append(ra1)
        rd_range['ra'].append(ra2)
        rd_range['dec'].append(dec1)
        rd_range['dec'].append(dec2)

        if (verbose > 3):
            print("   Input Cat had {:d} objects ".format(NewCat['ALPHAWIN_J2000'].size))
            print("   RA Range: {:9.5f} {:9.5f} ".format(ra1,ra2))
            print("  Dec Range: {:9.5f} {:9.5f} ".format(dec1,dec2))

#    print(MetaCat)

    RaDecRange={}
    rd_range['ra']=np.array(rd_range['ra'])
    rd_range['dec']=np.array(rd_range['dec'])
    if (rd_range['crossra0']):
        RaDecRange['crossra0']=True
        wsm1=np.where(rd_range['ra']>180.0)
        wsm2=np.where(rd_range['ra']<180.0)
        RaDecRange['ra1']=np.min(rd_range['ra'][wsm1])
        RaDecRange['ra2']=np.max(rd_range['ra'][wsm2])
    else:
        RaDecRange['crossra0']=False
        RaDecRange['ra1']=np.min(rd_range['ra'])
        RaDecRange['ra2']=np.max(rd_range['ra'])
    RaDecRange['dec1']=np.min(rd_range['dec'])
    RaDecRange['dec2']=np.max(rd_range['dec'])

    if (args.verbose > 0):
        print(" Read {:d} catalogs ".format(len(grp_list)))
        print(" Overall RA-Dec coverage is: ")
        print("   RA Range: {:9.5f} {:9.5f} ".format(RaDecRange['ra1'],RaDecRange['ra2']))
        print("  Dec Range: {:9.5f} {:9.5f} ".format(RaDecRange['dec1'],RaDecRange['dec2']))
        if (RaDecRange['crossra0']):
            print("  Note that RA range crosses RA=0h")

    return ExpCat,MetaCat,RaDecRange


######################################################################################
def get_GAIADR2_objects(radec_box,dbh,dbSchema,Timing=False,verbose=0):

    """ Query code to obtain list of images that overlap another

        Inputs:
            radec_box: Dict with range to search in RA,Dec (with flag to handle case where RA range crosses RA=0h
            dbh:       Database connection to be used
            dbSchema:  Schema over which queries will occur.
            verbose:   Integer setting level of verbosity when running.

        Returns:
            CatDict: Resulting Image dictionary
    """

    t0=time.time()

    if (radec_box['crossra0']):
#
#       Form Query for case where RA ranges crosses RA=0h (not very good at poles)
#
        query="""select g.source_id,g.ra,g.dec
            from des_admin.gaia_dr2 g
            where (g.ra < {r2:.6f} or g.ra > {r1:.6f})
                and g.dec between {d1:.6f} and {d2:.6f}""".format(
        r1=radec_box['ra1'],
        r2=radec_box['ra2'],
        d1=radec_box['dec1'],
        d2=radec_box['dec2'])
    else:
#
#       Form query for normal workhorse case 
#
        query="""select g.source_id,g.ra,g.dec
            from des_admin.gaia_dr2 g
            where g.ra between {r1:.6f} and {r2:.6f}
                and g.dec between {d1:.6f} and {d2:.6f}""".format(
        r1=radec_box['ra1'],
        r2=radec_box['ra2'],
        d1=radec_box['dec1'],
        d2=radec_box['dec2'])
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
#
#   Establish a DB cursor
#
    curDB = dbh.cursor()
#    curDB.execute(query)
#    desc = [d[0].lower() for d in curDB.description]

    prefetch=100000
    curDB.arraysize=int(prefetch)
    curDB.execute(query)
#    header=[d[0].lower() for d in curDB.description]
    header=[d[0].upper() for d in curDB.description]
    cat_data=pd.DataFrame(curDB.fetchall())

    CatDict={}
    if (cat_data.empty):
        print("# No values returned from query of {tval:s} ".format(tval="GAIA_DR2"))
        for val in header:
            CatDict[val]=np.array([])
    else:
        cat_data.columns=header
        for val in header:
            CatDict[val]=np.array(cat_data[val])
    curDB.close()

    if (verbose>0):
        print("# Number of GAIA objects found is {nval:d} ".format(nval=CatDict[header[0]].size))
    if (Timing):
        t1=time.time()
        print(" Query execution time: {:.2f}".format(t1-t0))

    return CatDict,header


######################################################################################
def get_VHS_objects(radec_box,dbh,dbSchema,Timing=False,verbose=0):

    """ Query code to obtain list of likely stars from VHS

        Inputs:
            radec_box: Dict with range to search in RA,Dec (with flag to handle case where RA range crosses RA=0h
            dbh:       Database connection to be used
            dbSchema:  Schema over which queries will occur.
            verbose:   Integer setting level of verbosity when running.

        Returns:
            CatDict: Resulting Image dictionary
    """

    t0=time.time()

    if (radec_box['crossra0']):
#
#       Cross RA=0 (not very good at poles)
#
        query="""select v.ra,v.dec,v.vhs_class 
            from bechtol.y3a2_vhs_des_class v
            where (v.ra < {r2:.6f} or v.ra > {r1:.6f})
                and v.dec between {d1:.6f} and {d2:.6f}
                and v.vhs_class=0""".format(
        r1=radec_box['ra1'],
        r2=radec_box['ra2'],
        d1=radec_box['dec1'],
        d2=radec_box['dec2'])
    else:
#
#       Normal workhorse case 
#
        query="""select v.ra,v.dec,v.vhs_class 
            from bechtol.y3a2_vhs_des_class v
            where v.ra between {r1:.6f} and {r2:.6f}
                and v.dec between {d1:.6f} and {d2:.6f} 
                and v.vhs_class=0""".format(
        r1=radec_box['ra1'],
        r2=radec_box['ra2'],
        d1=radec_box['dec1'],
        d2=radec_box['dec2'])
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
#
#   Establish a DB connection
#
    curDB = dbh.cursor()
#    curDB.execute(query)
#    desc = [d[0].lower() for d in curDB.description]

    prefetch=100000
    curDB.arraysize=int(prefetch)
    curDB.execute(query)
#    header=[d[0].lower() for d in curDB.description]
    header=[d[0].upper() for d in curDB.description]
    cat_data=pd.DataFrame(curDB.fetchall())

    CatDict={}
    if (cat_data.empty):
        print("# No values returned from query of {tval:s} ".format(tval="Y3A2_VHS_DES_CLASS"))
        for val in header:
            CatDict[val]=np.array([])
    else:
        cat_data.columns=header
        for val in header:
            CatDict[val]=np.array(cat_data[val])
#            if (val in ColDict):
#                CatDict[val]=np.array(cat_data[val],dtype=ColDict[val])
#            else:
#                CatDict[val]=np.array(cat_data[val])
    curDB.close()

    if (verbose>0):
        print("# Number of VHS objects found is {nval:d} ".format(nval=CatDict[header[0]].size))
    if (Timing):
        t1=time.time()
        print(" Query execution time: {:.2f}".format(t1-t0))

    return CatDict,header


############################################################
############################################################

if __name__ == "__main__":
    t00=time.time()
    parser = argparse.ArgumentParser(description='Compare a set of DES refcats to VISTA')

    parser.add_argument('--cat',          action='store', type=str, default=None, required=True,     help='Input catalog to be compared to reference')
    parser.add_argument('--gcat',         action='store', type=str, default=None, required=False,    help='Alternative Input catalog from GAIA (default is to use DB)')
    parser.add_argument('--odir',         action='store', type=str, default='cat', required=True,    help='Output directory name')
    parser.add_argument('--reqnum',       action='store', type=int, default=0,    required=True,     help='Processing Request number (for output file pattern)')
    parser.add_argument('--attnum',       action='store', type=int, default=0,    required=True,     help='Processing Attempt number (for output file pattern)')
    parser.add_argument('--suffix',       action='store', type=str, default='piffcat', required=False,  help='Output filename pattern suffix (default=piffcat)')
    parser.add_argument('--useDB',        action='store_true', default=False, help='Flag to obtain reference data from DB rather than files')
    parser.add_argument('--checkVHS',     action='store_true', default=False, help='Flag to also perform comparison using VHS selected stars')
    parser.add_argument('--sncut',        action='store', type=float, default=20., help='Apply a S/N cut at XX-sigma on the input catalog (default=None)')
    parser.add_argument('--remove_top_sn',action='store', type=float, default=1.2, help='Remove high signal-to-noise objects to avoid brighter-fatter biases (default=1.2 mag)')
    parser.add_argument('--top_sn_thresh',action='store', type=float, default=1.e3, help='Signal-to-noise threshold to be exceeded before bright objects are removed (default=1.e3)')
    parser.add_argument('--min_sn_add',   action='store', type=float, default=20., help='Signal-to-noise threshold to be exceeded before appropriately sized objects are added (default=20.)')
    parser.add_argument('--qa_select',    action='store', type=str, default=None, help='File name for selection QA plots')
    parser.add_argument('--qa_dist',      action='store', type=str, default=None, help='File name for distribution QA plots')

    parser.add_argument('-T','--Timing',  action='store_true', default=False, help='If set timing information accompanies output')
#    parser.add_argument('--debug'       , action='store_true', default=False, help='Debug mode resticts code to work on a handful of objects')
    parser.add_argument('-v','--verbose', action='store', type=int, default=0, help='Verbosity (defualt:0; currently values up to 2)')
    parser.add_argument('-s', '--section', action='store', type=str, default=None, help='section of .desservices file with connection info')
    parser.add_argument('-S', '--Schema', action='store', type=str, default=None, help='Schema')

    args = parser.parse_args()
    if (args.verbose > 0):
        print("Args: {:s}".format(args))

##########################################################
#   Handle simple args (verbose, Schema, bandlist)
#
    verbose=args.verbose
    checkVHS=args.checkVHS
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
    t0=time.time()

#
#   Get DES data
#
    DESColList=['ALPHAWIN_J2000','DELTAWIN_J2000','ERRAWIN_WORLD','ERRBWIN_WORLD','ERRTHETAWIN_J2000',
                'XWIN_IMAGE','YWIN_IMAGE','BACKGROUND','MAG_AUTO','MAGERR_AUTO',
                'FLUX_AUTO','FLUXERR_AUTO','FLUX_RADIUS','FLAGS','SPREAD_MODEL','SPREADERR_MODEL','IMAFLAGS_ISO']
    DESColDict={
        'ALPHAWIN_J2000':'f8',
        'DELTAWIN_J2000':'f8',
        'ERRAWIN_WORLD':'f8',
        'ERRBWIN_WORLD':'f8',
        'ERRTHETAWIN_J2000':'f8',
        'XWIN_IMAGE':'f8',
        'YWIN_IMAGE':'f8',
        'FLUX_RADIUS':'f8',
        'BACKGROUND':'f8',
        'MAG_AUTO':'f8',
        'MAGERR_AUTO':'f8',
        'FLUX_AUTO':'f8',
        'FLUXERR_AUTO':'f8',
        'FLAGS':'i2',
        'IMAFLAGS_ISO':'i2',
        'SPREAD_MODEL':'f8',
        'SPREADERR_MODEL':'f8'}

#
#   Form list of catalogs that are going to be worked on/over
#
    grp_list=[]
    if (os.path.isfile(args.cat)):
        if (args.cat[-4:] == "fits"):
#           File is a single FITS table
            grp_list.append(args.cat)
        else:
#           File is a list/group of tables
            f1=open(args.cat,'r')
            for line in f1:
                line=line.strip()
                columns=line.split(',')
                if (columns[0] != "#"):
                    grp_list.append(columns[0].strip())
            f1.close()
    else:
        print("Input catalog {:s}, not found.  Abort!".format(args.cat))
        exit(1)
#
#   Read DES (single-epoch) catalogs
#
    ExpCat,MetaCat,RaDecRange=form_ExpCat(grp_list,DESColDict,verbose=0)


#   Setup DB connection if needed
    if ((args.useDB)or(args.checkVHS)):
        dbh = despydb.desdbi.DesDbi(None,args.section,retry=True)
#
#   Get GAIA data
#
    if (args.useDB):
#
#       Setup for database interactions (through despydb)
#
        GaiaCat,GaiaCatCols=get_GAIADR2_objects(RaDecRange,dbh,dbSchema,Timing=True,verbose=2)
    else:
#
#       Alternatively you can feed it a FITS table
#
        if (args.gcat is None):
            GaiaCat=None
            print("NO GAIA catalog given... skipping.")
        else:
            if (os.path.isfile(args.gcat)):
                data,cols=read_data(args.gcat,hdu=1,verbose=0)
                GaiaCatCols=[]
                GaiaCat={}                
                for col in cols:
                    GaiaCatCols.append(col.upper())
                    GaiaCat[col.upper()]=data[col]
                print("Number of GAIA objects (total): {:d}".format(GaiaCat[GaiaCatCols[0]].size))
                
            else:
                print("Input catalog {:s}, not found.  Abort!".format(args.gcat))
                exit(1)

#
#   Get VHS data for comparison (if requested)
#
    if (args.checkVHS):
        VHSCat,VHSCatCols=get_VHS_objects(RaDecRange,dbh,dbSchema,Timing=True,verbose=2)
    else:
        VHSCat={}
        VHSCatCols=[]

    if ((args.useDB)or(args.checkVHS)):
        dbh.close()

    if (args.checkVHS):
        if (VHSCat[VHSCatCols[0]].size < 2):
            checkVHS=False

#
#   Pre-Prepare GAIA data (and VHS data) for matching
#

    c2=SkyCoord(ra=GaiaCat['RA']*u.degree,dec=GaiaCat['DEC']*u.degree)
    if (checkVHS):
        c3=SkyCoord(ra=VHSCat['RA']*u.degree,dec=VHSCat['DEC']*u.degree)
  
#
#   Main body catalog comparison and source selection
# 

    diag_cat={}
    kept_cat={} 
    kept_cat_VHS={}
    for Cat in grp_list:
        if (Cat not in MetaCat):
            MetaCat[Cat]={}
        print("Working on catalog: {:s}".format(Cat))
        nobj0=ExpCat[Cat][DESColList[0]].size

        c1=SkyCoord(ra=ExpCat[Cat]['ALPHAWIN_J2000']*u.degree,dec=ExpCat[Cat]['DELTAWIN_J2000']*u.degree)
        idx2, d2d, d3d = c1.match_to_catalog_sky(c2)
        idx1=np.arange(ExpCat[Cat]['ALPHAWIN_J2000'].size)
        wsm=np.where(d2d.arcsecond<0.50)
        MetaCat[Cat]['nm_GAIA']=idx1[wsm].size

        diag_cat[Cat]={}
        kept_cat[Cat]={}
        for key in DESColDict:
            kept_cat[Cat][key]=ExpCat[Cat][key][idx1[wsm]]
            diag_cat[Cat][key]=np.copy(ExpCat[Cat][key])
#
#       Remove FLAGGED and LOW signal-to-noise objects
#       For the DIAGNOSTIC just remove flagged....
#
        wsm=np.where(np.logical_and(np.logical_and(kept_cat[Cat]['FLAGS']==0,kept_cat[Cat]['IMAFLAGS_ISO']==0),
                                    kept_cat[Cat]['FLUX_AUTO']/kept_cat[Cat]['FLUXERR_AUTO']>args.sncut))
        dwsm=np.where(np.logical_and(diag_cat[Cat]['FLAGS']==0,diag_cat[Cat]['IMAFLAGS_ISO']==0))
        for key in DESColDict:
            kept_cat[Cat][key]=kept_cat[Cat][key][wsm]
            diag_cat[Cat][key]=diag_cat[Cat][key][dwsm]
        MetaCat[Cat]['nm_GAIA_cut0']=kept_cat[Cat][DESColList[0]].size

#
#       Find typical size of objects... clip and get approrpriate range of sizes (for stellar locus)
#
        avg_GM_size,med_GM_size,std_GM_size=medclip(kept_cat[Cat]['FLUX_RADIUS'],verbose=0)
        min_GM_size=avg_GM_size-(3.0*std_GM_size)
        max_GM_size=avg_GM_size+(3.0*std_GM_size)
        wsm=np.where(np.logical_and(kept_cat[Cat]['FLUX_RADIUS']>min_GM_size,kept_cat[Cat]['FLUX_RADIUS']<max_GM_size))

#
#       Look at S/N range of remaining items...
#
        tmp_sn=kept_cat[Cat]['FLUX_AUTO'][wsm]/kept_cat[Cat]['FLUXERR_AUTO'][wsm]
        min_GM_sn=np.amin(tmp_sn)
        if (min_GM_sn < args.sncut):
            min_GM_sn=args.sncut
        max_GM_sn=np.amax(tmp_sn)
        max_GM_flux_cut=np.amax(kept_cat[Cat]['FLUX_AUTO'][wsm])
        print("Peak flux currently: {:6.3e} ".format(max_GM_flux_cut))
        if (max_GM_sn > args.top_sn_thresh):
            max_GM_flux_cut=10.0**((-2.5*np.log10(max_GM_flux_cut)+args.remove_top_sn)/-2.5)
            print("   Peak flux cut to: {:6.3e} ".format(max_GM_flux_cut))
        tmp_sn=kept_cat[Cat]['FLUX_AUTO']/kept_cat[Cat]['FLUXERR_AUTO']
        wsm=np.where(np.logical_and(np.logical_and(kept_cat[Cat]['FLUX_RADIUS']>min_GM_size,kept_cat[Cat]['FLUX_RADIUS']<max_GM_size),
                                    np.logical_and(tmp_sn>min_GM_sn,kept_cat[Cat]['FLUX_AUTO']<max_GM_flux_cut)))
        for key in DESColDict:
            kept_cat[Cat][key]=kept_cat[Cat][key][wsm]
        MetaCat[Cat]['nm_GAIA_cut']=kept_cat[Cat][DESColList[0]].size

#
#       Look to see if there are objects that can be added.
#
        tmp_sn=diag_cat[Cat]['FLUX_AUTO']/diag_cat[Cat]['FLUXERR_AUTO']
        wsm=np.where(np.logical_and(np.logical_and(diag_cat[Cat]['FLUX_RADIUS']>min_GM_size,diag_cat[Cat]['FLUX_RADIUS']<max_GM_size),
                                    np.logical_and(tmp_sn>min_GM_sn,diag_cat[Cat]['FLUX_AUTO']<max_GM_flux_cut)))

        MetaCat[Cat]['nm_GAIA_add']=diag_cat[Cat][DESColList[0]][wsm].size
        MetaCat[Cat]['min_GM_sn']=min_GM_sn
        MetaCat[Cat]['max_GM_sn']=max_GM_sn
        MetaCat[Cat]['max_GM_flux_cut']=max_GM_flux_cut
        MetaCat[Cat]['min_GM_size']=min_GM_size
        MetaCat[Cat]['max_GM_size']=max_GM_size
        MetaCat[Cat]['avg_GM_size']=avg_GM_size
        MetaCat[Cat]['med_GM_size']=med_GM_size
        MetaCat[Cat]['std_GM_size']=std_GM_size
        
#
#       Repeat for VHS
#
        MetaCat[Cat]['nm_VHS']=0
        MetaCat[Cat]['nm_VHS_cut']=0
        MetaCat[Cat]['nm_VHS_add']=0
        avg_VM_size=-1
        med_VM_size=-1
        std_VM_size=-1
        min_VM_size=-1
        max_VM_size=-1
        min_VM_sn=-1
        max_VM_sn=-1
        if (checkVHS):
            idx2, d2d, d3d = c1.match_to_catalog_sky(c3)
            idx1=np.arange(ExpCat[Cat]['ALPHAWIN_J2000'].size)
            wsm=np.where(d2d.arcsecond<0.50)
            MetaCat[Cat]['nm_VHS']=idx1[wsm].size

            kept_cat_VHS[Cat]={}
            for key in DESColDict:
                kept_cat_VHS[Cat][key]=ExpCat[Cat][key][idx1[wsm]]
#
#           Remove FLAGGED and LOW signal-to-noise objects
#
            wsm=np.where(np.logical_and(
                np.logical_and(kept_cat_VHS[Cat]['FLAGS']==0,kept_cat_VHS[Cat]['IMAFLAGS_ISO']==0),
                kept_cat_VHS[Cat]['FLUX_AUTO']/kept_cat_VHS[Cat]['FLUXERR_AUTO']>args.sncut))
            for key in DESColDict:
                kept_cat_VHS[Cat][key]=kept_cat_VHS[Cat][key][wsm]
            MetaCat[Cat]['nm_VHS_cut']=kept_cat_VHS[Cat][DESColList[0]].size

            if (kept_cat_VHS[Cat]['FLUX_RADIUS'].size > 1):
                avg_VM_size,med_VM_size,std_VM_size=medclip(kept_cat_VHS[Cat]['FLUX_RADIUS'],verbose=0)
                min_VM_size=avg_VM_size-(3.0*std_VM_size)
                max_VM_size=avg_VM_size+(3.0*std_VM_size)
                wsm=np.where(np.logical_and(kept_cat_VHS[Cat]['FLUX_RADIUS']>min_VM_size,kept_cat_VHS[Cat]['FLUX_RADIUS']<max_VM_size))
                tmp_sn=kept_cat_VHS[Cat]['FLUX_AUTO'][wsm]/kept_cat_VHS[Cat]['FLUXERR_AUTO'][wsm]
                min_VM_sn=np.amin(tmp_sn)
                max_VM_sn=np.amax(tmp_sn)
                tmp_sn=diag_cat[Cat]['FLUX_AUTO']/diag_cat[Cat]['FLUXERR_AUTO']
                wsm=np.where(np.logical_and(np.logical_and(diag_cat[Cat]['FLUX_RADIUS']>min_VM_size,diag_cat[Cat]['FLUX_RADIUS']<max_VM_size),
                                        np.logical_and(tmp_sn>min_VM_sn,tmp_sn<max_VM_sn)))
                MetaCat[Cat]['nm_VHS_add']=diag_cat[Cat][DESColList[0]][wsm].size
                MetaCat[Cat]['avg_VM_size']=avg_VM_size
                MetaCat[Cat]['med_VM_size']=med_VM_size
                MetaCat[Cat]['std_VM_size']=std_VM_size
                MetaCat[Cat]['min_VM_size']=min_VM_size
                MetaCat[Cat]['max_VM_size']=max_VM_size
                MetaCat[Cat]['min_VM_sn']=min_VM_sn
                MetaCat[Cat]['max_VM_sn']=max_VM_sn

        print(" Match for {:7d} objects:  GAIA --> {:6d} cut {:6d} add {:6d},  VHS --> {:6d} cut {:6d} add {:6d} ".format(
            nobj0,
            MetaCat[Cat]['nm_GAIA'], MetaCat[Cat]['nm_GAIA_cut'], MetaCat[Cat]['nm_GAIA_add'], 
            MetaCat[Cat]['nm_VHS'],  MetaCat[Cat]['nm_VHS_cut'],  MetaCat[Cat]['nm_VHS_add']))
        print("   GAIA size: {:7.2f} {:7.2f} {:7.4f} -->  {:8.2f} {:8.2f} wS/N: {:8.1f} {:8.1f} ".format(
            MetaCat[Cat]['avg_GM_size'],MetaCat[Cat]['med_GM_size'],MetaCat[Cat]['std_GM_size'],
            MetaCat[Cat]['min_GM_size'],MetaCat[Cat]['max_GM_size'],MetaCat[Cat]['min_GM_sn'],MetaCat[Cat]['max_GM_sn']))
        if ('avg_VM_size' in MetaCat[Cat]):
            print("   VHS  size: {:7.2f} {:7.2f} {:7.4f} -->  {:8.2f} {:8.2f} wS/N: {:8.1f} {:8.1f} ".format(
                MetaCat[Cat]['avg_VM_size'],MetaCat[Cat]['med_VM_size'],MetaCat[Cat]['std_VM_size'],
                MetaCat[Cat]['min_VM_size'],MetaCat[Cat]['max_VM_size'],MetaCat[Cat]['min_VM_sn'],MetaCat[Cat]['max_VM_sn']))
        else:
            print("   VHS  size: {:7.2f} {:7.2f} {:7.4f} -->  {:8.2f} {:8.2f} wS/N: {:8.1f} {:8.1f} ".format(
                avg_VM_size,med_VM_size,std_VM_size,
                min_VM_size,max_VM_size,min_VM_sn,max_VM_sn))

#
#   Make the QA plots.
#

    if (args.qa_select is not None):
        AccumSizeAll=0
        AccumSizeCut=0
        AccumSizeCutVHS=0
        for Cat in grp_list:
            AccumSizeAll+=diag_cat[Cat]['FLUX_AUTO'].size
            AccumSizeCut+=kept_cat[Cat]['FLUX_AUTO'].size
            if (Cat in kept_cat_VHS):
                AccumSizeCutVHS+=kept_cat_VHS[Cat]['FLUX_AUTO'].size

        AccumDataAll={}
        AccumDataCut={}
        AccumDataCutVHS={}
        for Col in ['sn','spread_model','flux_radius']:
            AccumDataAll[Col]=np.zeros(AccumSizeAll,dtype='f8')
            AccumDataCut[Col]=np.zeros(AccumSizeCut,dtype='f8')
            AccumDataCutVHS[Col]=np.zeros(AccumSizeCutVHS,dtype='f8')

        print(AccumDataAll['sn'].size)
        print(AccumDataCut['sn'].size)
        print(AccumDataCutVHS['sn'].size)

        ctr_all=0
        ctr_cut=0
        ctr_cut_vhs=0
        for Cat in grp_list:
            key_array=diag_cat[Cat]['FLUX_AUTO']/diag_cat[Cat]['FLUXERR_AUTO']
            AccumDataAll['sn'][ctr_all:ctr_all+key_array.size]=key_array
            key_array=diag_cat[Cat]['SPREAD_MODEL']
            AccumDataAll['spread_model'][ctr_all:ctr_all+key_array.size]=key_array
            key_array=diag_cat[Cat]['FLUX_RADIUS']
            AccumDataAll['flux_radius'][ctr_all:ctr_all+key_array.size]=key_array
            ctr_all+=key_array.size

            key_array=kept_cat[Cat]['FLUX_AUTO']/kept_cat[Cat]['FLUXERR_AUTO']
            AccumDataCut['sn'][ctr_cut:ctr_cut+key_array.size]=key_array
            key_array=kept_cat[Cat]['SPREAD_MODEL']
            AccumDataCut['spread_model'][ctr_cut:ctr_cut+key_array.size]=key_array
            key_array=kept_cat[Cat]['FLUX_RADIUS']
            AccumDataCut['flux_radius'][ctr_cut:ctr_cut+key_array.size]=key_array
            ctr_cut+=key_array.size

            if (checkVHS):
                key_array=kept_cat_VHS[Cat]['FLUX_AUTO']/kept_cat_VHS[Cat]['FLUXERR_AUTO']
                AccumDataCutVHS['sn'][ctr_cut_vhs:ctr_cut_vhs+key_array.size]=key_array
                key_array=kept_cat_VHS[Cat]['SPREAD_MODEL']
                AccumDataCutVHS['spread_model'][ctr_cut_vhs:ctr_cut_vhs+key_array.size]=key_array
                key_array=kept_cat_VHS[Cat]['FLUX_RADIUS']
                AccumDataCutVHS['flux_radius'][ctr_cut_vhs:ctr_cut_vhs+key_array.size]=key_array
                ctr_cut_vhs+=key_array.size

#        wsm=np.where(AccumDataCut['spread_model']>0.005)
#        print("Interloper w/ SPREAD_MODEL>0.005: {:d}".format(AccumDataCut['spread_model'][wsm].size))
#
#        avg_size,med_size,std_size=medclip(AccumDataCut['flux_radius'],verbose=0)
#        print("Size (avg,med,RMS): {:8.2f} {:8.2f} {:8.4f} ".format(avg_size,med_size,std_size))
#        min_size=avg_size-(3.0*std_size)
#        max_size=avg_size+(3.0*std_size)
#        wsm=np.where(np.logical_and(AccumDataCut['flux_radius']>min_size,AccumDataCut['flux_radius']<max_size))
#        min_sn=np.amin(AccumDataCut['sn'][wsm])
#        max_sn=np.amax(AccumDataCut['sn'][wsm])
#        print("size range: {:8.2f} {:8.2f}  s/n range: {:8.1f} {:8.1f} ".format(min_size,max_size,min_sn,max_sn))
#        print(" N GAIA(input): {:7d} ".format(AccumDataCut['sn'].size))
#        print(" N GAIA: {:7d} ".format(AccumDataCut['sn'][wsm].size))
#
#        min_sn=np.amin(AccumDataCut['sn'][wsm])
#        max_sn=np.amax(AccumDataCut['sn'][wsm])
#
#        wsm=np.where(np.logical_and(np.logical_and(AccumDataAll['flux_radius']>min_size,AccumDataAll['flux_radius']<max_size),
#                                    np.logical_and(AccumDataAll['sn']>min_sn,AccumDataAll['sn']<max_sn)))
#        print(" N DES: {:7d} ".format(AccumDataAll['sn'][wsm].size))

        qa.plot_selection('{:s}'.format(args.qa_select),AccumDataAll,AccumDataCut,AccumDataCutVHS)

    if (args.qa_dist is not None):
        data={}
        data['m_gaia']={}
        data['s_gaia']={}
        data['n_gaia']={}
        data['m_vhs']={}
        data['s_vhs']={}
        data['n_vhs']={}

        for Cat in MetaCat:
            data['m_gaia'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['med_GM_size']
            data['s_gaia'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['std_GM_size']
            data['n_gaia'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['nm_GAIA_cut']
            if ('med_VM_size' in MetaCat[Cat]):
                data['m_vhs'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['med_VM_size']
            if ('std_VM_size' in MetaCat[Cat]):
                data['s_vhs'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['std_VM_size']
            if ('med_VM_size' in MetaCat[Cat]):
                data['n_vhs'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['nm_VHS_cut']
            if ('EXPNUM' in MetaCat[Cat]):
                data['EXPNUM']=MetaCat[Cat]['EXPNUM']

        qa.plot_FP_quant('{:s}'.format(args.qa_dist),data)


    for Cat in grp_list:
        oname='{odir:s}/D{exp:08d}_{band:s}_c{ccd:02d}_r{reqnum:04d}p{attnum:02d}_{osuf:s}.fits'.format(
            odir=args.odir,
            exp=MetaCat[Cat]['EXPNUM'],
            band=MetaCat[Cat]['BAND'],
            ccd=MetaCat[Cat]['CCDNUM'],
            reqnum=args.reqnum,
            attnum=args.attnum,
            osuf=args.suffix)
        print(oname)
        ofits = fitsio.FITS(oname,'rw',clobber=True)
        ofits.write(kept_cat[Cat],names=DESColList,extname='OBJECTS')
        ofits[0].write_key('OBJECTS',kept_cat[Cat]['XWIN_IMAGE'].size,comment=None)
        ofits.close()

#        if (checkVHS):
#            oname=re.sub("_red-fullcat.fits","_piffcat_vhs.fits",Cat)
#            ofits = fitsio.FITS(oname,'rw',clobber=True)
#            ofits.write(kept_cat_VHS[Cat],names=DESColList,extname='OBJECTS')
#            ofits[0].write_key('OBJECTS',kept_cat_VHS[Cat]['XWIN_IMAGE'].size,comment=None)
#            ofits.close()

#    print(DFP.fp_layout)

    exit(0)


#        print("Post GAIA: {:d}".format(kept_cat[Cat][DESColList[0]].size))
#        wsm=np.where(kept_cat[Cat]['FLAGS']==0)
#        print("Post GAIA [FLAGS=0]: {:d}".format(kept_cat[Cat][DESColList[0]][wsm].size))
#        wsm=np.where(kept_cat[Cat]['IMAFLAGS_ISO']==0)
#        print("Post GAIA [IMAFLAGS_ISO=0]: {:d}".format(kept_cat[Cat][DESColList[0]][wsm].size))
#        wsm=np.where((kept_cat[Cat]['FLUX_AUTO']/kept_cat[Cat]['FLUXERR_AUTO'])>args.sncut)
#        print("Post GAIA [S/N > {:.1f}]: {:d}".format(args.sncut,kept_cat[Cat][DESColList[0]][wsm].size))


