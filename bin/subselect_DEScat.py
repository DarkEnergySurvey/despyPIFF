#!/usr/bin/env python3
"""
Subselection of a DES (FINALCUT) catalog for use with PIFF
Subselection uses GAIA DR2 (and potentially VHS) 
"""

#from __future__ import print_function
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
import despydb.desdbi
from despyPIFF import subselect_QA as qa
import despyPIFF.piff_qa_utils  as pqu

from astropy.coordinates import SkyCoord
from astropy import units as u

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from matplotlib.collections import PolyCollection
#from matplotlib.patches import Polygon

###########################################
def read_data(fname,ldac=False,hdu=1,verbose=0):
    """
    Read a DES catalog (and optionally the LDAC_IMHEAD that might be associated
    """
    rfits=fitsio.FITS(fname,'r')
    if (ldac):
        ldachead=rfits[hdu-1].read()
    cols=rfits[hdu].get_colnames()
    if (verbose > 0):
        print("Cols: ",cols)
    data = rfits[hdu].read()
    rfits.close()

    if (ldac):
        return data,cols,ldachead
    else:
        return data,cols


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

#   Finished reading all catalogs.

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


######################################################################################
def MatchUndStellarSelect(c1,c2,InCat,c2name,ColDict,ColSize,SND,MatchRad=0.5,verbose=0):

    """ Work Horse routine to match a pair of catalogs and then go through and refine 
        the entries based on size and signal-to-noise

        Inputs:
            c1:         AstroPy SkyCoord struct for input catalog (InCat)
            c2:         AstroPy SkyCoord struct for matching catalog
            InCat:      The corresponding catalog that will be operated on
            c2name:     Name/Identifier (for reporting and output dict) of the matching catalog
            ColDict:    Dictionay of columns that will be carried forward from InCat
            ColSize:    A column name in InCat that is used when probing output catalog size
            SND:        Dict containing attributes when making decisions about Signal-to-Noise Cuts
            MatchRad:   Radius [in arcseconds] to use when making a match (default=0.5)
            dbSchema:   Schema over which queries will occur.
            verbose:    Integer setting level of verbosity when running.

        Returns:
            ret_cat:    Dict: Subset of InCat that matched c2 and survived further size/signal-to-noise cuts
            DMCat:      Diagnostic/MetaData returned from matching
    """
#
#   Initialize metadata being returned
#
    DMCat={'nm':0,
           'nm_cut':0,
           'nm_cut0':0,
           'nm_add':0,
           'avg_size':-1.,
           'med_size':-1.,
           'std_size':-1.,
           'min_size':-1.,
           'max_size':-1.,
           'min_sn':-1.,
           'max_sn':-1.,
           'max_flux_cut':-1.}
#
#   Find entries in c1 that spatially match those in c2
#
    idx2, d2d, d3d = c1.match_to_catalog_sky(c2)
    idx1=np.arange(InCat[ColSize].size)
    wsm=np.where(d2d.arcsecond<MatchRad)
    DMCat['nm']=idx1[wsm].size

    ret_cat={}
    for key in ColDict:
        ret_cat[key]=InCat[key][idx1[wsm]]
#
#   Remove FLAGGED and LOW signal-to-noise objects
#   For the DIAGNOSTIC just remove flagged....
#
    wsm=np.where(np.logical_and(np.logical_and(ret_cat['FLAGS']==0,ret_cat['IMAFLAGS_ISO']==0),
                                ret_cat['FLUX_AUTO']/ret_cat['FLUXERR_AUTO']>SND['sncut']))
    for key in ColDict:
        ret_cat[key]=ret_cat[key][wsm]
    DMCat['nm_cut0']=ret_cat[ColSize].size

#
#   If there is still data present then go ahead and analyze size and signal-to-noise distributions...
#
    if (ret_cat[ColSize].size > 1):
#
#       Find typical size of objects... clip and get approrpriate range of sizes (for stellar locus)
#
        avg_size,med_size,std_size=pqu.medclip(ret_cat['FLUX_RADIUS'],verbose=0)
        min_size=avg_size-(3.0*std_size)
        max_size=avg_size+(3.0*std_size)
        DMCat['avg_size']=avg_size
        DMCat['med_size']=med_size
        DMCat['std_size']=std_size
        DMCat['min_size']=min_size
        DMCat['max_size']=max_size
        wsm=np.where(np.logical_and(ret_cat['FLUX_RADIUS']>min_size,ret_cat['FLUX_RADIUS']<max_size))
#
#       Look at S/N range of remaining items...
#
        tmp_sn=ret_cat['FLUX_AUTO'][wsm]/ret_cat['FLUXERR_AUTO'][wsm]
        min_sn=np.amin(tmp_sn)
        if (min_sn < SND['sncut']):
            min_sn=SND['sncut']
        max_sn=np.amax(tmp_sn)
        max_flux_cut=np.amax(ret_cat['FLUX_AUTO'][wsm])
        print("Peak {:s} flux currently: {:6.3e} ".format(c2name,max_flux_cut))
        if (max_sn > SND['top_sn_thresh']):
            max_flux_cut=10.0**((-2.5*np.log10(max_flux_cut)+SND['remove_top_sn'])/-2.5)
            print("   Peak {:s} flux cut to: {:6.3e} ".format(c2name,max_flux_cut))
        DMCat['min_sn']=min_sn
        DMCat['max_sn']=max_sn
        DMCat['max_flux_cut']=max_flux_cut
#
#       Back to actual catalog... now removing points based on S/N and SIZE
#    
        tmp_sn=ret_cat['FLUX_AUTO']/ret_cat['FLUXERR_AUTO']
        wsm=np.where(np.logical_and(np.logical_and(ret_cat['FLUX_RADIUS']>min_size,ret_cat['FLUX_RADIUS']<max_size),
                                    np.logical_and(tmp_sn>min_sn,ret_cat['FLUX_AUTO']<max_flux_cut)))
        for key in ColDict:
            ret_cat[key]=ret_cat[key][wsm]
        DMCat['nm_cut']=ret_cat[ColSize].size

#
#       Look to get a sense whether there are objects that could be added back in
#
        tmp_sn=InCat['FLUX_AUTO']/InCat['FLUXERR_AUTO']
        wsm=np.where(np.logical_and(np.logical_and(InCat['FLUX_RADIUS']>min_size,InCat['FLUX_RADIUS']<max_size),
                                    np.logical_and(tmp_sn>min_sn,InCat['FLUX_AUTO']<max_flux_cut)))
        DMCat['nm_add']=InCat[ColSize][wsm].size

    return ret_cat,DMCat


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
        print("Args: {:}".format(args))

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
        try:
            desdmfile = os.environ["des_services"]
        except KeyError:
            desdmfile = None

        dbh = despydb.desdbi.DesDbi(desdmfile,args.section,retry=True)
#        cur = dbh.cursor()
#        dbh = despydb.desdbi.DesDbi(None,args.section,retry=True)
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
    SNdict={'sncut':args.sncut,
            'top_sn_thresh':args.top_sn_thresh,
            'remove_top_sn':args.remove_top_sn}
    diag_cat={}
    kept_cat_GAIA={} 
    kept_cat_VHS={}
    for Cat in grp_list:
        if (Cat not in MetaCat):
            MetaCat[Cat]={}
        print("Working on catalog: {:s}".format(Cat))
        nobj0=ExpCat[Cat][DESColList[0]].size
#
#       Form the diagnostic cat (basically a copy the input (but then remove FLAGGED objects)
#       (possibly don't need the np.copy any more)
#
        diag_cat[Cat]={}
        dwsm=np.where(np.logical_and(ExpCat[Cat]['FLAGS']==0,ExpCat[Cat]['IMAFLAGS_ISO']==0))
        for key in DESColDict:
            diag_cat[Cat][key]=np.copy(ExpCat[Cat][key][dwsm])

#
#       Perform matching against GAIA (and VHS) and then attempt to isolate further the appropriate portion of stellar locus
#
        c1=SkyCoord(ra=ExpCat[Cat]['ALPHAWIN_J2000']*u.degree,dec=ExpCat[Cat]['DELTAWIN_J2000']*u.degree)

        kept_cat_GAIA[Cat],GMCat=MatchUndStellarSelect(c1,c2,ExpCat[Cat],'GAIA',DESColDict,DESColList[0],SNdict,verbose=verbose)
        MetaCat[Cat]['GAIA']=GMCat
        if (checkVHS):
            kept_cat_VHS[Cat],VMCat=MatchUndStellarSelect(c1,c3,ExpCat[Cat],'VHS',DESColDict,DESColList[0],SNdict,verbose=verbose)
            MetaCat[Cat]['VHS']=VMCat
        else:
            MetaCat[Cat]['VHS']={'nm':0,'nm_cut':0,'nm_cut0':0,'nm_add':0,'avg_size':-1.,'med_size':-1.,'std_size':-1.,
                                 'min_size':-1.,'max_size':-1.,'min_sn':-1.,'max_sn':-1.,'max_flux_cut':-1.}
#  
 
        print(" Match for {:7d} objects:  GAIA --> {:6d} cut {:6d} add {:6d},  VHS --> {:6d} cut {:6d} add {:6d} ".format(
            nobj0,
            MetaCat[Cat]['GAIA']['nm'], MetaCat[Cat]['GAIA']['nm_cut'], MetaCat[Cat]['GAIA']['nm_add'], 
            MetaCat[Cat]['VHS']['nm'],  MetaCat[Cat]['VHS']['nm_cut'],  MetaCat[Cat]['VHS']['nm_add']))
        print("   GAIA size: {:7.2f} {:7.2f} {:7.4f} -->  {:8.2f} {:8.2f} wS/N: {:8.1f} {:8.1f} ".format(
            MetaCat[Cat]['GAIA']['avg_size'],MetaCat[Cat]['GAIA']['med_size'],MetaCat[Cat]['GAIA']['std_size'],
            MetaCat[Cat]['GAIA']['min_size'],MetaCat[Cat]['GAIA']['max_size'],MetaCat[Cat]['GAIA']['min_sn'],MetaCat[Cat]['GAIA']['max_sn']))
        print(" {:8.3} {:8.3f} ".format(np.amin(kept_cat_GAIA[Cat]['MAG_AUTO']),np.amax(kept_cat_GAIA[Cat]['MAG_AUTO'])))
        if (checkVHS):
            print("   VHS  size: {:7.2f} {:7.2f} {:7.4f} -->  {:8.2f} {:8.2f} wS/N: {:8.1f} {:8.1f} ".format(
                MetaCat[Cat]['VHS']['avg_size'],MetaCat[Cat]['VHS']['med_size'],MetaCat[Cat]['VHS']['std_size'],
                MetaCat[Cat]['VHS']['min_size'],MetaCat[Cat]['VHS']['max_size'],MetaCat[Cat]['VHS']['min_sn'],MetaCat[Cat]['VHS']['max_sn']))

#
#   Make the QA plots.
#

    if (args.qa_select is not None):
        AccumSize={'All':0,'GAIA':0,'VHS':0}
        for Cat in grp_list:
            AccumSize['All']+=diag_cat[Cat]['FLUX_AUTO'].size
            AccumSize['GAIA']+=kept_cat_GAIA[Cat]['FLUX_AUTO'].size
            if (Cat in kept_cat_VHS):
                AccumSize['VHS']+=kept_cat_VHS[Cat]['FLUX_AUTO'].size

        AccumDataAll={}
        AccumDataGAIA={}
        AccumDataVHS={}
        for Col in ['sn','spread_model','flux_radius']:
            AccumDataAll[Col]=np.zeros(AccumSize['All'],dtype='f8')
            AccumDataGAIA[Col]=np.zeros(AccumSize['GAIA'],dtype='f8')
            AccumDataVHS[Col]=np.zeros(AccumSize['VHS'],dtype='f8')

        print(AccumDataAll['sn'].size)
        print(AccumDataGAIA['sn'].size)
        print(AccumDataVHS['sn'].size)

        ctr_all=0
        ctr_gaia=0
        ctr_vhs=0
        for Cat in grp_list:
            key_array=diag_cat[Cat]['FLUX_AUTO']/diag_cat[Cat]['FLUXERR_AUTO']
            ks=key_array.size
            AccumDataAll['sn'][ctr_all:ctr_all+ks]=key_array
            key_array=diag_cat[Cat]['SPREAD_MODEL']
            AccumDataAll['spread_model'][ctr_all:ctr_all+ks]=key_array
            key_array=diag_cat[Cat]['FLUX_RADIUS']
            AccumDataAll['flux_radius'][ctr_all:ctr_all+ks]=key_array
            ctr_all+=ks

            key_array=kept_cat_GAIA[Cat]['FLUX_AUTO']/kept_cat_GAIA[Cat]['FLUXERR_AUTO']
            ks=key_array.size
            AccumDataGAIA['sn'][ctr_gaia:ctr_gaia+ks]=key_array
            key_array=kept_cat_GAIA[Cat]['SPREAD_MODEL']
            AccumDataGAIA['spread_model'][ctr_gaia:ctr_gaia+ks]=key_array
            key_array=kept_cat_GAIA[Cat]['FLUX_RADIUS']
            AccumDataGAIA['flux_radius'][ctr_gaia:ctr_gaia+ks]=key_array
            ctr_gaia+=ks

            if (checkVHS):
                key_array=kept_cat_VHS[Cat]['FLUX_AUTO']/kept_cat_VHS[Cat]['FLUXERR_AUTO']
                ks=key_array.size
                AccumDataVHS['sn'][ctr_vhs:ctr_vhs+ks]=key_array
                key_array=kept_cat_VHS[Cat]['SPREAD_MODEL']
                AccumDataVHS['spread_model'][ctr_vhs:ctr_vhs+ks]=key_array
                key_array=kept_cat_VHS[Cat]['FLUX_RADIUS']
                AccumDataVHS['flux_radius'][ctr_vhs:ctr_vhs+ks]=key_array
                ctr_vhs+=ks

        qa.plot_selection3('{:s}'.format(args.qa_select),AccumDataAll,AccumDataGAIA,AccumDataVHS)

    if (args.qa_dist is not None):
        data={}
        data['m_gaia']={}
        data['s_gaia']={}
        data['n_gaia']={}
        data['m_vhs']={}
        data['s_vhs']={}
        data['n_vhs']={}

        for Cat in MetaCat:
            data['m_gaia'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['GAIA']['med_size']
            data['s_gaia'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['GAIA']['std_size']
            data['n_gaia'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['GAIA']['nm_cut']
            if (checkVHS):
                if (MetaCat[Cat]['VHS']['med_size']>0):
                    data['m_vhs'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['VHS']['med_size']
                if (MetaCat[Cat]['VHS']['std_size']>0):
                    data['s_vhs'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['VHS']['std_size']
                if (MetaCat[Cat]['VHS']['nm_cut']>0):
                    data['n_vhs'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['VHS']['nm_cut']
            if ('EXPNUM' in MetaCat[Cat]):
                data['EXPNUM']=MetaCat[Cat]['EXPNUM']
        
        qa.plot_FP_quant('{:s}'.format(args.qa_dist),data)

#
#   Write out the GAIA selected entries 
#

    for Cat in grp_list:
        oname='{odir:s}/D{exp:08d}_{band:s}_c{ccd:02d}_r{reqnum:04d}p{attnum:02d}_{osuf:s}.fits'.format(
            odir=args.odir,
            exp=MetaCat[Cat]['EXPNUM'],
            band=MetaCat[Cat]['BAND'],
            ccd=MetaCat[Cat]['CCDNUM'],
            reqnum=args.reqnum,
            attnum=args.attnum,
            osuf=args.suffix)
        ofits = fitsio.FITS(oname,'rw',clobber=True)
        print("Writing GAIA subselected catalog (expnum,ccdnum,#entries,filename): {exp:8d} {ccd:02d} {num:5d} {fn:s} ".format(
            exp=MetaCat[Cat]['EXPNUM'],
            ccd=MetaCat[Cat]['CCDNUM'],
            num=kept_cat_GAIA[Cat][DESColList[0]].size,
            fn=oname))
        ofits.write(kept_cat_GAIA[Cat],names=DESColList,extname='OBJECTS')
        ofits[0].write_key('OBJECTS',kept_cat_GAIA[Cat][DESColList[0]].size,comment=None)
        ofits[0].write_key('BAND',MetaCat[Cat]['BAND'],comment='Short name for filter')
        ofits[0].write_key('EXPNUM',MetaCat[Cat]['EXPNUM'],comment='DECam Exposure Number')
        ofits[0].write_key('CCDNUM',MetaCat[Cat]['CCDNUM'],comment='CCD Number')
        ofits.close()


    exit(0)


