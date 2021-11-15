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
        query="""select /*+ INDEX(g GAIA_DR2_RADEC_BTX) */ g.source_id,g.ra,g.dec
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
        query="""select /*+ INDEX(g GAIA_DR2_RADEC_BTX) */ g.source_id,g.ra,g.dec
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
def get_Y6Gold_objects(radec_box,dbh,dbSchema,GoldTab='Y6_GOLD_2_0',Timing=False,verbose=0):

    """ Query code to obtain list of images that overlap another

        Inputs:
            radec_box: Dict with range to search in RA,Dec (with flag to handle case where RA range crosses RA=0h
            dbh:       Database connection to be used
            dbSchema:  Schema over which queries will occur.
            GoldTab:   Gives the name of the specific Y6_GOLD table version to use (defaults to Y6_GOLD_2_0).
            Timing:    Output timing information as query runs.
            verbose:   Integer setting level of verbosity when running.

        Returns:
            CatDict: Resulting Image dictionary
    """

    t0=time.time()

    if (radec_box['crossra0']):
#
#       Form Query for case where RA ranges crosses RA=0h (not very good at poles)
#
#        query="""select y.COADD_OBJECT_ID,y.ALPHAWIN_J2000,y.DELTAWIN_J2000,
        query="""select /*+ INDEX(y Y6_GOLD_2_0_RADEC_BTX) */ y.COADD_OBJECT_ID,y.ALPHAWIN_J2000,y.DELTAWIN_J2000,
            y.PSF_FLUX_APER_8_G,y.PSF_FLUX_ERR_APER_8_G,y.PSF_FLUX_FLAGS_G,
            y.PSF_FLUX_APER_8_R,y.PSF_FLUX_ERR_APER_8_R,y.PSF_FLUX_FLAGS_R,
            y.PSF_FLUX_APER_8_I,y.PSF_FLUX_ERR_APER_8_I,y.PSF_FLUX_FLAGS_I,
            y.PSF_FLUX_APER_8_Z,y.PSF_FLUX_ERR_APER_8_Z,y.PSF_FLUX_FLAGS_Z,
            y.PSF_FLUX_APER_8_Y,y.PSF_FLUX_ERR_APER_8_Y,y.PSF_FLUX_FLAGS_Y,
            y.EXT_MASH 
        from {schema:s}{gtab:s} y
        where (y.ra < {r2:.6f} or y.ra > {r1:.6f})
            and y.dec between {d1:.6f} and {d2:.6f}""".format(
        schema=dbSchema,
        gtab=GoldTab,
        r1=radec_box['ra1'],
        r2=radec_box['ra2'],
        d1=radec_box['dec1'],
        d2=radec_box['dec2'])

    else:
#
#       Form query for normal workhorse case 
#
#        query="""select y.COADD_OBJECT_ID,y.ALPHAWIN_J2000,y.DELTAWIN_J2000,
        query="""select /*+ INDEX(y Y6_GOLD_2_0_RADEC_BTX) */ y.COADD_OBJECT_ID,y.ALPHAWIN_J2000,y.DELTAWIN_J2000,
            y.PSF_FLUX_APER_8_G,y.PSF_FLUX_ERR_APER_8_G,y.PSF_FLUX_FLAGS_G,
            y.PSF_FLUX_APER_8_R,y.PSF_FLUX_ERR_APER_8_R,y.PSF_FLUX_FLAGS_R,
            y.PSF_FLUX_APER_8_I,y.PSF_FLUX_ERR_APER_8_I,y.PSF_FLUX_FLAGS_I,
            y.PSF_FLUX_APER_8_Z,y.PSF_FLUX_ERR_APER_8_Z,y.PSF_FLUX_FLAGS_Z,
            y.PSF_FLUX_APER_8_Y,y.PSF_FLUX_ERR_APER_8_Y,y.PSF_FLUX_FLAGS_Y,
            y.EXT_MASH 
        from {schema:s}{gtab:s} y
        where (y.ra between {r1:.6f} and {r2:.6f})
            and y.dec between {d1:.6f} and {d2:.6f}""".format(
        schema=dbSchema,
        gtab=GoldTab,
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
        print("# No values returned from query of {tval:s} ".format(tval=GoldTab))
        for val in header:
            CatDict[val]=np.array([])
    else:
        cat_data.columns=header
        for val in header:
            CatDict[val]=np.array(cat_data[val])
    curDB.close()

    if (verbose>0):
        print("# Number of objects in {tval:s} found is {nval:d} ".format(tval=GoldTab,nval=CatDict[header[0]].size))
    if (Timing):
        t1=time.time()
        print(" Query execution time: {:.2f}".format(t1-t0))

    return CatDict,header


######################################################################################
def get_Y6GoldPhot_GAIAObjects(SourceIDList,dbh,dbSchema,Timing=False,verbose=0):

    """ Query code to obtain list of images that overlap another

        Inputs:
            SourceIDList: List (of Lists) that contain GAIA_DR2 SOURCE_IDs to search for pre-matched photometry.
            dbh:       Database connection to be used
            dbSchema:  Schema over which queries will occur.
            verbose:   Integer setting level of verbosity when running.

        Returns:
            CatDict: Resulting Image dictionary
    """

    t0=time.time()

#
#   Establish a DB cursor
#   Load the temp table...
#
    curDB = dbh.cursor()
    curDB.execute('delete from GTT_ID')
    # load ids into gtt_id table
    print(f"# Loading GTT_ID table for secondary query with entries for {len(SourceIDList):d} images")
    dbh.insert_many('GTT_ID', ['ID'], SourceIDList)

#
#   Form query and obtain pre-determined matches
#

    query="""select x.source_id,
            y.PSF_FLUX_APER_8_G,y.PSF_FLUX_ERR_APER_8_G,y.PSF_FLUX_FLAGS_G,
            y.PSF_FLUX_APER_8_R,y.PSF_FLUX_ERR_APER_8_R,y.PSF_FLUX_FLAGS_R,
            y.PSF_FLUX_APER_8_I,y.PSF_FLUX_ERR_APER_8_I,y.PSF_FLUX_FLAGS_I,
            y.PSF_FLUX_APER_8_Z,y.PSF_FLUX_ERR_APER_8_Z,y.PSF_FLUX_FLAGS_Z,
            y.PSF_FLUX_APER_8_Y,y.PSF_FLUX_ERR_APER_8_Y,y.PSF_FLUX_FLAGS_Y
        from Y6_GOLD_2_0 y, gaia_dr2_x_y6_gold_2_0 x, gtt_id g 
        where g.id=x.source_id and x.coadd_object_id=y.coadd_object_id"""
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

    prefetch=100000
    curDB.arraysize=int(prefetch)
    curDB.execute(query)
    desc = [d[0].lower() for d in curDB.description]

    CatDict={}
    for row in curDB:
        rowd = dict(zip(desc, row))
        source_id = rowd.pop('source_id')
        CatDict[source_id] = rowd

    curDB.close()

    if (verbose>0):
        print("# Number of DES objects found is {nval:d} ".format(nval=len(CatDict)))
    if (Timing):
        t1=time.time()
        print(" Query execution time: {:.2f}".format(t1-t0))

#
#   Drop source_id since it was pulled from the dictionary contents and used as a key
#
    desc.remove('source_id')

    return CatDict,desc


######################################################################################
def SimplifiedMatch(c1,c2,InCat1,InCat2,ColDict2,FlagCol=None,MatchRad=0.5,verbose=0):

    """ Work Horse routine to match a pair of catalogs and then go through and refine 
        the entries based on size and signal-to-noise

        Inputs:
            c1:         AstroPy SkyCoord struct for input catalog (InCat1)
            c2:         AstroPy SkyCoord struct for matching catalog (InCat2)
            InCat1:     The catalog corresponding to c1 that will be operated on
            InCat2:     The catalog corresponding to c2 that will be operated on
            ColDict2:    Dictionary/List of columns that will be carried forward from InCat
            FlagCol:    Optional name for a new column inidicating a match existed.
            MatchRad:   Radius [in arcseconds] to use when making a match (default=0.5)
            verbose:    Integer setting level of verbosity when running.

        Returns:
            ret_cat:    Dict: Subset of InCat1 and Incat2 that matched c2 and survived further size/signal-to-noise cuts
            DMCat:      Diagnostic/MetaData returned from matching
    """
#
#   This is here to maintain some backward compatibility (so can accept a list or dict for ColDict1/2)
#
    keylist=[*InCat1]
    refkey=keylist[0]

#
#   Find entries in c1 that spatially match those in c2
#
    if (c2.size > 0):
        idx2, d2d, d3d = c1.match_to_catalog_sky(c2)
        idx1=np.arange(InCat1[refkey].size)
        wsm=np.where(d2d.arcsecond<MatchRad)
#    DMCat['nm']=idx1[wsm].size

#
#   If requested create FLAG column
#
    if (not(FlagCol is None)):
        InCat1[FlagCol]=np.zeros(InCat1[refkey].size,dtype='i4')
        if (c2.size > 0):
            InCat1[FlagCol][idx1[wsm]]=1
#
#   Copy specified columns from InCat2 (for entries that matched)
#
    for key in ColDict2:
        if (key in InCat2):
            InCat1[key]=np.zeros(InCat1[refkey].size,dtype=InCat2[key].dtype)
            if (c2.size > 0):
                InCat1[key][idx1[wsm]]=InCat2[key][idx2[wsm]]

    return InCat1


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
    parser.add_argument('--color',        action='store', type=str, default='g-i', help='Color to write to output file (default=g-i; written as GI_COLOR)')
    parser.add_argument('--sentinel_color',  action='store', type=float, default=1.6, help='Sentinel value to write for color (default=1.6)')
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
#
#   Break --color into a numerator and denominator
#
    color_comp=args.color.split('-')
    cnum=color_comp[0]
    cden=color_comp[1]
    

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

    if (args.checkVHS):
        if (VHSCat[VHSCatCols[0]].size < 2):
            checkVHS=False

#
#   Pre-Prepare GAIA data (and VHS data) for matching
#

    c2=SkyCoord(ra=GaiaCat['RA']*u.degree,dec=GaiaCat['DEC']*u.degree)
    GaiaColFwd=['SOURCE_ID']
    if (checkVHS):
        VHSColFwd=[]
        c3=SkyCoord(ra=VHSCat['RA']*u.degree,dec=VHSCat['DEC']*u.degree)
#
#
    PhotCat,PhotCatCols=get_Y6Gold_objects(RaDecRange,dbh,dbSchema,Timing=True,verbose=verbose)
    c4=SkyCoord(ra=PhotCat['ALPHAWIN_J2000']*u.degree,dec=PhotCat['DELTAWIN_J2000']*u.degree)
  
    PhotCatKeep=[]
    for col in PhotCatCols:
        if (col not in ['ALPHAWIN_J2000','DELTAWIN_J2000','COADD_OBJECT_ID']):
            PhotCatKeep.append(col)
 
#
#   Main body catalog comparison and source selection
# 
    SNdict={'sncut':args.sncut,
            'top_sn_thresh':args.top_sn_thresh,
            'remove_top_sn':args.remove_top_sn}
    AccumSize=0
    for Cat in grp_list:
        if (Cat not in MetaCat):
            MetaCat[Cat]={}
        print("Working on catalog: {:s}".format(Cat))
        nobj0=ExpCat[Cat][DESColList[0]].size
#
#       Perform matching against GAIA (and VHS) and then attempt to isolate further the appropriate portion of stellar locus
#
        c1=SkyCoord(ra=ExpCat[Cat]['ALPHAWIN_J2000']*u.degree,dec=ExpCat[Cat]['DELTAWIN_J2000']*u.degree)
#
#       Check for what columns exist
#        for key in ExpCat[Cat]:
#            print(key,ExpCat[Cat][key].dtype)

        ExpCat[Cat]=SimplifiedMatch(c1,c2,ExpCat[Cat],GaiaCat,[],FlagCol='GAIA_STAR',verbose=verbose)
        if (verbose > 0):
            wsm=np.where(ExpCat[Cat]['GAIA_STAR']>0)
            print("Performed match to GAIA. Matched {:d} objects.".format(ExpCat[Cat]['GAIA_STAR'][wsm].size))

        ExpCat[Cat]=SimplifiedMatch(c1,c4,ExpCat[Cat],PhotCat,PhotCatKeep,FlagCol='PHOT_OBJ',verbose=verbose)
        if (verbose > 0):
            wsm=np.where(ExpCat[Cat]['PHOT_OBJ']>0)
            print("Performed match to photoemtric catalog.  Matched {:d} objects.".format(ExpCat[Cat]['PHOT_OBJ'][wsm].size))
            if ('EXT_MASH' in ExpCat[Cat]):
                wsm=np.where(ExpCat[Cat]['PHOT_OBJ']>0)

        if (checkVHS):
            ExpCat[Cat]=SimplifiedMatch(c1,c3,ExpCat[Cat],VHSCAT,[],FlagCol='VHS_STAR',verbose=verbose)
            if (verbose > 0):
                wsm=np.where(ExpCat[Cat]['VHS_STAR']>0)
                print("Performed match to VHS.  Matched {:d} objects.".format(ExpCat[Cat]['VHS_STAR'][wsm].size))

#
#       Remove objects with SExtractor or Image Flags havee been thrown
#
        wsm=np.where(np.logical_and(ExpCat[Cat]['FLAGS']==0,ExpCat[Cat]['IMAFLAGS_ISO']==0))
        nobj0=ExpCat[Cat][DESColList[0]].size
        for key in ExpCat[Cat]:
           ExpCat[Cat][key]=ExpCat[Cat][key][wsm]
        nobj_flag=ExpCat[Cat][DESColList[0]].size
        print("SExtractor and Image flags reduce catalog from {:d} to {:d} objects".format(nobj0,nobj_flag))
#
#       Check for brightest GAIA Source.  If it exceeds a S/N of 1000. (  ) then find a limit equivalent to 1.2 mag fainter and cut sources brighter than this
#       - split out GAIA matches
#       - get typical size (so that some non-stellar objects might be removed
#       - perform check on S/N of brightest object remaining

        wsm=np.where(ExpCat[Cat]['GAIA_STAR']==1)
        GaiaSource={
            'sn':ExpCat[Cat]['FLUX_AUTO'][wsm]/ExpCat[Cat]['FLUXERR_AUTO'][wsm],
            'flux':ExpCat[Cat]['FLUX_AUTO'][wsm],
            'size':ExpCat[Cat]['FLUX_RADIUS'][wsm],
            'smod':ExpCat[Cat]['SPREAD_MODEL'][wsm]}
#
#       Find typical size of objects... clip and get approrpriate range of sizes (for stellar locus)
#
        avg_size,med_size,std_size=pqu.medclip(GaiaSource['size'],verbose=0)
        min_size=avg_size-(3.0*std_size)
        max_size=avg_size+(3.0*std_size)
        wsm=np.where(np.logical_and(GaiaSource['size']>min_size,GaiaSource['size']<max_size))
        max_sn=np.amax(GaiaSource['sn'][wsm])
        max_flux_cut=np.amax(GaiaSource['flux'][wsm])
        print("Peak flux currently: {:6.3e} ".format(max_flux_cut))
#
        if (max_sn > SNdict['top_sn_thresh']):
            max_flux_cut=10.0**((-2.5*np.log10(max_flux_cut)+SNdict['remove_top_sn'])/-2.5)
            print("   Peak flux cut to: {:6.3e} ".format(max_flux_cut))
#
#       Flag GAIA matches that were outside range of normal sizes
#
        wsm=np.where(np.logical_and(ExpCat[Cat]['GAIA_STAR']==1,
                     np.logical_or(ExpCat[Cat]['FLUX_RADIUS']>max_size,ExpCat[Cat]['FLUX_RADIUS']<min_size)))
        ExpCat[Cat]['GAIA_STAR'][wsm]+=2
#        print("size: {:.2f} {:.2f}  {:d}".format(min_size,max_size,ExpCat[Cat]['GAIA_STAR'][wsm].size))
#
#       Remove all objects that exceed the limit established by GAIA
#       (performed as keep all objects with lower S/N
#
        wsm=np.where(ExpCat[Cat]['FLUX_AUTO']<max_flux_cut)
        for key in ExpCat[Cat]:
           ExpCat[Cat][key]=ExpCat[Cat][key][wsm]
        nobj_highSN=ExpCat[Cat][DESColList[0]].size
        print("  High S/N cut further reduced catalog from {:d} to {:d} objects".format(nobj_flag,nobj_highSN))

#       Below is leftover from debugging... remains for now (in case)

#        for i in range(ExpCat[Cat]['MAG_AUTO'].size):
#            if (ExpCat[Cat]['GAIA_STAR'][i] == 3):
#                print("A {:5d} {:2d} {:2d} {:2d} {:6d} {:6d} {:6.1f} {:8.3f}  {:7.2f}  {:8.1f} {:8.1f} {:11.7f} {:11.7f} ".format(
#                i,ExpCat[Cat]['GAIA_STAR'][i],ExpCat[Cat]['PHOT_OBJ'][i],ExpCat[Cat]['EXT_MASH'][i],
#                ExpCat[Cat]['FLAGS'][i],ExpCat[Cat]['IMAFLAGS_ISO'][i],
#                ExpCat[Cat]['FLUX_AUTO'][i]/ExpCat[Cat]['FLUXERR_AUTO'][i],ExpCat[Cat]['MAG_AUTO'][i],
#                ExpCat[Cat]['FLUX_RADIUS'][i],ExpCat[Cat]['XWIN_IMAGE'][i],ExpCat[Cat]['YWIN_IMAGE'][i],
#                ExpCat[Cat]['ALPHAWIN_J2000'][i],ExpCat[Cat]['DELTAWIN_J2000'][i]))
  
        AccumSize=AccumSize+nobj_highSN 

#       FINISHED LOOPING OVER CATS

#
#   Make the QA plots.
#
    if (args.qa_select is not None):

        AccumData={}
        for Col in ['sn','spread_model','flux_radius']:
            AccumData[Col]=np.zeros(AccumSize,dtype='f8')
        for Col in ['gaia_star','phot_obj','vhs_star']:
            AccumData[Col]=np.zeros(AccumSize,dtype='i4')

        ctr_all=0
        for Cat in grp_list:
            key_array=ExpCat[Cat]['FLUX_AUTO']/ExpCat[Cat]['FLUXERR_AUTO']
            ks=key_array.size
            AccumData['sn'][ctr_all:ctr_all+ks]=key_array
            key_array=ExpCat[Cat]['SPREAD_MODEL']
            AccumData['spread_model'][ctr_all:ctr_all+ks]=key_array
            key_array=ExpCat[Cat]['FLUX_RADIUS']
            AccumData['flux_radius'][ctr_all:ctr_all+ks]=key_array
            key_array=ExpCat[Cat]['GAIA_STAR']
            AccumData['gaia_star'][ctr_all:ctr_all+ks]=key_array
            key_array=ExpCat[Cat]['PHOT_OBJ']
            AccumData['phot_obj'][ctr_all:ctr_all+ks]=key_array
            if ('VHS_STAR' in ExpCat[Cat]):
                key_array=ExpCat[Cat]['VHS_STAR']
                AccumData['vhs_star'][ctr_all:ctr_all+ks]=key_array
            ctr_all+=ks

        qa.plot_selection3('{:s}'.format(args.qa_select),AccumData)

#    if (args.qa_dist is not None):
#        data={}
#        data['m_gaia']={}
#        data['s_gaia']={}
#        data['n_gaia']={}
#        data['m_vhs']={}
#        data['s_vhs']={}
#        data['n_vhs']={}
#
#        for Cat in MetaCat:
#            data['m_gaia'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['GAIA']['med_size']
#            data['s_gaia'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['GAIA']['std_size']
#            data['n_gaia'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['GAIA']['nm_cut']
#            if (checkVHS):
#                if (MetaCat[Cat]['VHS']['med_size']>0):
#                    data['m_vhs'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['VHS']['med_size']
#                if (MetaCat[Cat]['VHS']['std_size']>0):
#                    data['s_vhs'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['VHS']['std_size']
#                if (MetaCat[Cat]['VHS']['nm_cut']>0):
#                    data['n_vhs'][MetaCat[Cat]['CCDNUM']]=MetaCat[Cat]['VHS']['nm_cut']
#            if ('EXPNUM' in MetaCat[Cat]):
#                data['EXPNUM']=MetaCat[Cat]['EXPNUM']
#        
#        qa.plot_FP_quant('{:s}'.format(args.qa_dist),data)

##
##   Obtain DES color information 
##   Form list of lists containing IDs (for upload in a search)
##
#    IDList=[]
#    for Cat in kept_cat_GAIA:
#        for i in range(kept_cat_GAIA[Cat]['SOURCE_ID'].size):
#            IDList.append([int(kept_cat_GAIA[Cat]['SOURCE_ID'][i])])
#    Y6GoldCat,Y6GoldCols=get_Y6GoldPhot_GAIAObjects(IDList,dbh,dbSchema,Timing=True,verbose=2)

#
#   Form photometry columns for output catalogs 
#

    color_dict={
        'GI_COLOR':{'bit':1,'num':'G','den':'I','sentinel':1.6},
        'IZ_COLOR':{'bit':2,'num':'I','den':'Z','sentinel':0.25}
    }
    next_bit=4
#   Could be useful if we need to form colors or over-ride defaults
    color_col='{:s}{:s}_COLOR'.format(cnum.upper(),cden.upper())
    if (color_col not in color_dict):
        print("Creating color definition on the fly for: {:s}".format(color_col))
        color_dict[color_col]={'bit':next_bit,'num':cnum.upper(),'den':cden.upper(),'sentinel':args.sentinel_color}
    else:
        if (color_dict[color_col]['sentinel']!=args.sentinel_color):
            print("Command line over-ride of sentinel color for {:s}... being set to: {:f}".format(color_col,args.sentinel_color))
            color_dict[color_col]['sentinel']=args.sentinel_color


    print("Forming color information for:")
    for col in color_dict:
        print(" {:s} with FLAG_COLOR bit: {:d} and sentinel {:f} ".format(col,color_dict[col]['bit'],color_dict[col]['sentinel']))
    
    for Cat in ExpCat:
#
#       Add columns (fill with sentinels and flag all
#
        nobj=ExpCat[Cat][DESColList[0]].size
        ExpCat[Cat]['FLAG_COLOR']=np.zeros(nobj,dtype='i4')
        for col in color_dict:
            num=color_dict[col]['num']
            den=color_dict[col]['den']
            ExpCat[Cat][col]=np.full(nobj,color_dict[col]['sentinel'],dtype='f8')
            ExpCat[Cat]['FLAG_COLOR']+=color_dict[col]['bit']

            s_num=ExpCat[Cat]['PSF_FLUX_APER_8_{:s}'.format(num)]
            n_num=ExpCat[Cat]['PSF_FLUX_ERR_APER_8_{:s}'.format(num)]
            sn_num = np.divide(s_num, n_num, out=np.zeros_like(s_num), where=n_num!=0)

            s_den=ExpCat[Cat]['PSF_FLUX_APER_8_{:s}'.format(den)]
            n_den=ExpCat[Cat]['PSF_FLUX_ERR_APER_8_{:s}'.format(den)]
            sn_den = np.divide(s_den, n_den, out=np.zeros_like(s_den), where=n_den!=0)

#            sn_num=ExpCat[Cat]['PSF_FLUX_APER_8_{:s}'.format(num)]/ExpCat[Cat]['PSF_FLUX_ERR_APER_8_{:s}'.format(num)]
#            sn_den=ExpCat[Cat]['PSF_FLUX_APER_8_{:s}'.format(den)]/ExpCat[Cat]['PSF_FLUX_ERR_APER_8_{:s}'.format(den)]
            cflag=ExpCat[Cat]['PSF_FLUX_FLAGS_{:s}'.format(num)]+ExpCat[Cat]['PSF_FLUX_FLAGS_{:s}'.format(den)]
           
            wsm=np.where(np.logical_and(np.logical_and(sn_num>10.,sn_den>10.),np.logical_and(ExpCat[Cat]['PHOT_OBJ']==1,cflag==0)))
#
#           calculate color (note numerator and denominator are flipped to remove need for "-2.5")
#           (also remove flag bit)
#
            ExpCat[Cat][col][wsm]=2.5*np.log10(ExpCat[Cat]['PSF_FLUX_APER_8_{:s}'.format(den)][wsm]/ExpCat[Cat]['PSF_FLUX_APER_8_{:s}'.format(num)][wsm]) 
            ExpCat[Cat]['FLAG_COLOR'][wsm]-=color_dict[col]['bit']

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
        print("Writing catalog with GAIA subselection and photometry (expnum,ccdnum,#entries,filename): {exp:8d} {ccd:02d} {num:5d} {fn:s} ".format(
            exp=MetaCat[Cat]['EXPNUM'],
            ccd=MetaCat[Cat]['CCDNUM'],
            num=ExpCat[Cat][DESColList[0]].size,
            fn=oname))
        CombCatCols=[]
        for col in ExpCat[Cat]:
            CombCatCols.append(col)
        ofits.write(ExpCat[Cat],names=CombCatCols,extname='OBJECTS')
        ofits[0].write_key('OBJECTS',ExpCat[Cat][DESColList[0]].size,comment=None)
        ofits[0].write_key('BAND',MetaCat[Cat]['BAND'],comment='Short name for filter')
        ofits[0].write_key('EXPNUM',MetaCat[Cat]['EXPNUM'],comment='DECam Exposure Number')
        ofits[0].write_key('CCDNUM',MetaCat[Cat]['CCDNUM'],comment='CCD Number')
        ofits.close()

    if ((args.useDB)or(args.checkVHS)):
        dbh.close()

    exit(0)


