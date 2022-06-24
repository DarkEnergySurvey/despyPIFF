#!/usr/bin/env python3
"""
Subselection of a DES (FINALCUT) catalog for use with PIFF
Subselection uses GAIA DR3 (and potentially VHS) 
"""

#from __future__ import print_function
import argparse
import os
import re
import time
import sys
import numpy as np
import fitsio

import pandas as pd
import despydb.desdbi
from despyPIFF import subselect_QA as qa
import despyPIFF.piff_qa_utils  as pqu

from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u


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
def get_GAIA_objects(radec_box,dbh,dbSchema,release='EDR3',Timing=False,verbose=0):

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
        query="""select /*+ INDEX(g GAIA_{rname:s}_RADEC_BTX) */ g.source_id as gaia_source_id,g.ra,g.dec
            from des_admin.GAIA_{rname:s} g
            where (g.ra < {r2:.6f} or g.ra > {r1:.6f})
                and g.dec between {d1:.6f} and {d2:.6f}""".format(
        rname=release,
        r1=radec_box['ra1'],
        r2=radec_box['ra2'],
        d1=radec_box['dec1'],
        d2=radec_box['dec2'])
    else:
#
#       Form query for normal workhorse case 
#
        query="""select /*+ INDEX(g GAIA_{rname:s}_RADEC_BTX) */ g.source_id as gaia_source_id,g.ra,g.dec
            from des_admin.GAIA_{rname:s} g
            where g.ra between {r1:.6f} and {r2:.6f}
                and g.dec between {d1:.6f} and {d2:.6f}""".format(
        rname=release,
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
        print("# No values returned from query of GAIA_{rname:s} ".format(rname=release))
        for val in header:
            CatDict[val]=np.array([])
    else:
        cat_data.columns=header
        for val in header:
            CatDict[val]=np.array(cat_data[val])
    curDB.close()

    if (verbose>0):
        print("# Number of GAIA_{rname:s} objects found is {nval:d} ".format(rname=release,nval=CatDict[header[0]].size))
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
        query="""select v.sourceid as vhs_sourceid,v.ra2000,v.dec2000,v.ksapermag3_ab as k_mag,v.ksapermag3err_ab as k_magerr,v.kserrbits as k_flags
            from des_admin.vhs_viking_ks v
            where (v.ra2000 < {r2:.6f} or v.ra2000 > {r1:.6f})
                and v.dec2000 between {d1:.6f} and {d2:.6f}""".format(
        r1=radec_box['ra1'],
        r2=radec_box['ra2'],
        d1=radec_box['dec1'],
        d2=radec_box['dec2'])
    else:
#
#       Normal workhorse case 
#
        query="""select v.sourceid as vhs_sourceid,v.ra2000,v.dec2000,v.ksapermag3_ab as k_mag,v.ksapermag3err_ab as k_magerr,v.kserrbits as k_flags
            from des_admin.vhs_viking_ks v
            where v.ra2000 between {r1:.6f} and {r2:.6f}
                and v.dec2000 between {d1:.6f} and {d2:.6f}""".format(
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

    """ Query code to obtain measurements from DES Gold catalog based on a ra-dec box constraint

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
#        query="""select /*+ INDEX(y Y6_GOLD_2_0_RADEC_BTX) */ y.COADD_OBJECT_ID,y.ALPHAWIN_J2000,y.DELTAWIN_J2000,
        query="""select y.COADD_OBJECT_ID,y.ALPHAWIN_J2000,y.DELTAWIN_J2000,
            y.PSF_MAG_APER_8_G as G_MAG, y.PSF_MAG_ERR_G as G_MAGERR, y.PSF_FLUX_FLAGS_G as G_FLAGS,
            y.PSF_MAG_APER_8_R as R_MAG, y.PSF_MAG_ERR_R as R_MAGERR, y.PSF_FLUX_FLAGS_R as R_FLAGS,
            y.PSF_MAG_APER_8_I as I_MAG, y.PSF_MAG_ERR_I as I_MAGERR, y.PSF_FLUX_FLAGS_I as I_FLAGS,
            y.PSF_MAG_APER_8_Z as Z_MAG, y.PSF_MAG_ERR_Z as Z_MAGERR, y.PSF_FLUX_FLAGS_Z as Z_FLAGS,
            y.PSF_MAG_APER_8_Y as Y_MAG, y.PSF_MAG_ERR_Y as Y_MAGERR, y.PSF_FLUX_FLAGS_Y as Y_FLAGS,
            y.EXT_MASH,y.BDF_T
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
#        query="""select /*+ INDEX(y Y6_GOLD_2_0_RADEC_BTX) */ y.COADD_OBJECT_ID,y.ALPHAWIN_J2000,y.DELTAWIN_J2000,
        query="""select y.COADD_OBJECT_ID,y.ALPHAWIN_J2000,y.DELTAWIN_J2000,
            y.PSF_MAG_APER_8_G as G_MAG, y.PSF_MAG_ERR_G as G_MAGERR, y.PSF_FLUX_FLAGS_G as G_FLAGS,
            y.PSF_MAG_APER_8_R as R_MAG, y.PSF_MAG_ERR_R as R_MAGERR, y.PSF_FLUX_FLAGS_R as R_FLAGS,
            y.PSF_MAG_APER_8_I as I_MAG, y.PSF_MAG_ERR_I as I_MAGERR, y.PSF_FLUX_FLAGS_I as I_FLAGS,
            y.PSF_MAG_APER_8_Z as Z_MAG, y.PSF_MAG_ERR_Z as Z_MAGERR, y.PSF_FLUX_FLAGS_Z as Z_FLAGS,
            y.PSF_MAG_APER_8_Y as Y_MAG, y.PSF_MAG_ERR_Y as Y_MAGERR, y.PSF_FLUX_FLAGS_Y as Y_FLAGS,
            y.EXT_MASH,y.BDF_T
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
    parser.add_argument('--sentinel_color',  action='store', type=float, default=1.3, help='Sentinel value to write for color (default=1.6)')
    parser.add_argument('--qa_select',    action='store', type=str, default=None, help='File name for selection QA plots')
    parser.add_argument('--qa_dist',      action='store', type=str, default=None, help='File name for distribution QA plots')

    parser.add_argument('--gaiadr2',      action='store_true', default=False, help='Flag to use GAIA_DR2 (default will query GAIA EDR3)')
    parser.add_argument('--gsm_rad',      action='store', type=float, default=2.0, help='GAIA self-match radius in arcseconds (default=2.0).  Set to negative to turn off.')

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
        if (args.gaiadr2):
            print("GAIA_DR2 over-ride chosen.  Will use GAIA_DR2 for GAIA source selection.")
            GaiaCat,GaiaCatCols=get_GAIA_objects(RaDecRange,dbh,dbSchema,release='DR2',Timing=True,verbose=2)
        else:
            print("Proceeding to query for sources in GAIA_EDR3.")
            GaiaCat,GaiaCatCols=get_GAIA_objects(RaDecRange,dbh,dbSchema,release='EDR3',Timing=True,verbose=2)
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
#   Pre-Prepare GAIA data for matching
#   Included performing a self-match among GAIA sources (so they can later be flagged)
#
    c2=SkyCoord(ra=GaiaCat['RA']*u.degree,dec=GaiaCat['DEC']*u.degree)
    GaiaCat['GaiaSelfMatch']=np.zeros(GaiaCat['RA'].size,dtype='i4')-1
    if (args.gsm_rad >= 0):
        gsm_idx,gsm_sep,gsm_dist=match_coordinates_sky(c2,c2,nthneighbor=2)
        idx1=np.arange(GaiaCat['RA'].size)
        wsm=np.where(gsm_sep.arcsecond<args.gsm_rad)
        GaiaCat['GaiaSelfMatch'][idx1[wsm]]=gsm_idx[wsm]
    GaiaColFwd=['GAIA_SOURCE_ID','GaiaSelfMatch']

#    Enable to get a sense that self-match was working
#    for i in range(GaiaCat['RA'].size):
#        if (GaiaCat['GaiaSelfMatch'][i]>=0):
#            print(i,GaiaCat['GaiaSelfMatch'][i])

#
#   Pre-Prepare VHS data for matching
#
    if (checkVHS):
        VHSColFwd=['VHS_SOURCEID','K_MAG','K_MAGERR','K_FLAGS']
        c3=SkyCoord(ra=VHSCat['RA2000']*u.degree,dec=VHSCat['DEC2000']*u.degree)
#
    PhotCat,PhotCatCols=get_Y6Gold_objects(RaDecRange,dbh,dbSchema,Timing=True,verbose=verbose)
    c4=SkyCoord(ra=PhotCat['ALPHAWIN_J2000']*u.degree,dec=PhotCat['DELTAWIN_J2000']*u.degree)
  
    PhotCatKeep=[]
    for col in PhotCatCols:
        if (col not in ['ALPHAWIN_J2000','DELTAWIN_J2000']):
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
        print("##################################################")
        print("Working on catalog: {:s}".format(Cat))
        nobj0=ExpCat[Cat][DESColList[0]].size
#
#       Perform matching against GAIA, a photometry catalog, and VHS.
#           - After GAIA match, add to GAIA_STAR flag for cases where a selfmatch was found (then drop the key)
#           - then attempt to isolate further the appropriate portion of stellar locus
#
        c1=SkyCoord(ra=ExpCat[Cat]['ALPHAWIN_J2000']*u.degree,dec=ExpCat[Cat]['DELTAWIN_J2000']*u.degree)

        ExpCat[Cat]=SimplifiedMatch(c1,c2,ExpCat[Cat],GaiaCat,GaiaColFwd,FlagCol='GAIA_STAR',verbose=verbose)
        if (verbose > 0):
            wsm=np.where(ExpCat[Cat]['GAIA_STAR']>0)
            print("Performed match to GAIA. Matched {:d} objects.".format(ExpCat[Cat]['GAIA_STAR'][wsm].size))
        wsm=np.where(np.logical_and(ExpCat[Cat]['GAIA_STAR']>0,ExpCat[Cat]['GaiaSelfMatch']>-1))
        ExpCat[Cat]['GAIA_STAR'][wsm]+=4
        del ExpCat[Cat]['GaiaSelfMatch']
        if (verbose > 0):
            wsm=np.where(ExpCat[Cat]['GAIA_STAR']==1)
            print("Removed GAIA self-match. Matched {:d} objects.".format(ExpCat[Cat]['GAIA_STAR'][wsm].size))

        ExpCat[Cat]=SimplifiedMatch(c1,c4,ExpCat[Cat],PhotCat,PhotCatKeep,FlagCol='PHOT_OBJ',verbose=verbose)
        if (verbose > 0):
            wsm=np.where(ExpCat[Cat]['PHOT_OBJ']>0)
            print("Performed match to photoemtric catalog.  Matched {:d} objects.".format(ExpCat[Cat]['PHOT_OBJ'][wsm].size))
            if ('EXT_MASH' in ExpCat[Cat]):
                wsm=np.where(np.logical_and(ExpCat[Cat]['PHOT_OBJ']>0,ExpCat[Cat]['EXT_MASH']==0))
                print("Performed match to photoemtric catalog.  Matched {:d} objects with EXT_MASH=0.".format(ExpCat[Cat]['PHOT_OBJ'][wsm].size))

        if (checkVHS):
            ExpCat[Cat]=SimplifiedMatch(c1,c3,ExpCat[Cat],VHSCat,VHSColFwd,FlagCol='VHS_OBJ',verbose=verbose)
            if (verbose > 0):
                wsm=np.where(ExpCat[Cat]['VHS_OBJ']>0)
                print("Performed match to VHS.  Matched {:d} objects.".format(ExpCat[Cat]['VHS_OBJ'][wsm].size))
        else:
            if (verbose > 0):
                print("No VHS data available (or not requested).  Creating columns with sentinels.")
            nobj0=ExpCat[Cat][DESColList[0]].size
            ExpCat[Cat]['VHS_SOURCEID']=np.full(nobj0,0,dtype=np.int64)
            ExpCat[Cat]['VHS_OBJ']=np.full(nobj0,0,dtype=np.int32)
            ExpCat[Cat]['K_MAG']=np.full(nobj0,-99.9,dtype=np.float64)
            ExpCat[Cat]['K_MAGERR']=np.full(nobj0,-99.9,dtype=np.float64)
            ExpCat[Cat]['K_FLAGS']=np.full(nobj0,0,dtype=np.int16)

#
#       Remove objects with SExtractor or Image Flags have been thrown
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
        if (GaiaSource['size'].size > 2):
            avg_size,med_size,std_size=pqu.medclip(GaiaSource['size'],verbose=0)
            min_size=avg_size-(3.0*std_size)
            max_size=avg_size+(3.0*std_size)
            max_size=avg_size+(3.0*std_size)
            wsm=np.where(np.logical_and(GaiaSource['size']>min_size,GaiaSource['size']<max_size))
            wsm=np.where(np.logical_and(GaiaSource['size']>min_size,GaiaSource['size']<max_size))
            max_sn=np.amax(GaiaSource['sn'][wsm])
            max_flux_cut=np.amax(GaiaSource['flux'][wsm])
            print("Peak flux currently: {:6.3e} ".format(max_flux_cut))
#
            if (max_sn > SNdict['top_sn_thresh']):
                max_flux_cut=10.0**((-2.5*np.log10(max_flux_cut)+SNdict['remove_top_sn'])/-2.5)
                print("   Peak flux cut to: {:6.3e} ".format(max_flux_cut))
#
#           Flag GAIA matches that were outside range of normal sizes
#
            wsm=np.where(np.logical_and(ExpCat[Cat]['GAIA_STAR']==1,
                         np.logical_or(ExpCat[Cat]['FLUX_RADIUS']>max_size,ExpCat[Cat]['FLUX_RADIUS']<min_size)))
            ExpCat[Cat]['GAIA_STAR'][wsm]+=2
#            print("size: {:.2f} {:.2f}  {:d}".format(min_size,max_size,ExpCat[Cat]['GAIA_STAR'][wsm].size))
#
#           Remove all objects that exceed the limit established by GAIA
#           (performed as keep all objects with lower S/N
#
            wsm=np.where(ExpCat[Cat]['FLUX_AUTO']<max_flux_cut)
            for key in ExpCat[Cat]:
               ExpCat[Cat][key]=ExpCat[Cat][key][wsm]
            nobj_highSN=ExpCat[Cat][DESColList[0]].size
            print("  High S/N cut further reduced catalog from {:d} to {:d} objects".format(nobj_flag,nobj_highSN))
            AccumSize=AccumSize+nobj_highSN 
        else:
            print("Warning: Insufficient GAIA matches for statistics.  Proceeding with {:d} GAIA sources.".format(GaiaSource['size'].size))
            AccumSize=AccumSize+nobj_flag

#
#       Make sure magnitude columns conform to sentinels being -99.9
#
        for mag in ['G','R','I','Z','Y','K']:
            col='{:s}_MAG'.format(mag)
            if (col in ExpCat[Cat]):
                wsm= np.where(np.logical_or(ExpCat[Cat][col]<0.01,ExpCat[Cat][col]>35.0))
                ExpCat[Cat][col][wsm]=-99.9

#       FINISHED LOOPING OVER CATS

#
#   Make the QA plots.
#
    if (args.qa_select is not None):

        AccumData={}
        for Col in ['SN','SPREAD_MODEL','FLUX_RADIUS','K_MAG','R_MAG','Z_MAG']:
            AccumData[Col]=np.zeros(AccumSize,dtype='f8')
        for Col in ['GAIA_STAR','PHOT_OBJ','VHS_OBJ','EXT_MASH']:
            AccumData[Col]=np.zeros(AccumSize,dtype='i4')

        ctr_all=0
        for Cat in grp_list:
            key_array=ExpCat[Cat]['FLUX_AUTO']/ExpCat[Cat]['FLUXERR_AUTO']
            ks=key_array.size
            AccumData['SN'][ctr_all:ctr_all+ks]=key_array
            for Col in ['SPREAD_MODEL','FLUX_RADIUS','K_MAG','R_MAG','Z_MAG','GAIA_STAR','PHOT_OBJ','VHS_OBJ','EXT_MASH']:
                if (Col in ExpCat[Cat]):
                    key_array=ExpCat[Cat][Col]
                    AccumData[Col][ctr_all:ctr_all+ks]=key_array
            ctr_all+=ks

        qa.plot_selection('{:s}'.format(args.qa_select),AccumData)

#
#   Form photometry columns for output catalogs 
#

    color_dict={
        'GI_COLOR':{'bit':1,'num':'G','den':'I','sentinel':1.3,'blue_lim':0.0,'red_lim':3.5, 'bbit':2, 'rbit':4},
        'IZ_COLOR':{'bit':8,'num':'I','den':'Z','sentinel':0.2,'blue_lim':0.0,'red_lim':0.7, 'bbit':16, 'rbit':32}
    }
    next_bit=64
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
        print(color_dict[col])

#   Mag based uncertainty:    
#   m=-2.5*log(fnu/(1.0e-26*zpt))/log(10.0)
#   dm=1.0857*dfnu/fnu
#   S/N= 1.0857/dm

 
    for Cat in grp_list:
#
#       Add columns (fill with sentinels and flag all)
#
        nobj=ExpCat[Cat][DESColList[0]].size
        ExpCat[Cat]['FLAG_COLOR']=np.zeros(nobj,dtype='i4')
        for col in color_dict:
            num=color_dict[col]['num']
            den=color_dict[col]['den']
#
#           Start by setting value for color to default and throwing the bit
#
            ExpCat[Cat][col]=np.full(nobj,color_dict[col]['sentinel'],dtype='f8')
            ExpCat[Cat]['FLAG_COLOR']+=color_dict[col]['bit']

            b_mag='{:s}_MAG'.format(num)
            b_err='{:s}_MAGERR'.format(num)
            b_flag='{:s}_FLAGS'.format(num)
            r_mag='{:s}_MAG'.format(den)
            r_err='{:s}_MAGERR'.format(den)
            r_flag='{:s}_FLAGS'.format(den)
##
##          If we wanted S/N cuts...
##
#            s_num=ExpCat[Cat][b_mag]
#            n_num=ExpCat[Cat][b_err]
#            sn_num = 1.0857*np.divide(1.0, n_num, out=np.zeros_like(s_num), where=n_num!=0)
#            s_den=ExpCat[Cat][r_mag]
#            n_den=ExpCat[Cat][r_err]
#            sn_den = 1.0857*np.divide(1.0, n_den, out=np.zeros_like(s_den), where=n_den!=0)
#
#           cflag --> Check that there are no flags set on MAG measurements
#           Later combine cflag with the fact that photometry exists 
#           Also make sure that MAG values were not set to seninels.
#
            cflag=ExpCat[Cat][b_flag]+ExpCat[Cat][r_flag]
            wsm=np.where(np.logical_and(
                            np.logical_and(ExpCat[Cat][b_mag]>-99.,ExpCat[Cat][r_mag]>-99.),
                            np.logical_and(ExpCat[Cat]['PHOT_OBJ']==1,cflag==0)))
#
#           Calculate color and then unset FLAG_COLOR
#
            ExpCat[Cat][col][wsm]=ExpCat[Cat][b_mag][wsm]-ExpCat[Cat][r_mag][wsm]
            ExpCat[Cat]['FLAG_COLOR'][wsm]-=color_dict[col]['bit']
#
#           Code below could handle red/blue drop-outs but current implementation is a little simplistic
#
##
##           Handle blue drop outs (Blue color does not exist but red does)
##
#            wsm=np.where(np.logical_and(
#                            np.logical_and(ExpCat[Cat][b_mag]<-99.,ExpCat[Cat][r_mag]>-99.),
#                            np.logical_and(ExpCat[Cat]['PHOT_OBJ']==1,ExpCat[Cat]['{:s}_FLAGS'.format(den)]==0)))
#            ExpCat[Cat][col][wsm]=color_dict[col]['red_lim']
#            ExpCat[Cat]['FLAG_COLOR'][wsm]-=color_dict[col]['bit']
#            ExpCat[Cat]['FLAG_COLOR'][wsm]+=color_dict[col]['rbit']
##
##           Handle red drop outs (Red color does not exist but blue does)
##
#            wsm=np.where(np.logical_and(
#                            np.logical_and(ExpCat[Cat][b_mag]>-99.,ExpCat[Cat][r_mag]<-99.),
#                            np.logical_and(ExpCat[Cat]['PHOT_OBJ']==1,ExpCat[Cat]['{:s}_FLAGS'.format(num)]==0)))
#            ExpCat[Cat][col][wsm]=color_dict[col]['blue_lim']
#            ExpCat[Cat]['FLAG_COLOR'][wsm]-=color_dict[col]['bit']
#            ExpCat[Cat]['FLAG_COLOR'][wsm]+=color_dict[col]['bbit']

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


