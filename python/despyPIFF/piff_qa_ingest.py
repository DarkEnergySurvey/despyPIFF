#!/usr/bin/env python3

"""
Ingestion function(s) for PIFF QA
"""

#################################
def ingest_piff_qa(data,QATab,StarTab,dbh,schema,verbose=0):
    """
    Input:
        data:       QA result dictionary
        qatab:      Main QA table (for values rolled up by CCD)
        startab:    QA table to hold NGMIX results from individual input stars
        dbh:        DB connection
        Schema:     DB Schema (to allow for overrides)
        verbose:    Verbosity level
    Outputs:
        n1_insert   Number of rows inserted in QATAB
        n2_insert   Number of rows inserted in STARTAB
    """    

    n1_insert=0
    n2_insert=0
    print("Preparing to ingest data")
#
#
#
    CatList=[]
    for Cat in data:
        if (Cat not in ['outland']):
            CatList.append(Cat)

#
#   First the MODEL_QA table (one row per CCD)
#   
    DBorder_QA=['FILENAME','EXPNUM','CCDNUM','NSTAR','NREMOVED','CHISQ','DOF','FWHM_CEN','FRAC_CEN_OUT','FRAC_WIDTH_OUT',
                'STAR_E1_MEAN','STAR_E1_STD','STAR_E2_MEAN','STAR_E2_STD','STAR_T_MEAN','STAR_T_STD','STAR_NFIT',
                'MODEL_E1_MEAN','MODEL_E1_STD','MODEL_E2_MEAN','MODEL_E2_STD','MODEL_T_MEAN','MODEL_T_STD','MODEL_NFIT',
                'FLAG']

    new_data=[] 
    for Cat in CatList:
        new_row=[]
        filename=Cat.split("/")[-1]
        new_row.append(filename)
        for key in DBorder_QA[1:]:
            if (key == 'FILENAME'):
                print("WTF!!!")
            if (key == 'FWHM_CEN'):
                new_row.append(data[Cat]['fwhm'])
            else:
                new_row.append(data[Cat][key.lower()])
#                print(key,type(data[Cat][key.lower()]))
        new_data.append(new_row)

    dbh.insert_many(schema+QATab,DBorder_QA,new_data)
    dbh.commit()
    n1_insert=len(new_data)
    print("Commit {:d} rows to {:s}".format(n1_insert,schema+QATab))

#
#   Now the STAR_QA table (one row per star per CCD)
#   
    DBorder_STAR_QA=['FILENAME','STAR_NUMBER','EXPNUM','CCDNUM','X','Y','RA','DEC','FLUX','SNR',
                     'STAR_E1','STAR_E2','STAR_T','STAR_FLAG','MODEL_E1','MODEL_E2','MODEL_T','MODEL_FLAG',
                     'HPIX_64','HPIX_16384','HPIX_65536']
    new_data=[]
    for Cat in CatList:
        filename=Cat.split("/")[-1]
        expnum=data[Cat]['expnum']
        ccdnum=data[Cat]['ccdnum']

        for i in range(data[Cat]['star_data']['x'].size):
            new_row=[filename,i+1,expnum,ccdnum]
            for col in ['x','y','ra','dec','flux','snr','s_e1','s_e2','s_T','s_flag','m_e1','m_e2','m_T','m_flag','hpix_64','hpix_16384','hpix_65536']:
                if (col in ['s_flag','m_flag','hpix_64','hpix_16384','hpix_65536']):
                    new_row.append(int(data[Cat]['star_data'][col][i]))
                else:
                    new_row.append(data[Cat]['star_data'][col][i])
            new_data.append(new_row)
    
    dbh.insert_many(schema+StarTab,DBorder_STAR_QA,new_data)
    dbh.commit()
    n2_insert=len(new_data)
    print("Commit {:d} rows to {:s}".format(n2_insert,schema+StarTab))

    return n1_insert,n2_insert
