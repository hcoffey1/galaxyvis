# Imports
import sys
import os
import numpy as np
import pandas as pd
from astropy.io import fits
import astropy.coordinates as acoords
from astropy.io import fits
import glob
from matplotlib import pyplot as plt
from matplotlib import colormaps as cm
np.set_printoptions(suppress=True)
import time
#from IPython import embed
import astropy.constants
from astropy.table import Table
#from mangadap.tests.util import drp_test_version
#from mangadap.datacube import MaNGADataCube
#from mangadap.config import manga
#from mangadap.util.sampling import spectral_coordinate_step
#from mangadap.util.resolution import SpectralResolution
#from mangadap.util.pixelmask import SpectralPixelMask
#from mangadap.par.artifactdb import ArtifactDB
#from mangadap.par.emissionmomentsdb import EmissionMomentsDB
#from mangadap.par.emissionlinedb import EmissionLineDB
#from mangadap.par.absorptionindexdb import AbsorptionIndexDB
#from mangadap.par.bandheadindexdb import BandheadIndexDB
#from mangadap.proc.emissionlinemoments import EmissionLineMoments
#from mangadap.proc.sasuke import Sasuke
#from mangadap.proc.ppxffit import PPXFFit
#from mangadap.proc.stellarcontinuummodel import StellarContinuumModel, StellarContinuumModelBitMask
#from mangadap.proc.emissionlinemodel import EmissionLineModelBitMask
#from mangadap.proc.spectralfitting import EmissionLineFit
#from mangadap.proc.spectralindices import SpectralIndices
#from mangadap.util.sampling import angstroms_per_pixel
#from mangadap.util.filter import interpolate_masked_vector
#from mangadap.proc.templatelibrary import TemplateLibrary, TemplateLibraryDef
#from mangadap.config import defaults
#from mangadap.proc.bandpassfilter import emission_line_equivalent_width
#quick cell stopper for debugging
class StopExecution(Exception):
    def _render_traceback_(self):
        return []

def process_df(df):
    #df['OII'] = df['OII_3726']+ df['OII_3729']
    #df['OII_err'] = np.sqrt(df['OII_3726_err']**2+ df['OII_3729_err']**2)
    #df['OII_ew'] = df['OII_3726_ew']+ df['OII_3729_ew']


    df['N2'] = np.log10(np.divide(df['nii6584'],df['ha6563']))
    df['R3'] = np.log10(np.divide(df['oiii5007'],df['hb4861']))

    #df['R2'] = np.log10(np.divide(df['OII'],df['Hbeta']))
    #df['Tr2'] = np.log10(df['OII_ew']+df['Hbeta_ew'])

def get_bpt_lines():
    bptLineDict = {}
    #creating the lines to be drawn for BPT
    bptLineDict['bptline1x'] = np.linspace(-2.0,-0.1,100)
    bptLineDict['bptline1y'] = 0.438/(bptLineDict['bptline1x'] + 0.023) + 1.222
    bptLineDict['bptline2y'] = np.linspace(-0.65,0.9,100)
    bptLineDict['bptline2x'] = -0.39 * bptLineDict['bptline2y']**4 \
        - 0.582*bptLineDict['bptline2y']**3 \
        - 0.637*bptLineDict['bptline2y']**2-0.048*bptLineDict['bptline2y']-0.119
    bptLineDict['bptline3x'] = np.linspace(-0.24,0.5,100)
    bptLineDict['bptline3y'] = 0.95*bptLineDict['bptline3x'] +0.56

    return bptLineDict

def get_bpt_demarc_df(dffin):
    # Creating the BPT demarcated Dataframes
    cut1 = dffin[np.where((-0.39 * dffin['R3']**4 - 0.582*dffin['R3']**3 - 0.637*dffin['R3']**2-0.048*dffin['R3']-0.119) >dffin['N2'] ,True,False)]
    cut1_1 = cut1[np.where((0.438/(cut1['N2'] + 0.023) + 1.222)<cut1['R3'],True,False)]
    BPT_composites = cut1_1[cut1_1['N2'] >-1.4]
    cut2 = dffin[np.where((0.438/(dffin['N2'] + 0.023) + 1.222)>dffin['R3'],True,False)]
    BPT_SFGs = cut2[np.where(cut2['N2']<-0.1,True,False)]
    cut3 = dffin[np.invert(dffin.index.isin(BPT_composites._append(BPT_SFGs).index))]
    BPT_LIERs = cut3[np.where((cut3['R3'] < (0.95*cut3['N2'] +0.56)),True,False)]
    BPT_AGNs = cut3[np.where((cut3['R3'] > (0.95*cut3['N2'] +0.56)),True,False)]

#def map_bpt_to_df():
    #Maps BPT classification onto final dataframe
    dffin.loc[dffin.tag.isin(BPT_AGNs.tag),'BPT_class'] = 0
    dffin.loc[dffin.tag.isin(BPT_SFGs.tag),'BPT_class'] = 1
    dffin.loc[dffin.tag.isin(BPT_composites.tag),'BPT_class'] = 2
    dffin.loc[dffin.tag.isin(BPT_LIERs.tag),'BPT_class'] = 3

    #TODO: Error bounds
    df = dffin[np.abs(dffin['NII']) >= np.abs(dffin['NII_err']*2)]
    dffin =df[np.abs(df['Halpha']) >= np.abs(df['Halpha_err']*3)]


    bptLines = get_bpt_lines()

    #cmap = plt.colormaps[dffin['BPT_class']]#('viridis', dffin['BPT_class'].nunique())  # Choose the colormap and the number of colors
#    cmap = plt.cm.get_cmap('viridis', dffin['BPT_class'].nunique())
    #num_labels = dffin['BPT_class'].nunique()
    #cmap = plt.cm.get_cmap('viridis', num_labels) 
    cmap = cm.get_cmap('viridis') 

    plt.scatter(dffin['N2'],dffin['R3'], c=dffin['BPT_class'])#label='AGNS',s=1,c='C0')
    plt.plot(bptLines['bptline1x'],bptLines['bptline1y'],c='k')
    plt.plot(bptLines['bptline2x'],bptLines['bptline2y'],c='k')
    plt.plot(bptLines['bptline3x'],bptLines['bptline3y'],c='k')
    plt.xlim(-2.0,0.5)
    plt.ylim(-1.0,1.5)
    plt.show()
    # Define colormap

    # Plot scatterplot with 'X' and 'Y', using 'Label' as color
    #plt.scatter(df['X'], df['Y'], c=df['Label'], cmap=cmap)
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.colorbar(label='Label')
    #plt.show()

    #plot_AGN = dffin[np.where(dffin[plot_tag+'_class'] == 0,True,False)]
    #plot_SFG = plotdffin[np.where(dffin[plot_tag+'_class'] == 1,True,False)]
    #plot_composite = plotdffin[np.where(dffin[plot_tag+'_class'] == 2,True,False)]
    #plot_LIER = plotdffin[np.where(dffin[plot_tag+'_class'] == 3,True,False)]
    #plot_confusion = plotdffin[np.where(plotdffin[plot_tag+'_class'] == 4,True,False)]
    #plot_error = plotdffin[np.where(plotdffin[plot_tag+'_class'] == -1,True,False)]
    #plot_NeVAGN = plotdffin[np.abs(plotdffin['NeV']) >= np.abs(plotdffin['NeV_err']*5)]

    #BPT_zmax1 = np.max(dffin[dffin['Halpha'] != -999.0]['Z'])
    #BPT_zmax2 = np.max(dffin[dffin['NII'] != -999.0]['Z'])
    #BPT_zmax = np.min([BPT_zmax1 ,BPT_zmax2 ])
    #OIII_zmax = np.max(dffin[dffin['OIII'] != -999.0]['Z'])
    #NeV_zmax = np.max(dffin[dffin['NeV'] != -999.0]['Z'])
#file="./data/gz_decals_auto_posteriors.parquet"

def bpt_redshift_correction():
    #redshift corrections to ensure that nothing is getting categorized after lines redshift from sample
    BPT_zcut = dffin[np.where(dffin['Z'] < BPT_zmax,True,False)]
    dffin.loc[np.invert(dffin.tag.isin(BPT_zcut.tag)),'BPT_class'] = -1
    MEx_zcut = dffin[np.where(dffin['Z'] < OIII_zmax,True,False)]
    dffin.loc[np.invert(dffin.tag.isin(MEx_zcut.tag)),'MEx_class'] = -1
    Blue_zcut = dffin[np.where(dffin['Z'] < OIII_zmax,True,False)]
    dffin.loc[np.invert(dffin.tag.isin(Blue_zcut.tag)),'Blue_class'] = -1
    TrEW_zcut = dffin[np.where(dffin['Z'] < OIII_zmax,True,False)]
    dffin.loc[np.invert(dffin.tag.isin(TrEW_zcut.tag)),'TrEW_class'] = -1
    NeV_zcut = dffin[np.where(dffin['Z'] < NeV_zmax,True,False)]
    dffin.loc[np.invert(dffin.tag.isin(NeV_zcut.tag)),'NeV_class'] = -1
    NeV_zcut2 = dffin[np.where(dffin['Z'] > 0.1,True,False)]
    dffin.loc[np.invert(dffin.tag.isin(NeV_zcut2.tag)),'NeV_class'] = -1

def read_fits(fits_path, hdu=1):
    dat = Table.read(fits_path, format='fits', hdu=hdu)
    names = [name for name in dat.colnames if len(dat[name].shape) <= 1]
    df = dat[names].to_pandas()
    if 'MANGAID' in df:
        df['MANGAID'] = df['MANGAID'].str.decode('utf-8')
        df['MANGAID'] = df['MANGAID'].str.strip()
    return df

base_path = 'https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
#zoo_df17_file_path = "./data/MaNGA_gz-v2_0_1.fits"

df = (read_fits("./data/drpall-v3_1_1.fits"))

#for c in df.columns:
    #print(c)
#print(df.columns)
j = 0
limit = 10
for i in range(1):
    plate = int((df['plate'][i]))
    ifu = int((df['ifudsgn'][i]))

    print(plate)
    print(ifu)
    path = base_path + f'{plate}/{ifu}/'
    file = path + f'manga-{plate}-{ifu}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'
    print(file)

    with fits.open(file) as hdu:
        #print(hdu)
        tmp_dict = {} 
        tmp_dict["ha6563"] = hdu[30].data[23].byteswap().newbyteorder()
        tmp_dict["hb4861"] = hdu[30].data[14].byteswap().newbyteorder()

        #tmp_dict["oii3727"] = hdu[30].data[0]
        #tmp_dict["e_oii3727"] = 1 / np.sqrt(hdu[31].data[0])
        #tmp_dict["ew_oii3727"] = hdu[33].data[0]

        #tmp_dict["oii3729"] = hdu[30].data[1]
        #tmp_dict["e_oii3729"] = 1 / np.sqrt(hdu[31].data[1])
        #tmp_dict["ew_oii3729"] = hdu[33].data[1]


        tmp_dict["oiii5007"] = hdu[30].data[16].byteswap().newbyteorder()
        tmp_dict["nii6584"] = hdu[30].data[24].byteswap().newbyteorder()

        #tmp_dict["e_ha6563"] = 1 / np.sqrt(hdu[31].data[23])
        tmp_dict["e_ha6563"] = 1 / np.sqrt((hdu[31].data[23]).byteswap().newbyteorder())

        #tmp_dict["e_hb4861"] = 1 / np.sqrt(hdu[31].data[14])
        #tmp_dict["e_oiii5007"] = 1 / np.sqrt(hdu[31].data[16])

        #tmp_dict["e_nii6584"] = 1 / np.sqrt(hdu[31].data[24])
        tmp_dict["e_nii6584"] = 1/np.sqrt((hdu[31].data[24]).byteswap().newbyteorder())

        #tmp_dict["snr"] = hdu[14].data
        process_df(tmp_dict)

        dfinit = pd.DataFrame({'N2':tmp_dict['N2'].flatten(), \
            'R3': tmp_dict['R3'].flatten(), \
            'NII': tmp_dict['nii6584'].flatten(), \
            'Halpha': tmp_dict['ha6563'].flatten(), \
            'NII_err': tmp_dict['e_nii6584'].flatten(), \
            'Halpha_err': tmp_dict['e_ha6563'].flatten()})
        dfinit['tag'] = dfinit.index

        #print(dfinit)
        get_bpt_demarc_df(dfinit)
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #    print(dfinit)

        #dfinit = pd.DataFrame({'PLATE':plate,'NII':NII,'Halpha':Halpha,'OIII':OIII,'Hbeta':Hbeta,'NII_err':NII_err,'Halpha_err':Halpha_err,'OIII_err':OIII_err,'Hbeta_err':Hbeta_err,'OII_3726':OII_3726,'OII_3729':OII_3729,'OII_3726_err':OII_3726_err,'OII_3729_err':OII_3729_err,'Hbeta_ew':Hbeta_ew,'Z':z,'OII_3726_ew':OII_3726_ew,'OII_3729_ew':OII_3729_ew, 'NeV':NeV,'NeV_err':NeV_err})
        #dfinit = pd.DataFrame({'PLATE':plate, 'NII':tmp_dict['ha6563']})
        #Combine lines into final dataframe

    #print(df[i])