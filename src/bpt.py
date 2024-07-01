import warnings
import numpy as np
import pandas as pd
import os

from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.table import Table

#Create df with bpt inputs from hdu
def create_df(hdu):
    tmp_dict = {} 
    tmp_dict["ha6563"] = hdu[30].data[23].byteswap().newbyteorder()
    tmp_dict["hb4861"] = hdu[30].data[14].byteswap().newbyteorder()

    tmp_dict["oiii5007"] = hdu[30].data[16].byteswap().newbyteorder()
    tmp_dict["nii6584"] = hdu[30].data[24].byteswap().newbyteorder()

    tmp_dict["e_ha6563"] = 1 / np.sqrt((hdu[31].data[23]).byteswap().newbyteorder())

    tmp_dict["e_nii6584"] = 1/np.sqrt((hdu[31].data[24]).byteswap().newbyteorder())

    tmp_dict['N2'] = np.log10(np.divide(tmp_dict['nii6584'], tmp_dict['ha6563']))
    tmp_dict['R3'] = np.log10(np.divide(tmp_dict['oiii5007'], tmp_dict['hb4861']))

    df = pd.DataFrame({'N2':tmp_dict['N2'].flatten(), \
        'R3': tmp_dict['R3'].flatten(), \
        'NII': tmp_dict['nii6584'].flatten(), \
        'Halpha': tmp_dict['ha6563'].flatten(), \
        'NII_err': tmp_dict['e_nii6584'].flatten(), \
        'Halpha_err': tmp_dict['e_ha6563'].flatten()})
    df['tag'] = df.index

    #Filter out entries on measurement error
    df = df[np.abs(df['NII']) >= np.abs(df['NII_err']*2)]
    df = df[np.abs(df['Halpha']) >= np.abs(df['Halpha_err']*3)]

    return df

#Create the lines to be drawn for BPT
def get_bpt_lines():
    bptLineDict = {}
    bptLineDict['bptline1x'] = np.linspace(-2.0,-0.1,100)
    bptLineDict['bptline1y'] = 0.438/(bptLineDict['bptline1x'] + 0.023) + 1.222
    bptLineDict['bptline2y'] = np.linspace(-0.65,0.9,100)
    bptLineDict['bptline2x'] = -0.39 * bptLineDict['bptline2y']**4 \
        - 0.582*bptLineDict['bptline2y']**3 \
        - 0.637*bptLineDict['bptline2y']**2-0.048*bptLineDict['bptline2y']-0.119
    bptLineDict['bptline3x'] = np.linspace(-0.24,0.5,100)
    bptLineDict['bptline3y'] = 0.95*bptLineDict['bptline3x'] +0.56

    return bptLineDict

#Add bpt labels to given df
def get_bpt_labels(dffin):
    # Creating the BPT demarcated Dataframes
    cut1 = dffin[np.where((-0.39 * dffin['R3']**4 - 0.582*dffin['R3']**3 - 0.637*dffin['R3']**2-0.048*dffin['R3']-0.119) >dffin['N2'] ,True,False)]
    cut1_1 = cut1[np.where((0.438/(cut1['N2'] + 0.023) + 1.222)<cut1['R3'],True,False)]
    BPT_composites = cut1_1[cut1_1['N2'] >-1.4]
    cut2 = dffin[np.where((0.438/(dffin['N2'] + 0.023) + 1.222)>dffin['R3'],True,False)]
    BPT_SFGs = cut2[np.where(cut2['N2']<-0.1,True,False)]
    cut3 = dffin[np.invert(dffin.index.isin(BPT_composites._append(BPT_SFGs).index))]
    BPT_LIERs = cut3[np.where((cut3['R3'] < (0.95*cut3['N2'] +0.56)),True,False)]
    BPT_AGNs = cut3[np.where((cut3['R3'] > (0.95*cut3['N2'] +0.56)),True,False)]

    #Maps BPT classification onto final dataframe
    dffin.loc[dffin.tag.isin(BPT_AGNs.tag),'BPT_class'] = 0
    dffin.loc[dffin.tag.isin(BPT_SFGs.tag),'BPT_class'] = 1
    dffin.loc[dffin.tag.isin(BPT_composites.tag),'BPT_class'] = 2
    dffin.loc[dffin.tag.isin(BPT_LIERs.tag),'BPT_class'] = 3

    return dffin

#Plot bpt for given df
def plot_bpt(dffin):
    plt.scatter(dffin['N2'],dffin['R3'], c=dffin['BPT_class'])#label='AGNS',s=1,c='C0')

    bptLines = get_bpt_lines()
    plt.plot(bptLines['bptline1x'],bptLines['bptline1y'],c='k')
    plt.plot(bptLines['bptline2x'],bptLines['bptline2y'],c='k')
    plt.plot(bptLines['bptline3x'],bptLines['bptline3y'],c='k')

    plt.xlim(-2.0,0.5)
    plt.ylim(-1.0,1.5)
    plt.show()

def get_file_path(base_path, df, i):
    plate = int((df['plate'][i]))
    ifu = int((df['ifudsgn'][i]))
    path = base_path + f'{plate}/{ifu}/'
    file = path + f'manga-{plate}-{ifu}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'
    return file

def get_file_path_str(base_path, plateifu):
    plateifu_split = plateifu.split('-')
    plate = int((plateifu_split[0]))
    ifu = int((plateifu_split[1]))
    path = base_path + f'{plate}/{ifu}/'
    file = path + f'manga-{plate}-{ifu}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'
    return file

def read_fits(fits_path, hdu=1):
    dat = Table.read(fits_path, format='fits', hdu=hdu)
    names = [name for name in dat.colnames if len(dat[name].shape) <= 1]
    df = dat[names].to_pandas()
    if 'MANGAID' in df:
        df['MANGAID'] = df['MANGAID'].str.decode('utf-8')
        df['MANGAID'] = df['MANGAID'].str.strip()
    return df

def get_spaxel_bpt_percent(file):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, \
            message="(invalid value|divide by zero) encountered in (log10|divide)")

        with fits.open(file) as hdu:
            tmp_df = create_df(hdu)

            label_percentages = pd.DataFrame()
            #label_percentages[1.0] = int(0)

            if not tmp_df.empty:
                df_labeled = get_bpt_labels(tmp_df)

                #Get count and % of each label type
                label_counts = df_labeled['BPT_class'].value_counts()
                label_percentages = (label_counts / len(df_labeled)) * 100
                label_percentages['bpt_error'] = 0
            
            else:
                label_percentages = pd.Series({1.0: 0, 2.0: 0, 3.0: 0, 0.0: 0, 'bpt_error': 1})


            labels = [1.0, 2.0, 3.0, 0.0]

            #Fill in 0 if no spaxels in category
            for l in labels:
                if l not in label_percentages:
                    label_percentages[l] = 0

            return label_percentages

        #   #Plot bpt figure 
        #   #plot_bpt(df_labeled)

#Take df with 'plateifu' column and get bpt spaxel data
def prepare_bpt_data(df):
    base_path = 'https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
    cache_path = './data/_cached_bpt.csv'
    
    bpt_df = None
    bpt_data = []

    #cached_bpt_df = None

    if os.path.exists(cache_path):
        print("BPT Cache hit! Reading csv...")
        bpt_df = pd.read_csv(cache_path)


    #TODO: Figure out merging cache and new data. pandas is being miserable 
    #print("BPT Cache miss! Downloading files...")
    #for i in range(len(df['plateifu'])):
    else:
        #for i in range(20):
        for i in range(len(df['plateifu'])):

            #if cached_bpt_df is not None:
            #    plateid = df['plateifu'][i]

            #    #TODO: Potentially retry downloading entries that errored out before
            #    if cached_bpt_df['plateifu'].str.contains(plateid).any():
            #        #and int(cached_bpt_df[cached_bpt_df['plateifu'] == plateid].iloc[0]['bpt_error']) == 0:
            #        continue
            #    else:
            #        print("Cache Miss!", plateid)


            file = get_file_path_str(base_path, df['plateifu'][i])
            print("Processing", i, ':' , file)

            try:
                bpt_class = get_spaxel_bpt_percent(file)

                bpt_class['plateifu'] = df['plateifu'][i]
                print(bpt_class)

                bpt_data_row = {'plateifu': df['plateifu'][i], 
                    1.0: bpt_class[1.0],
                    2.0: bpt_class[2.0],
                    3.0: bpt_class[3.0],
                    0.0: bpt_class[0.0],
                    'bpt_error': bpt_class['bpt_error'],
                    }

            except:
                bpt_data_row = {'plateifu': df['plateifu'][i], 
                    1.0: 0,
                    2.0: 0,
                    3.0: 0,
                    0.0: 0,
                    'bpt_error': 1,
                    }

            bpt_data.append(bpt_data_row)

            bpt_df = pd.DataFrame(bpt_data)
            bpt_df.to_csv(cache_path, index=False)

    #print('---')
    #print(bpt_df)
    #print('+++')
    #print(cached_bpt_df)


    #print('---')
    #bpt_df = pd.concat([bpt_df, cached_bpt_df], ignore_index=True)
    #bpt_df = pd.merge(bpt_df, cached_bpt_df, on='plateifu', how='outer')
    #bpt_df = bpt_df.combine_first(cached_bpt_df)
    #bpt_df = (bpt_df.merge(cached_bpt_df, left_on='plateifu', right_on='plateifu', ignore_index=True))

    #print(bpt_df)


    assert bpt_df is not None

    print(bpt_df)

    return bpt_df


def main():
    base_path = 'https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'

    df_drpall = (read_fits("./data/drpall-v3_1_1.fits"))

    for i in range(5):
        file = get_file_path(base_path, df_drpall, i)
        print("Processing: ", file)

        print(get_spaxel_bpt_percent(file))


if __name__ == "__main__":
    main()