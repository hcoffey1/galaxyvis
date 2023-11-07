#from marvin import config
#print(config.access)

from astropy.io import fits
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Define the path to your FITS file
#fits_file_path = "./data/MaNGA_GZD_auto-v1_0_1.fits"
#fits_file_path = "./data/MaNGA_gz-v2_0_1.fits"
fits_file_path = './data/MaNGA_gz-v1_0_1.fits'
#fits_file_path = "./data/MaNGA_gzUKIDSS-v1_0_1.fits"

swift_fits_path="./data/SwiM_all_v4.fits"
#swift_fits_path="./data/drpall-v3_1_1.fits"
# Open the FITS file

def read_fits(fits_path):
    dat = Table.read(fits_path, format='fits')
    names = [name for name in dat.colnames if len(dat[name].shape) <= 1]
    df = dat[names].to_pandas()
    return df

def clean_df(df):
    #Remove outliers
    df = df[merge_df['SFR_1RE'] >= -10] #May cause a crash
    df = df.dropna(axis=1)
    return df

def get_numeric_df(df):
    #Remove non-numeric data
    non_numeric_columns = df.select_dtypes(include=['object']).columns
    df = df.drop(columns=non_numeric_columns)
    return df

zoo_df = read_fits(fits_file_path)
swift_df = read_fits(swift_fits_path)

merge_df = (zoo_df.merge(swift_df, left_on='MANGAID', right_on='MANGAID'))

#Remove outliers
merge_df = clean_df(merge_df)
#Remove non-numeric data
numeric_df = get_numeric_df(merge_df)

#Select debiased columns
debiased_columns = [col for col in numeric_df.columns if "debiased" in col]
#for c in (debiased_columns):
#    print(c)
numeric_df = numeric_df[debiased_columns]

print("Numeric data----------")
print(numeric_df)

#Scale data before PCA
scaler = StandardScaler()
scaled_df = scaler.fit_transform(numeric_df)

#PCA
pca = PCA(n_components=2)
pca_df = pd.DataFrame(pca.fit_transform(scaled_df), columns=['pc1', 'pc2'])
tsne_model = TSNE(n_components=2, perplexity=20, random_state=42)
tsne_df = pd.DataFrame(tsne_model.fit_transform(scaled_df), columns=['tsne1', 'tsne2'])

print(merge_df)
print(pca_df)

#Merge PCA back into original data
merge_df = merge_df.reset_index(drop=True)
pca_df = pca_df.reset_index(drop=True)
tsne_df = tsne_df.reset_index(drop=True)
merge_df = pd.concat([merge_df, pca_df, tsne_df], axis=1, ignore_index=False)

print(merge_df)
#x1 = merge_df['pc1']
#y1 = merge_df['pc2']
c = merge_df['t04_spiral_a08_spiral_debiased']
#c = merge_df['SFR_1RE']

x2 = merge_df['tsne1']
y2 = merge_df['tsne2']
#plt.scatter(x1,y1,c=c, cmap='viridis')
#cbar = plt.colorbar()
#cbar.set_label('SFR')
#plt.show()
plt.scatter(x2,y2,c=c, cmap='viridis')
cbar = plt.colorbar()
cbar.set_label('SFR')
plt.show()

# Check for overlap in the 'ID' column
#shared_column = 'MANGAID'
#overlap_df = df1[df1[shared_column].isin(df2[shared_column])]
#
#if not overlap_df.empty:
#    print(f"There are {len(overlap_df)} rows overlapping in the '{shared_column}' column.")
#    print(overlap_df)
#else:
#    print(f"There are no overlapping rows in the '{shared_column}' column.")
#print(pd.concat([zoo_df, swift_df], axis=0))

#Print the header information
#print("Header Information:")
#print(header)
#print(data)

# Now you can work with the data as a NumPy array
# For example, you can print the shape of the data array
#print("Data shape:", data.shape)


