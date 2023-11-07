#from marvin import config
#print(config.access)

from astropy.io import fits
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt

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

zoo_dat = Table.read(fits_file_path, format='fits')
zoo_df = zoo_dat.to_pandas()

swift_dat = Table.read(swift_fits_path, format='fits')
names = [name for name in swift_dat.colnames if len(swift_dat[name].shape) <= 1]
swift_df = swift_dat[names].to_pandas()
#swift_df = swift_dat.to_pandas()

#with fits.open(fits_file_path) as hdul:
#    # Get the primary header
#    header = hdul[1].header
#    print(hdul)
#    
#    # Access the data (assuming it's in the primary HDU)
#    data = hdul[1].data


#df = pd.DataFrame(data)
print(zoo_df)
print(swift_df)

print("Trying merge")

merge_df = (zoo_df.merge(swift_df, left_on='MANGAID', right_on='MANGAID'))

merge_df.to_csv('merged.csv', index=False)

#Remove outliers
merge_df = merge_df[merge_df['SFR_1RE'] >= -10]
merge_df = merge_df.dropna(axis=1)

#Remove non-numeric data
non_numeric_columns = merge_df.select_dtypes(include=['object']).columns
numeric_df = merge_df.drop(columns=non_numeric_columns)


print(numeric_df)
debiased_columns = [col for col in numeric_df.columns if "debiased" in col]
#for c in (debiased_columns):
#    print(c)
#exit(1)

#Remove extra data values
#dropped_columns=['nsa_id']
#numeric_df = numeric_df.drop(columns=dropped_columns)
numeric_df = numeric_df[debiased_columns]


print("Numeric data----------")
print(numeric_df)

#Scale data before PCA
scaler = StandardScaler()
scaled_df = scaler.fit_transform(numeric_df)

#PCA
pca = PCA(n_components=2)
pca_df = pd.DataFrame(pca.fit_transform(scaled_df), columns=['pc1', 'pc2'])

print(merge_df)
print(pca_df)

#Merge PCA back into original data
merge_df = merge_df.reset_index(drop=True)
pca_df = pca_df.reset_index(drop=True)
merge_df = pd.concat([merge_df, pca_df], axis=1, ignore_index=False)

print(merge_df)
x = merge_df['pc1']
y = merge_df['pc2']
c = merge_df['SFR_1RE']
#x = merge_df['T04_SPIRAL_A08_SPIRAL_WEIGHT'.lower()]
#y = merge_df['SFR_1RE']


plt.scatter(x,y,c=c, cmap='viridis')
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


