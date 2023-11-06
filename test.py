#from marvin import config
#print(config.access)

from astropy.io import fits
from astropy.table import Table
import pandas as pd

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

print(zoo_df.merge(swift_df, left_on='MANGAID', right_on='MANGAID'))

df1 = zoo_df
df2 = swift_df 

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


