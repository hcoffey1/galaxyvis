from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

#from mangadap.dapfits import DAPCubeBitMask
#bm = DAPCubeBitMask()

tmp_fits="./data/tmp/manga-7443-12703-LOGCUBE-HYB10-MILESHC-MASTARHC2.fits"
#tmp_fits="./data/tmp/manga-8547-12704-LOGCUBE-HYB10-MILESHC-MASTARHC2.fits"
#tmp_fits="./data/tmp/manga-11754-12705-LOGCUBE-HYB10-MILESHC-MASTARHC2.fits"
# Open the FITS file
hdul = fits.open(tmp_fits)

i = 7 

print(hdul['WAVE'].data)
print(hdul[i].header['EXTNAME'])
#print(hdul[1].data)
#for i in range(0,4563):
#avgData = np.mean(hdul[i].data, axis=0)
#plt.imshow(avgData, cmap='viridis')
#plt.colorbar()
#plt.show()

avgMaskData = np.mean(hdul['MODEL_MASK'].data, axis=2)
avgMaskData = np.mean(avgMaskData, axis=1)

print(len(hdul['WAVE'].data))
avgFluxData = np.mean(hdul['FLUX'].data, axis=2)
avgFluxData = np.mean(avgFluxData, axis=1)
#avgFluxData = np.mean(hdul['FLUX'].data)
print(len(avgFluxData))
plt.plot(hdul['WAVE'].data, avgFluxData)
plt.show()


#maskedFluxData = np.ma.MaskedArray(avgFluxData, mask=bm.flagged(avgMaskData, ['IGNORED']))
