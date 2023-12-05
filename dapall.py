import numpy as np
from astropy.io import fits

dapall = fits.open('data/dapall-v3_1_1-3.1.0.fits')['HYB10-MILESHC-MASTARSSP'].data
indx = dapall['DAPDONE'] == 1
tbdata = dapall[indx]

print(dapall['ELS01'])
exit(1)
import matplotlib.pyplot as plt

plt.scatter(np.ma.log10(tbdata['ha_gsigma_1re']),
               np.ma.log10(tbdata['stellar_sigma_1re']),
               alpha=0.5, marker='.', s=30, lw=0)
plt.xlim(1,3.2)
plt.ylim(0.8,3)
plt.xlabel(r'ionized-gas $\sigma$ within 1 $R_e$ [km/s]')
plt.ylabel(r'stellar $\sigma$ within 1 $R_e$ [km/s]')
plt.show()
