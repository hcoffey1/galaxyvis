#!/bin/bash

mkdir -p data

cd data

#Galaxy Zoo DR 17, small overlap
#wget https://data.sdss.org/sas/dr17/env/MANGA_MORPHOLOGY/galaxyzoo/MaNGA_GZD_auto-v1_0_1.fits
wget https://data.sdss.org/sas/dr17/env/MANGA_MORPHOLOGY/galaxyzoo/MaNGA_gz-v2_0_1.fits
#wget https://data.sdss.org/sas/dr17/env/MANGA_MORPHOLOGY/galaxyzoo/MaNGA_gzUKIDSS-v1_0_1.fits

#Galaxy Zoo DR 15, much more overlap with SWIFT
#wget https://dr15.sdss.org/sas/dr15/manga/morphology/galaxyzoo/MaNGA_gz-v1_0_1.fits

#SWIFT
#wget https://data.sdss.org/sas/dr17/manga/swim/v4.1/SwiM_all_v4.fits

#AGN
#wget https://data.sdss.org/sas/dr17/env/MANGA_AGN/v1_0_1/manga_agn-v1_0_1.fits

#DAPALL
wget https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/dapall-v3_1_1-3.1.0.fits

#Firefly global data
wget https://data.sdss.org/sas/dr17/manga/spectro/firefly/v3_1_1/manga-firefly-globalprop-v3_1_1-mastar.fits

#Galaxy Morphologies
wget https://data.sdss.org/sas/dr17/env/MANGA_MORPHOLOGY/manga_visual_morpho/2.0.1/manga_visual_morpho-2.0.1.fits

#decals galaxy zoo
wget https://zenodo.org/records/4573248/files/gz_decals_auto_posteriors.parquet?download=1
