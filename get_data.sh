#!/bin/bash

mkdir -p data

cd data

#Galaxy Zoo DR 17, small overlap
wget https://data.sdss.org/sas/dr17/env/MANGA_MORPHOLOGY/galaxyzoo/MaNGA_GZD_auto-v1_0_1.fits
wget https://data.sdss.org/sas/dr17/env/MANGA_MORPHOLOGY/galaxyzoo/MaNGA_gz-v2_0_1.fits
wget https://data.sdss.org/sas/dr17/env/MANGA_MORPHOLOGY/galaxyzoo/MaNGA_gzUKIDSS-v1_0_1.fits

#Galaxy Zoo DR 15, much more overlap with SWIFT
wget https://dr15.sdss.org/sas/dr15/manga/morphology/galaxyzoo/MaNGA_gz-v1_0_1.fits

#SWIFT
wget https://data.sdss.org/sas/dr17/manga/swim/v4.1/SwiM_all_v4.fits
