#!/bin/bash

#src_dir="/dat1/JJ_PRDCT/work/nagatani/test_data4/N11E003"
src_dir="/dat1/JJ_PRDCT/work/nagatani/test_data3/S08W075"
fnfmsk_dir="$src_dir"
srtmmsk_dir="$src_dir"
demmsk_dir="$src_dir"

dst_dir=$(pwd)

stt_infile="S08W075_20161124_sl_HV.tif"     # reference data
end_infile="S08W075_20170105_sl_HV.tif"     # latest data
mskfile1="S08W075_10_C.tif"                 # FNF mask (1:forest, 2:non-forest, 3:water)
mskfile2="S08W075_slp.tif"                  # SRTM slope
mskfile3="S08W075_20161124_MASK.tif"        # MASK for reference data
mskfile4="S08W075_20170105_MASK.tif"        # MASK for latest data

gadm28_0="../gadm28/gadm28_adm0.shp"        # gadm28_0
gadm28_2="../gadm28/gadm28_adm2.shp"        # gadm28_2

outdir=${dst_dir}

#-- exec python ---
time ./jjfastexec.py \
           ${src_dir}/$stt_infile \
           ${src_dir}/$end_infile \
           ${fnfmsk_dir}/$mskfile1 \
           ${srtmmsk_dir}/$mskfile2 \
           ${demmsk_dir}/$mskfile3 \
           ${demmsk_dir}/$mskfile4 \
           ${src_dir}/$gadm28_0 \
           ${src_dir}/$gadm28_2 \
           $outdir
