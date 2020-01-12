#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
# $Id$ Jul 5, 2017
#
# Name:     JJFast.py
#                 ver1.0.0
# Class:    JJFast
#
###############################################################################
# Copyright (c) 2017, Izumi Nagatani <nagatani.izumi@jaxa.jp>

#-- set user defined param ----------
minimum_area_ha = "5.0"    #minimum area in ha for filtering polygon
#minimum_area_ha = "4.0"    #minimum area in ha for filtering polygon
#minimum_area_ha = "3.0"    #minimum area in ha for filtering polygon
#minimum_area_ha = "2.0"    #minimum area in ha for filtering polygon
#minimum_area_ha = "1.0"    #minimum area in ha for filtering polygon


TOTAL_PROB_THRESH = 0.1    #probability threshold (0-0.4)

THRESH_SRTM_SLP = 30.000   #threshold for srtm slop

#------------
import sys
import os.path
import time
import math
from osgeo import gdal, gdal_array
from osgeo import ogr
import numpy as np
from skimage import color
import cv2
import pprint
import rasterio
from rasterio import features
from rasterstats import zonal_stats
from affine import Affine
import fiona
import subprocess
from subprocess import check_output
import json
from shapely.geometry import Polygon



# *****************************************************************
# Exception
# *****************************************************************
class JJFastError(Exception):
	'''JJFast Exception'''


# *****************************************************************
#  Create empty output shapefile
# *****************************************************************
def emptyshapefile(obj):

	source_crs = {'no_defs': True, 'ellps': 'WGS84', 'datum': 'WGS84',  'proj': 'longlat'}
	src_schema = {'geometry': 'Polygon', 'properties':{}}

	with fiona.open( obj.outfile, 'w', driver='ESRI Shapefile', crs=source_crs, schema=src_schema ) as c:
		pass

	return 0


# *****************************************************************
# Marge mask files to one image
# *****************************************************************
#def margemask(ds_FNFband, ds_SRTMband, ds_DEMband1, ds_DEMband2):
def margemask(obj):

	#thresh_srtm_slp = 30.000
	thresh_srtm_slp = THRESH_SRTM_SLP

	#--- get band (GDAL) -------------------------------------
	ds_maskband = obj.mask_ds.GetRasterBand(1)           #FNF data
	ds_srtmmaskband = obj.srtmmask_ds.GetRasterBand(1)   #SRTM slope
	ds_demmask1band = obj.demmask1_ds.GetRasterBand(1)   #PALSAR1
	ds_demmask2band = obj.demmask2_ds.GetRasterBand(1)   #PALSAR2

	#--- read data (GDAL) -------------------------------------
	np_mask = np.array(ds_maskband.ReadAsArray())
	np_srtmmask = np.array(ds_srtmmaskband.ReadAsArray())
	np_demmask1 = np.array(ds_demmask1band.ReadAsArray())
	np_demmask2 = np.array(ds_demmask2band.ReadAsArray())

	#--- marge mask arrays ------------------------------------
	outfocus = np.where( (np_srtmmask > thresh_srtm_slp) | (np_demmask1 > 0) | (np_demmask2 > 0) | (np_mask != 1) )
	np_mask[ outfocus ] = 0

	#--- check num of zero ----
	numpix = np_mask.size
	numpix0 = np_mask[ outfocus ].size

	if numpix == numpix0:
		raise JJFastError("Input images were entirely masked. Thus, the images might be invalid.")

	#print("numpix: {}".format(numpix))
	#print("numpix0: {}".format(numpix0))

	print("--, finished, margemask()")

	return np_mask


# *****************************************************************
#  PALSAR Gamma nought conversion
#  Usage: JJFast.gamma_nought()
# *****************************************************************
def gamma_noughts(gdal_ds):
	#--- read data (GDAL) --------
	array = gdal_ds.GetRasterBand(1).ReadAsArray()

	np_array_64 = array.astype(np.float64)

	#--- convert gamma nought ----
	np_array_64[ np_array_64 <= 0.000 ] = 0.10000
	np_array_64 = 10.00 * np.log10(np_array_64 * np_array_64) - 83.00

	return np_array_64

# *****************************************************************
#  Multiply PALSAR-DN
#  Usage: JJFast.dn_multiply()
# *****************************************************************
def dn_multiply(gdal_ds):
	#--- read data (GDAL) --------
	array = gdal_ds.GetRasterBand(1).ReadAsArray()

	np_array_64 = array.astype(np.float64)

	#--- multipy DN ----
	np_array_64 = np_array_64 * np_array_64

	return np_array_64


# *****************************************************************
#  Gaussian function
# *****************************************************************
def gaussian(x, mu, sig):
	fx = (1/(math.sqrt(2.0*math.pi))) * (np.exp(-np.power(x-mu, 2.0) / (2 * np.power(sig, 2.0))))
	return fx


# *****************************************************************
#  Segmentation using opencv.meanshift
#  Usage: JJFast.segmentation()
# *****************************************************************
def DNcal(obj, np_mask):

	#--- read data (GDAL) -------------------------------------
	np_hv1 = np.array(obj.src_ds1.GetRasterBand(1).ReadAsArray())
	np_hv2 = np.array(obj.src_ds2.GetRasterBand(1).ReadAsArray())

	#--- masking ----------------------------------------------
	np_hv1_masked = np_hv1 * np_mask
	np_hv2_masked = np_hv2 * np_mask

	#--- dataset ----------------------------------------------
	dataset1 = np_hv1_masked[ np_hv1_masked > 0 ]
	dataset2 = np_hv2_masked[ np_hv2_masked > 0 ]

	#--- get stats --------------------------------------------
	mean1 = dataset1.mean()
	std1 = dataset1.std()
	mean2 = dataset2.mean()
	std2 = dataset2.std()

	min_val1 = mean1 - 2.0 * std1
	max_val1 = mean1 + 2.0 * std1
	min_val2 = mean2 - 2.0 * std2
	max_val2 = mean2 + 2.0 * std2

	diff_mean = math.fabs(mean1 - mean2)

	#--- calibration ----------------------------------------------
	if(diff_mean >= 250):
		#--- calibrate DN -------------------------------------
		a = (max_val1 - min_val1)/(max_val2 - min_val2)
		np_hv2_calibrated = (a * (np_hv2 - min_val2)) + min_val1
	else:
		np_hv2_calibrated = np_hv2


	print("--, finished, DNcal()")

	return [np_hv1, np_hv2_calibrated]


# *****************************************************************
#  Segmentation using opencv.meanshift
#  Usage: JJFast.segmentation()
# *****************************************************************
def segmentation(obj, array1, array2):
	""" Execute segmentation using opencv.meanshift. """

	#--- convert data type -------------
	array1_64 = array1.astype(np.float64)
	array2_64 = array2.astype(np.float64)

	# ----------------------------------
	min_val = 1000.0
	max_val = 6000.0

	array1_64[ array1_64 < min_val ] = min_val
	array1_64[ array1_64 > max_val ] = max_val
	array2_64[ array2_64 < min_val ] = min_val
	array2_64[ array2_64 > max_val ] = max_val

	array1_64 -= min_val
	array1_64 //= ( max_val - min_val +1 )/256
	array2_64 -= min_val
	array2_64 //= ( max_val - min_val +1 )/256


	#--- stack layer (numpy) --------------------------------------
	np_stack_64 = np.dstack((np.dstack((array2_64, array1_64)), array1_64))

	#--- convert to byte array (numpy) -------------------------------
	np_stack = np_stack_64.astype(np.uint8)


	#--- Meanshift for nose filtering --------------------------------
	cv2.pyrMeanShiftFiltering(np_stack, 15.0, 1.025, np_stack, 6)

	#--- Meanshift for color degradation -----------------------------
	cv2.pyrMeanShiftFiltering(np_stack, 15.0, 10.0, np_stack, 6)
	#cv2.pyrMeanShiftFiltering(np_stack, 15.0, 5.0, np_stack, 6)


	print("--, finished, segmentation()")

	return np_stack


# *****************************************************************
#  make class image
#  Usage: JJFast.class( cv_src )
# *****************************************************************
def make_class( cv_src ):

	max_class = 23

	#-- common std -----
	std_all = 7.071

	#-- init class array --
	mean_rg_arr = np.zeros(max_class, dtype=np.float32)
	mean_b_arr = np.zeros(max_class, dtype=np.float32)

	#-- define class parameters --
	mean_rg_arr[22] = 150.00
	mean_b_arr[22] =  100.00
	mean_rg_arr[21] = 150.00
	mean_b_arr[21] =  90.00
	mean_rg_arr[20] = 150.00
	mean_b_arr[20] =  70.00
	mean_rg_arr[19] = 150.00
	mean_b_arr[19] =  50.00
	mean_rg_arr[18] = 150.00
	mean_b_arr[18] =  30.00
	mean_rg_arr[17] = 150.00
	mean_b_arr[17] =  10.00
	mean_rg_arr[16] = 130.00
	mean_b_arr[16] =  90.00
	mean_rg_arr[15] = 130.00
	mean_b_arr[15] =  70.00
	mean_rg_arr[14] = 130.00
	mean_b_arr[14] =  50.00
	mean_rg_arr[13] = 130.00
	mean_b_arr[13] =  30.00
	mean_rg_arr[12] = 130.00
	mean_b_arr[12] =  10.00
	mean_rg_arr[11] = 110.00
	mean_b_arr[11] =  70.00
	mean_rg_arr[10] = 110.00
	mean_b_arr[10] = 50.00
	mean_rg_arr[9] = 110.00
	mean_b_arr[9] =  30.00
	mean_rg_arr[8] = 110.00
	mean_b_arr[8] =  10.00

	mean_rg_arr[7] = 90.00
	mean_b_arr[7] =  60.00

	mean_rg_arr[6] = 90.00
	mean_b_arr[6] =  50.00
	mean_rg_arr[5] = 90.00
	mean_b_arr[5] =  30.00
	mean_rg_arr[4] = 90.00
	mean_b_arr[4] =  10.00
	mean_rg_arr[3] = 70.00
	mean_b_arr[3] =  30.00
	mean_rg_arr[2] = 70.00
	mean_b_arr[2] =  10.00
	mean_rg_arr[1] = 50.00
	mean_b_arr[1] =  10.00

	#--- split r,g,b array ---
	cv_data_b, cv_data_g, cv_data_r = cv2.split(cv_src)

	np_arr_rg_64 = np.array(cv_data_r).astype(np.float64)
	np_arr_b_64 = np.array(cv_data_b).astype(np.float64)
	nrows = np_arr_rg_64.shape[0]
	ncols = np_arr_rg_64.shape[1]

	#--- probability array ---
	total_prob_array = np.zeros((nrows, ncols, max_class), dtype=np.float32)

	for i in range(1,max_class):
		prob_array_rg = gaussian(np_arr_rg_64, mean_rg_arr[i], std_all)
		prob_array_b  = gaussian(np_arr_b_64, mean_b_arr[i], std_all)
		total_prob = pow((prob_array_rg * prob_array_b), (1.0/2.0) )

		#--- threshold ---
		total_prob[total_prob < TOTAL_PROB_THRESH] = -0.1
		total_prob_array[:,:,i] = total_prob

	#--- class array ----------------------------
	class_array = np.zeros((nrows, ncols), dtype=np.byte)
	class_array = np.argmax(total_prob_array, axis=2)

	#--- mask ----------------------------
	class_array[ (class_array > 0) & (class_array < max_class) ] = 1.0

	print("--, finished, class()")

	return class_array


# *****************************************************************
#  Polygonization
#  Usage: JJFast.polygonize()
# *****************************************************************
def polygonize(obj, np_img, np_mask, outshpFile=None):
	""" Execute polygonization using gdal.polygonize """

	if outshpFile is None:
		outshpFile = obj.tmpshpFile1
	#--- open tmp file (GDAL) -----------------------------------------
	ogr_shp = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource( outshpFile )

	#--- get band (GDAL) -------------------------------------
	datatype = 1   # uint8
	drv = gdal.GetDriverByName('MEM')
	ds_img = drv.Create( '', obj.cols, obj.rows, 1, datatype )
	ds_img.SetProjection( obj.proj )
	ds_img.SetGeoTransform( obj.geo )
	ds_imgband = ds_img.GetRasterBand(1)
	#ds_imgband.WriteArray(np_img)

	#---  mask band (GDAL) -----------------------------------
	datatype = 1   # uint8
	drv = gdal.GetDriverByName('MEM')
	ds_mask = drv.Create( '', obj.cols, obj.rows, 1, datatype )
	ds_mask.SetProjection( obj.proj )
	ds_mask.SetGeoTransform( obj.geo )
	ds_maskband = ds_img.GetRasterBand(1)
	ds_maskband.WriteArray(np_mask)

	#--- masking -------------------------------------------
	np_img = np_img * np_mask
	ds_imgband.WriteArray(np_img)

	#--- create layer  (ogr) -------------------------------
	ogr_layer = ogr_shp.CreateLayer("polygonized")

	#--- exec raster to polygon (GDAL) ----------------------------------
	gdal.Polygonize( ds_imgband, ds_maskband, ogr_layer, 0, [], callback=None )


	#--- number of features -----
	featureCount = ogr_layer.GetFeatureCount()

	ogr_shp = None


	print("--, finished, polygonize()")

	return featureCount


# *****************************************************************
#  Polygon stats
#  Usage: JJFast.polystats()
# *****************************************************************
def polystats(obj, ds1_img, ds2_img, outshpFile1=None, outshpFile2=None):
	""" Execute polygon stats """

	print("--, start:, polystats()")

	#--- set geotrans ----------------------------------------------
	gt = obj.geo
	at = Affine.from_gdal(gt[0],gt[1],gt[2],gt[3],gt[4],gt[5])

	#--- polygon stats (rasterstats) ------------------------------
	if outshpFile1 is None:
		outshpFile1 = obj.tmpshpFile1


	ds2_img[np.where(ds2_img==0)] = 0.0000000001
	ratio_img = ds1_img/ds2_img
	stats_stt = zonal_stats( outshpFile1, ds1_img, affine=at, nodata=-999)
	stats_end = zonal_stats( outshpFile1, ds2_img, affine=at, nodata=-999)
	stats_ratio = zonal_stats( outshpFile1, ratio_img, affine=at, nodata=-999)

	print("--, file created:, " + outshpFile1)

	stats_means_stt = [stat['mean'] for stat in stats_stt]
	stats_means_end = [stat['mean'] for stat in stats_end]
	stats_means_ratio = [stat['mean'] for stat in stats_ratio]
	stats_count = [stat['count'] for stat in stats_end]


	#--- write stats(array value) to shape file (fiona) ---
	inshpFile = outshpFile1
	if outshpFile2 is None:
		outshpFile2 = obj.tmpshpFile2

	with fiona.open( inshpFile ) as fiona_input:
		src_driver = fiona_input.driver
		src_crs = fiona_input.crs   #coordinate system (projection)
		src_schema = fiona_input.schema
		src_schema['properties']['mean_stt'] = 'float:10.5'
		src_schema['properties']['mean_end'] = 'float:10.5'
		src_schema['properties']['mean_diff'] = 'float:10.5'
		src_schema['properties']['ChangeArea'] = 'float:10.5'
		src_schema['properties']['Accuracy'] = 'int:5'

		with fiona.open( outshpFile2, 'w', driver=src_driver,
                                 crs=src_crs, schema=src_schema) as fiona_output:
			for i,feature in enumerate(fiona_input):
				if stats_count[i] >= 4 and stats_count[i] < 2000 :  # 1-500 ha

					#--- escape zero ------
					if stats_means_ratio[i] < 1:
						stats_means_ratio[i] = 0.000000001
					if stats_means_stt[i] < 1:
						stats_means_stt[i] = 0.000000001
					if stats_means_end[i] < 1:
						stats_means_end[i] = 0.000000001


					#--- check accuracy ---
					accuracy = 0
					diff_dB = 10.00 * np.log10(stats_means_ratio[i])

					if diff_dB >= 5.0:
						accuracy = 1
					if diff_dB >= 3.0 and diff_dB < 5.0:
						accuracy = 2
					if diff_dB >= 2.0 and diff_dB < 3.0:
						accuracy = 3

					# -- set attributes ----
					feature['properties']['mean_stt'] = 10.00 * np.log10(stats_means_stt[i]) - 83.00
					feature['properties']['mean_end'] = 10.00 * np.log10(stats_means_end[i]) - 83.00
					feature['properties']['mean_diff'] = diff_dB
					feature['properties']['ChangeArea'] = float(stats_count[i]) * 0.25
					feature['properties']['Accuracy'] = accuracy
					fiona_output.write(feature)

			print("--, file created:, " + outshpFile2)
	print("--, finished:, polystats()")




# *****************************************************************
#  Polygon extraction
#  Usage: JJFast.extractpolygon()
# *****************************************************************
def extractpolygon(obj, min_ha, shpFile=None, outFile=None, ogr_com=None):
	""" Execute polygon extraction """

	if shpFile is None:
		shpFile = obj.tmpshpFile2
	if outFile is None:
		outFile = obj.outfile

	if ogr_com is None:
		ogr_com = obj.ogr2ogr

	com = [ogr_com, '-f', 'ESRI Shapefile', outFile, shpFile,
                 '-where', 'mean_diff >= 3.0 and mean_diff <= 20.0 and ChangeArea > '+ min_ha +' and mean_stt < -6.0 and mean_stt > -20.0 and mean_end < -6.0 and mean_end > -29.0']

	#--- extract polygon (external command, ogr2ogr) ----
	check_output(com, stderr=subprocess.STDOUT)

	print("--, finished:, extractpolygon()")


# *****************************************************************
# Generate prj file
# *****************************************************************
def mkprjfile(filename):

	prj = open(filename, "w")

	epsg = 'GEOGCS["GCS_WGS_1984",'
	epsg += 'DATUM["D_WGS_1984",'
	epsg += 'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
	epsg += 'PRIMEM["Greenwich",0.0],'
	epsg += 'UNIT["Degree",0.0174532925199433]]'

	prj.write(epsg)
	prj.close()


# *****************************************************************
# Generate json file
# *****************************************************************
def mkjson(tilefile_end, tilefile_begin, shpfile):

	mkjson_com = '/dat1/JJ_PRDCT/tool/JJ_divide_polygon/mk_json.rb'

	com = [mkjson_com, tilefile_end, tilefile_begin, shpfile, '-v', '1.0', '-m', 'AUTO']

	#--- run comamnd ----
	check_output(com, stderr=subprocess.STDOUT)

	print("--, finished:, mkjson()")


# *****************************************************************
#  JJFast class definition
# *****************************************************************
class JJFast:
	"""
	JJFAST Class

	methods:
		JJFast.mkproduct()

	"""

	def __init__(self, infile1, infile2, FNFmaskfile, SRTMmaskfile,
                 DEMmask1, DEMmask2, outfile, tmpDir=None,
                 proj=None, geo=None, cols=None, rows=None):
		"""Initialize args and file names """
		self.infile1 = infile1
		self.infile2 = infile2
		self.FNFmaskfile = FNFmaskfile
		self.SRTMmaskfile = SRTMmaskfile
		self.DEMmask1 = DEMmask1
		self.DEMmask2 = DEMmask2
		self.outfile = outfile


		if tmpDir is None:
			tmpDir = os.path.abspath(os.path.dirname(__file__))
		self.tmpshpFile1 = tmpDir + "/tmp1.shp"
		self.tmpshpFile2 = tmpDir + "/tmp2.shp"

		self.proj = proj
		self.geo = geo
		self.cols = cols
		self.rows = rows
		self.bands = 1
		self.ogr2ogr = 'ogr2ogr'


	def __enter__(self):
		#--- open file (GDAL) --------------------------------
		self.src_ds1 = gdal.Open(self.infile1, gdal.GA_ReadOnly)
		self.src_ds2 = gdal.Open(self.infile2, gdal.GA_ReadOnly)
		self.mask_ds = gdal.Open(self.FNFmaskfile, gdal.GA_ReadOnly)
		self.srtmmask_ds = gdal.Open(self.SRTMmaskfile, gdal.GA_ReadOnly)
		self.demmask1_ds = gdal.Open(self.DEMmask1, gdal.GA_ReadOnly)
		self.demmask2_ds = gdal.Open(self.DEMmask2, gdal.GA_ReadOnly)

		if self.src_ds1 is None:
			print('InputError: cannot input file ' + self.infile1)
			print('')
			sys.exit( 1 )

		if self.src_ds2 is None:
			print('InputError: cannot input file ' + self.infile2)
			print('')
			sys.exit( 1 )

		if self.mask_ds is None:
			print('InputError: cannot input file ' + self.FNFmaskfile)
			print('')
			sys.exit( 1 )

		if self.srtmmask_ds is None:
			print('InputError: cannot input file ' + self.SRTMmaskfile)
			print('')
			sys.exit( 1 )

		if self.demmask1_ds is None:
			print('InputError: cannot input file ' + self.DEMmask1)
			print('')
			sys.exit( 1 )

		if self.demmask2_ds is None:
			print('InputError: cannot input file ' + self.DEMmask2)
			print('')
			sys.exit( 1 )


		#--- metadata (GDAL) ---------------------------------
		self.proj = self.src_ds1.GetProjection()
		self.geo = self.src_ds1.GetGeoTransform()
		self.cols = self.src_ds1.RasterXSize
		self.rows = self.src_ds1.RasterYSize
		self.bands = self.src_ds1.RasterCount

		return self


	def __exit__(self, exception_type, exception_value, traceback):
		#--- close file --------------------------------------
		self.src_ds1 = None
		self.src_ds2 = None
		self.mask_ds = None
		self.srtmmask_ds = None
		self.demmask1_ds = None
		self.demmask2_ds = None



	# *****************************************************************
	#  PALSAR Forest change product
	#  Usage: JJFast.mkproduct()
	# *****************************************************************
	def mkproduct(self):
		""" mkproduct: generate forest change products """

		#--- marge mask ---------------------------
		mask_img = margemask(self)

		#--- DN calibration ------------------------
		cal_data = DNcal(self, mask_img)

		#--- create segmentation image -------------
		seg_img = segmentation(self, cal_data[0], cal_data[1])

		#--- create class image --------------------
		class_img = make_class(seg_img)

		#--- polygonization ------------------------
		numPoly = polygonize(self, class_img, mask_img) 

		if numPoly < 1:
			emptyshapefile(self)
			return 0


		#--- DN multiplication ------------------------
		mlt1_img = dn_multiply(self.src_ds1)
		mlt2_img = dn_multiply(self.src_ds2)

		#--- polygon stats ------------------------
		polystats(self, mlt1_img, mlt2_img)

		#--- polygon extract ------------------------
		extractpolygon(self, minimum_area_ha)
