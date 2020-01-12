#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------
# $ID$ Jul 5, 2017
#
# JJFattr.py
#               Izumi Nagatani (nagatani.izumi@jaxa.jp)
#-------------------------------------------------------
prj_version="1.0.0"

import sys
import os
import math
from datetime import datetime
from osgeo import ogr
from shapely.geometry import Point, Polygon, asShape, mapping
from fiona import collection
from collections import OrderedDict

os.environ['SHAPE_ENCODING'] = "utf-8"

#-----------------------------------------------
# Exception
#-----------------------------------------------
class JJFattrError(Exception):
	'''Exception'''

#-----------------------------------------------
# get attributes of adm0
#-----------------------------------------------
def getattr_adm0(ogr_layer, plat, plon):

	#--- set geo point -----------
	pt = ogr.Geometry(ogr.wkbPoint)
	pt.SetPoint_2D(0, plon, plat)

	#--- filtering ---- 
	ogr_layer.SetSpatialFilter(pt)

	#--- get feature attributes -----
	atr_country = "Unidentified"
	atr_continent = "Unidentified"

	for ogr_feature in ogr_layer:
		atr_country = ogr_feature.GetFieldAsString("NAME_ENGLI")
		atr_continent = ogr_feature.GetFieldAsString("UNREGION1")

	return [atr_country, atr_continent]

#-----------------------------------------------
# get attributes of adm2
#-----------------------------------------------
def getattr_adm2(ogr_layer, plat, plon):

	#--- set geo point -----------
	pt = ogr.Geometry(ogr.wkbPoint)
	pt.SetPoint_2D(0, plon, plat)

	#--- filtering ---- 
	ogr_layer.SetSpatialFilter(pt)

	#--- get feature attributes -----
	atr_state = "Unidentified"
	atr_town = "Unidentified"

	for ogr_feature in ogr_layer:
		atr_state = ogr_feature.GetFieldAsString("NAME_1")
		atr_town = ogr_feature.GetFieldAsString("NAME_2")

	return [atr_state, atr_town]

#-----------------------------------------------
# change unit degree to degree-min-sec
#-----------------------------------------------
def chgunit_lat(degree_value):

	if degree_value > 0:
		head = 'N'
	else:
		head = 'S'

	degree_value = abs(degree_value)

	degree = math.floor(degree_value)
	minute_float = (degree_value - degree) * 60.0
	minute = math.floor(minute_float)
	sec_float = (minute_float - minute) * 60.0
	second = math.floor(sec_float)

	return [ head, degree, minute, second ]


def chgunit_lon(degree_value):

	if degree_value > 0:
		head = 'E'
	else:
		head = 'W'

	degree_value = abs(degree_value)

	degree = math.floor(degree_value)
	minute_float = (degree_value - degree) * 60.0
	minute = math.floor(minute_float)
	sec_float = (minute_float - minute) * 60.0
	second = math.floor(sec_float)

	return [ head, degree, minute, second ]


#-----------------------------------------------
# Modify attribute
#-----------------------------------------------
def modify_attr(imghead, input_shp, adm28_0, adm28_2, output_shp):
	''' modify_attr '''

	#--- file open ---
	inshpfile1 = input_shp    #input file
	inshpfile2 = adm28_0    # adm0
	inshpfile3 = adm28_2    # adm2
	outshpfile = output_shp    # output

	#-- check file exists ---
	if os.path.isfile(inshpfile1) is False:
		raise JJFattrError('Error, file does not exist. : ' + inshpfile1)
	if os.path.isfile(inshpfile2) is False:
		raise JJFattrError('Error, file does not exist. : ' + inshpfile2)
	if os.path.isfile(inshpfile3) is False:
		raise JJFattrError('Error, file does not exist. : ' + inshpfile3)
	if outshpfile is None:
		raise JJFattrError('Error, invalid output file. : ' + outshpfile)

	#--- init list ---
	ppoint = []
	plon = []
	plat = []
	poly_geometry = []
	changearea = []
	accuracy = []
	poly_id = []
	
	country_list = []
	continent_list = []
	state_list = []
	town_list = []


	#--- open file and  get ogr datasorce and layer ---------------
	ogr_ds_adm0 = ogr.GetDriverByName('ESRI Shapefile').Open(inshpfile2)
	ogr_ds_adm2 = ogr.GetDriverByName('ESRI Shapefile').Open(inshpfile3)

	ogr_layer_adm0 = ogr_ds_adm0.GetLayer(0)
	ogr_layer_adm2 = ogr_ds_adm2.GetLayer(0)


	#--- check each points -----------
	numskip=0
	with collection( inshpfile1, 'r', encoding='utf-8') as shp:
		for i,feature in enumerate(shp):


			#--- point within the polygon ---
			try:
				apoint = asShape(feature['geometry']).representative_point()
			except:
				numskip = numskip + 1
				continue

			alon = apoint.coords[0][0]
			alat = apoint.coords[0][1]

			#--- append attr value -----
			plon.append(alon)
			plat.append(alat)
			ppoint.append(apoint)
			poly_geometry.append(feature['geometry'])

			if 'CHANGEAREA' in feature['properties']:
				changearea.append(feature['properties']['CHANGEAREA'])
			else:
				changearea.append(feature['properties']['ChangeArea'])

			if 'Accuracy' in feature['properties']:
				accuracy.append(feature['properties']['Accuracy'])
			else:
				accuracy.append(feature['properties']['ACCURACY'])

			#--- get attributes from adm28_0 using lat,lon ----
			country, continent = getattr_adm0(ogr_layer_adm0, alat, alon)
			#--- get attributes from adm28_2 using lat,lon ----
			state, town = getattr_adm2(ogr_layer_adm2, alat, alon)

			#--- append attr value ----
			country_list.append(country)
			continent_list.append(continent)
			state_list.append(state)
			town_list.append(town)

	
		npoints = len(shp) - numskip


	#--- close ogr data source ---
	ogr_ds_adm0.Destroy()
	ogr_ds_adm2.Destroy()


	#-- define schema for output -----
	schema_props = OrderedDict([('Country','str'),('Continent','str'),
                                    ('ChangeArea','float:6.2'),
                                    ('Accuracy','int'),
                                    ('Polygon_id','str'),
                                    ('State','str'),('Town','str'),
                                    ('Latitude','str'),('Longitude','str')])

	schema1 = { 'geometry':'Polygon', 'properties': schema_props }


	#--- generate output file ---
	with collection( outshpfile, 'w', 'ESRI Shapefile', schema1, encoding='utf-8') as output:
		u_deg = '\u00b0'
		u_min = '\u2032'
		u_sec = '\u2033'

		for i in range(0, npoints):
			poly_id = '%07d' % i
			poly_id = imghead + '_' + poly_id + 'A'

			lathead,latdeg,latmin,latsec = chgunit_lat(plat[i])
			lonhead,londeg,lonmin,lonsec = chgunit_lon(plon[i])

			output.write(
                               {'properties':({
                                'Country': country_list[i].title().replace(' ', '_'),
                                'Continent': continent_list[i].title().replace(' ', '_'),
                                'ChangeArea': changearea[i],
                                'Accuracy': accuracy[i],
                                'Polygon_id': poly_id,
                                'State': state_list[i].title().replace(' ', '_'),
                                'Town': town_list[i].title().replace(' ', '_'),
                                'Latitude': str(lathead) + str(latdeg) + u_deg + str(latmin) + u_min + str(latsec) + u_sec,
                                'Longitude': str(lonhead) + str(londeg) + u_deg + str(lonmin) + u_min + str(lonsec) + u_sec}),
                                'geometry': poly_geometry[i],
                               }
                        )


	print("--, finished:, modify_attr()")

	return 0	



