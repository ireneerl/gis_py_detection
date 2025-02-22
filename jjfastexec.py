#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
# $Id$ Jul 11, 2017
#
# Name:     jjfastexec.py
# Project:  JJFast
# Purpose:  JJFast main
#
###############################################################################
# Copyright (c) 2017, Izumi Nagatani <nagatani.rs@gmail.com>

projversion="1.0.0"
projname = 'JJ-FAST python'

import sys
import os
import os.path
import shutil
import JJFast.JJFast as jjf
from datetime import datetime
import JJFast.JJFattr as jjfattr

#--- flag to generate json file ---
OUTPUTJSON=False
#OUTPUTJSON=True


#-----------------------------------------------
# Exception
#-----------------------------------------------
class JJFmainError(Exception):
	'''Exception'''


#-----------------------------------------------
# Check file suffixs
#-----------------------------------------------
def chk_suffix(filename, suffix):

	#-- get file extension
	dirpath, ext = os.path.splitext(filename)

	#--
	ext = ext[1:]
	up_suffix = suffix.upper()
	lw_suffix = suffix.lower()

	if ext == up_suffix:
		#print("ok!: " + suffix)
		pass
	elif ext == lw_suffix:
		#print("ok!:" + suffix)
		pass
	else:
		return False


# -- Usage -------------------------------------------------
def Usage():
	print('')
	print('Usage: jjfastexec.py infile1 infile2 FNFmaskfile SRTMmaskfile DEMmaskfile1 DEMmaskfile2 gadm28_0_file gadm28_2_file out_dir', file=sys.stderr)
	print('')
	sys.exit(-1)
# ----------------------------------------------------------


#--- main --------------------------------
def main(*args, **kwargs):


	#--- check args ---
	if len(sys.argv) != 10 and len(sys.argv) != 2:
		raise JJFmainError('Error, invalid args.')

	#--- print usage ---
	if sys.argv[1] == "-h":
		print("Usage:")
		print('jjfastexec.py infile1 infile2 FNFmaskfile SRTMmaskfile DEMmaskfile1 DEMmaskfile2 gadm28_0_file gadm28_2_file out_dir')
		print("")
		print("Option:")
		print("  -h   print help (this)")
		print("  -v   print version")
		print("")
		sys.exit(0)

	#--- print version ---
	if sys.argv[1] == "-v":
		print("jjfastexec.py version " + projversion)
		print("")
		sys.exit(0)


	#-- check file exists ---
	for i in range(1,9):
		if os.path.isfile(sys.argv[i]) is False:
			raise JJFmainError('Error, file does not exist. : ' + sys.argv[i])


	#-- set filename ---
	infile1 = sys.argv[1]       # PALSAR2 slant HV begin
	infile2 = sys.argv[2]       # PALSAR2 slant HV end
	FNFmaskfile = sys.argv[3]   # FNF Mask
	SRTMmaskfile = sys.argv[4]  # SRTM DEM Mask
	DEMmaskfile1 = sys.argv[5]  # Mask for begin data
	DEMmaskfile2 = sys.argv[6]  # Mask for end data
	gadm28_0_file = sys.argv[7] # gadm28_0 shapefile
	gadm28_2_file = sys.argv[8] # gadm28_2 shapefile
	out_dir = sys.argv[9]       # output directory


	basename_infile1 = os.path.basename(infile1)
	basename_infile2 = os.path.basename(infile2)
	Onedeg_img_class = basename_infile1[0:7]
	date_begin = basename_infile1[10:16]
	date_end   = basename_infile2[10:16]

	imghead = Onedeg_img_class + '_' + date_end + '_' + date_begin
	outfile = out_dir + '/' + imghead + '_01.shp'

	#print(imghead)
	#print(outfile)

	#--- check file extensions ---
	if chk_suffix(infile1, 'tif') is False:
		raise JJFmainError('Error, invalid file suffix : ' + infile1)

	if chk_suffix(infile2, 'tif') is False:
		raise JJFmainError('Error, invalid file suffix : ' + infile2)

	if chk_suffix(FNFmaskfile, 'tif') is False:
		raise JJFmainError('Error, invalid file suffix : ' + FNFmaskfile)

	if chk_suffix(SRTMmaskfile, 'tif') is False:
		raise JJFmainError('Error, invalid file suffix : ' + SRTMmaskfile)

	if chk_suffix(DEMmaskfile1, 'tif') is False:
		raise JJFmainError('Error, invalid file suffix : ' + DEMmaskfile1)

	if chk_suffix(DEMmaskfile2, 'tif') is False:
		raise JJFmainError('Error, invalid file suffix : ' + DEMmaskfile2)

	if chk_suffix(gadm28_0_file, 'shp') is False:
		raise JJFmainError('Error, invalid file suffix : ' + gadm28_0_file)

	if chk_suffix(gadm28_2_file, 'shp') is False:
		raise JJFmainError('Error, invalid file suffix : ' + gadm28_2_file)

	#print('check: JJFast inputfile extensions are ok.')

	#--- create tmp directory ----
	cdir = os.getcwd()
	pid = str(os.getpid())

	tmpDir = cdir + '/jjf' + pid
	os.mkdir(tmpDir)

	if not os.path.exists(tmpDir):
		raise jjf.JJFastError('Tmp directory does not exist. : ' + tmpDir)


	#--- tmp file ------------------
	outfile0 = tmpDir + "/out0.shp"

	#--- JJFast core processing ------------------------------------------
	with jjf.JJFast(infile1, infile2, FNFmaskfile, SRTMmaskfile,
                    DEMmaskfile1, DEMmaskfile2, outfile0, tmpDir=tmpDir) as jjfobj:

		#--- JJFast core-algorithm ----
		print('run: JJFast core-algorithm ver ' + projversion)
		jjfobj.mkproduct()



	if os.path.isfile(outfile0) is False:
		raise jjf.JJFastError('File does not exit. : ' + outfile0)


	#--- JJFast modify polygon attribute ---
	jjfattr.modify_attr(imghead, outfile0, gadm28_0_file, gadm28_2_file, outfile)


	#--- create .prj file ---
	if os.path.isfile(outfile) is True:
		prjfile = outfile.replace('.shp', '.prj')
		jjf.mkprjfile(prjfile)

	#--- create .json file ---
	if OUTPUTJSON is True:
		if os.path.isfile(outfile) is True:
			fsize = os.path.getsize(outfile)
			if fsize > 100:
				jjf.mkjson(infile2, infile1, outfile)
			else:
				#-- remove tmp directory ---
				shutil.rmtree(tmpDir)
				print("--, removed tmp directory : " + tmpDir)
				#raise JJFmainError('The output file has no polygon : ' + outfile)
				print("¥n")
				print("The output file has no polygon : " + outfile)
				print("exit(99)")
				sys.exit(99)


	#--- remove tmp directory ----
	shutil.rmtree(tmpDir)
	print("--, removed tmp directory : " + tmpDir)

	if os.path.exists(tmpDir):
		raise JJFmainError('Cannot remove tmp directory : ' + tmpDir)



#---------------------------------------------------------------------
#  Exec main of JJFast core-algorithm
#---------------------------------------------------------------------
if __name__ == '__main__':

	#-------------------------------------
	# run
	#-------------------------------------
	print( datetime.now().strftime("%Y.%m.%d-%H:%M:%S") + ', start, '
           + projname + ', ' + __file__ )

	main()

	print( datetime.now().strftime("%Y.%m.%d-%H:%M:%S") + ', end, '
           + projname + ', ' + __file__ )
