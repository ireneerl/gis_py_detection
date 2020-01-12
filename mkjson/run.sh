#!/bin/bash

TileFile_1="../test_peru/S08W077_20161013_sl_HV.tif"
TileFile_2="../test_peru/S08W077_20160901_sl_HV.tif"
ShapeFile="../S08W077_161013_160901_01.shp"

 /dat1/JJ_PRDCT/tool/JJ_divide_polygon/mk_json.rb $TileFile_1 $TileFile_2 $ShapeFile -v 1.0 -m AUTO


