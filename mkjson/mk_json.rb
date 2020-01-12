#! /dat1/JJ_PRDCT/program/ruby-2.3.3/bin/ruby
#=Make json file from tile's json and dbf
# Author::    asugano
# Date::      2017/02/15
require 'optparse'
require 'fileutils'
require 'dbf'

$VERSION = "1.0" #this tool version

PRODUCT_NAME = "JFP" #JJ-Fast-Polygone

CONTENTS = "Deforestation"

EXCEPTION_SHAPEFILE_NAME = "WRONG_SHAPEFILE_NAME"

EXCEPTION_DBF_ZERO = "WARNING_DBF_RECORD_ZERO_DATA"

EXCEPTION_EXIST_JSON = "EXIST_JSON"

ARG_ERROR_CORE_VERSION = "ARG_ERROR_CORE_VERSION"

ARG_ERROR_METHOD = "ARG_ERROR_METHOD_TYPE"

OPT_BANNER = <<EOS
=====
Usage
=====
Command Line: $> #{File.basename(__FILE__)} <TileFile_1> <TileFile_2> <ShapeFile> [option]

TileFile(1, 2, ...): tile files
ShapeFile: shape file (input last term)

ex: #{File.basename(__FILE__)} /dats3/palsar/JJFAST/data/Tile/C065/N00W050/N00W048_20170109_sl_HV.tif /dats3/palsar/JJFAST/data/Tile/C062/N00W050/N00W048_20161128_sl_HV.tif /dats3/palsar/JJFAST/data/DFT/C065/N00W050/N00W048_170109_161128_01.shp -v 1.2 -m AUTO

[option]
EOS


# Tile Class
class Tile

  attr :tile_path, :tile_json_path, :tile_json, :tile_name

  def initialize(tile_path)

    @tile_path = tile_path

    @tile_json_path = @tile_path.sub("_sl_HV", "").sub(".tif", ".json")

    @tile_json = {}

    open(@tile_json_path) {|f| @tile_json = JSON.load(f)}

    @tile_name = @tile_json["file_name"].split("_")[0]
  end

  def json_pretty_generate

    JSON.pretty_generate @tile_json
  end
end


# Shape Class
class Shape

  def initialize(tiles, shpe_path)

    @tiles = tiles

    @shp_path = shpe_path

    @dbf_path = @shp_path.sub(".shp", ".dbf")

    @shp_json_path = @shp_path.sub(".shp", ".json")

    mk_json
  end

  def mk_json

    @shp_json = {"file_name" => File.basename(@shp_path, '.shp'),
                   "product" => PRODUCT_NAME,
                   "source_data" => Hash.new
                  }

    s = 0
    @tiles.each do |tl|

      @shp_json["source_data"][sprintf("S%02d", s)] = tl.tile_json

      s += 1
    end

    polygon_info = {"polygon_info" =>
                     {"method" => OPTS[:method],
                      "version" => OPTS[:core_version],
                     }
                   }

    polygons = dbf2hash(@dbf_path)

    polygon_info["polygon_info"].merge!(polygons)

    @shp_json.merge!(polygon_info)

    if polygons.size == 0

      raise EXCEPTION_DBF_ZERO
    end
  end

  def write_json

    unless File.exist?(@shp_json_path)

      open(@shp_json_path, "w") {|g| g.puts JSON.pretty_generate(@shp_json)}
    else

      raise EXCEPTION_EXIST_JSON
    end
  end

  def dbf2hash(dbf)

    tmplt_array = ["", "", nil, "", ""]

    dbf = DBF::Table.new(dbf)

    i = 1
    dbf_entry = {}

    dbf.each do |dd|

      if dd.to_a == tmplt_array

        next
      else

        buf = dd.attributes

        buf["CONTENTS"] = CONTENTS

        dbf_entry[sprintf("P%04d", i)] = dd.attributes

        i += 1
      end
    end

    return dbf_entry
  end
end


begin

  puts "--------------------------"
  puts "Start make JSON of polygon"
  puts "--------------------------"
  puts

  if ARGV.size == 0

    system("#{__FILE__} --help")

    exit 1
  end

  puts "/// parse arguments ///"
  puts

  OPTS = Hash.new(0)

  opt = OptionParser.new

  opt.banner = OPT_BANNER

  opt.on('-v VAL', '--version=VAL', 'The core program\'s version'){|v| OPTS[:core_version] = v}

  opt.on('-m VAL', '--method=VAL', 'MANUAL/AUTO'){|v| OPTS[:method] = v}

  opt.parse!(ARGV)

  unless OPTS[:core_version] =~ /\d\.\d/

    raise ARG_ERROR_CORE_VERSION
  end

  unless OPTS[:method] == "MANUAL" || OPTS[:method] == "AUTO"

    raise ARG_ERROR_METHOD
  end

  tiles = Array.new

  shp_path = ""

  ARGV.each do |fl|

    unless fl =~ /shp$/

      tiles << Tile.new(File.expand_path(fl))
    else

      shp_path = File.expand_path(fl)
    end
  end

  unless File.basename(shp_path) =~ /[NS]\d{2}[WE]\d{3}_\d{6}_\d{6}_\d{2}.shp/

    raise EXCEPTION_SHAPEFILE_NAME
  end

  puts "/// Input ///"
  tiles.each {|t| puts t.tile_path}
  puts shp_path
  puts "Option: #{OPTS}"

  shp = Shape.new(tiles, shp_path)
  puts
  puts "/// write json ///"
  shp.write_json
rescue

  if $!.to_s == EXCEPTION_DBF_ZERO

    puts $!
    puts "---------------------"
    puts $@

    puts "------"
    puts "Finish"
    puts "------"

    exit 2
  else

    puts "-----"
    puts "ERROR"
    puts "-----"
    puts $!
    puts "---------------------"
    puts $@

    exit 1
  end
end

puts "------"
puts "Finish"
puts "------"
exit 0
