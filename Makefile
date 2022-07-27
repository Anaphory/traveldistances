include gmsl-1.1.9/gmsl

lat := 70S 50S 30S 10S 10N 30N 50N 70N
lon := 180W 150W 120W 090W 060W 030W 000E 030E 060E 090E 120E 150E

tiles := $(foreach LAT,$(lat),$(foreach LON,$(lon),$(LAT)$(LON)))

$(info $$tiles is [${tiles}])

.PHONY: DOWNLOAD ECOREGIONS LOCAL_DISTANCES EXTENDED_DISTANCES

tiling.mk: raster_data.py
	python raster_data.py $(tiles) > tiling.mk

include tiling.mk

DOWNLOAD: $(foreach TILE,$(tiles),../elevation/GMTED2010/$(TILE)_20101117_gmted_mea150.tif) ../ecoregions/Ecoregions2017.zip

../elevation/GMTED2010/%_20101117_gmted_mea150.tif:
	wget https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/topo/downloads/GMTED/Global_tiles_GMTED/150darcsec/mea/$(call substr,$*,7,7)$(call substr,$*,4,6)/$*_20101117_gmted_mea150.tif -O $@

../ecoregions/Ecoregions2017.zip:
	wget https://storage.googleapis.com/teow2016/Ecoregions2017.zip -O $@

ECOREGIONS: $(foreach TILE,$(tiles),../ecoregions/ECOREGIONS-$(TILE)_20101117_gmted_mea150.tif) ../ecoregions/Ecoregions2017.zip

../ecoregions/ECOREGIONS-%.tif: ../elevation/GMTED2010/%.tif ../ecoregions/Ecoregions2017.tif
	gdalsrsinfo -o wkt $< > GMTED.wkt
	gdalinfo $< | grep 'Lower Left\|Upper Right' | sed -E 's/[^\(]+\(([^,]+),([^\)]+)\).+/\1 \2/' | paste --serial > $*-extent.txt
	gdal_rasterize -l Ecoregions2017 -ts 7200 4800 -a OBJECTID -a_nodata 999 -a_srs GMTED.wkt -ot UInt16 -of GTiff /vsizip/../ecoregions/Ecoregions2017.zip -te `cat $*-extent.txt` $@

ALL.json COAST.json LAND.jar:
	python earth.py

DISTANCES: $(foreach TILE,$(tiles),distances-$(TILE).tif)

RIVER-WADES: $(foreach TILE,$(tiles),x-rivers-$(TILE).tif)

rivers-%.sqlite rivers-%.tif: ../rivers/GloRiC_v10_shapefile.zip ../elevation/GMTED2010/%_20101117_gmted_mea150.tif by_river.py
	python by_river.py sqlite:///rivers-$*.sqlite $*
	touch rivers-$*.sqlite

rivers.sqlite:  $(foreach TILE,$(tiles),rivers-$(TILE).sqlite)
	for input in $^; \
	  do sqlite3 $$input .dump | \
	  sed -e 's/CREATE TABLE/CREATE TABLE IF NOT EXISTS/' | \
	  sqlite3 /tmp/$@ ; \
	  echo $$input done ; \
	done
	mv /tmp/$@ $@

core-points-%.sqlite: distances-%.tif x-rivers-%.tif core.py
	python core.py sqlite:///core-points-$*.sqlite $*
	touch core-points-$*.sqlite

core-points.sqlite: $(foreach TILE,$(tiles),core-points-$(TILE).sqlite)
	for input in $^; \
	  do sqlite3 $$input .dump | \
	  sed -e 's/CREATE TABLE/CREATE TABLE IF NOT EXISTS/' | \
	  sqlite3 /tmp/$@ ; \
	  echo $$input done ; \
	done
	mv /tmp/$@ $@

all-points.sqlite: core-points.sqlite rivers.sqlite
	for input in $^; \
	  do sqlite3 $$input .dump | \
	  sed -e 's/CREATE TABLE/CREATE TABLE IF NOT EXISTS/' | \
	  sqlite3 /tmp/$@ ; \
	done
	mv /tmp/$@ $@

sea_distance.sqlite: all-points.sqlite
	cp $< /tmp/$@
	python by_sea.py sqlite:////tmp/$@
	mv /tmp/$@ $@

voronoi-%.tif min_distances-%.tif x-pairwise-%.sqlite: distances-%.tif x-rivers-%.tif all-points.sqlite
	cp all-points.sqlite /tmp/pairwise-$*.sqlite
	python main.py sqlite:////tmp/pairwise-$*.sqlite $*
	mv /tmp/pairwise-$*.sqlite x-pairwise-$*.sqlite

# This way of generating the pairwise distances has less extensive requirements,
# so it can be executed for an individual tile before all core points have been
# generated. But it may have issues with core points near the boundary.
pairwise-%.sqlite: distances-%.tif x-rivers-%.tif rivers-%.sqlite core-points-%.sqlite
	cp all-points.sqlite /tmp/pairwise-$*.sqlite
	for input in rivers-$*.sqlite core-points-$*.sqlite ; \
	  do sqlite3 $$iinput .dump | \
	  sed -e 's/CREATE TABLE/CREATE TABLE IF NOT EXISTS/' | \
	  sqlite3 /tmp/$@ ; \
	done
	python main.py sqlite:////tmp/pairwise-$*.sqlite $*
	mv /tmp/pairwise-$*.sqlite pairwise-$*.sqlite

all-distances.sqlite: $(foreach TILE,$(tiles),x-pairwise-$(TILE).sqlite) sea_distance.sqlite
	for input in $^; \
	  do sqlite3 $$input .dump | \
	  sed -e 's/CREATE TABLE/CREATE TABLE IF NOT EXISTS/' | \
	  sqlite3 /tmp/$@ ; \
	  echo $$input ; \
	done
	mv /tmp/$@ $@

database-with-build-boats.sqlite: full-database.sqlite
	cp full-database.sqlite /tmp/database-with-build-boats.sqlite
	echo "UPDATE edges SET travel_time = travel_time + 8*3600 WHERE node2 < 100000000 AND source = 'grid'" | sqlite3 /tmp/database-with-build-boats.sqlite
	echo "UPDATE edges SET travel_time = travel_time + 8*3600 WHERE node1 > 100000000 AND source = 'sea'" | sqlite3 /tmp/database-with-build-boats.sqlite
	mv /tmp/database-with-build-boats.sqlite database-with-build-boats.sqlite
