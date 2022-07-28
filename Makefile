include gmsl-1.1.9/gmsl

lat := 70S 50S 30S 10S 10N 30N 50N 70N
lon := 180W 150W 120W 090W 060W 030W 000E 030E 060E 090E 120E 150E

tiles := $(foreach LAT,$(lat),$(foreach LON,$(lon),$(LAT)$(LON)))

$(info $$tiles is [${tiles}])

.PHONY: DOWNLOAD ECOREGIONS LOCAL_DISTANCES EXTENDED_DISTANCES

tiling.mk: code/raster_data.py
	python code/raster_data.py $(tiles) > tiling.mk

include tiling.mk

DOWNLOAD: $(foreach TILE,$(tiles),elevation/GMTED2010/$(TILE)_20101117_gmted_mea150.tif) ecoregions/Ecoregions2017.zip

elevation/GMTED2010/%_20101117_gmted_mea150.tif:
	wget https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/topo/downloads/GMTED/Global_tiles_GMTED/150darcsec/mea/$(call substr,$*,7,7)$(call substr,$*,4,6)/$*_20101117_gmted_mea150.tif -O $@

ecoregions/Ecoregions2017.zip:
	wget https://storage.googleapis.com/teow2016/Ecoregions2017.zip -O $@

ECOREGIONS: $(foreach TILE,$(tiles),ecoregions/ECOREGIONS-$(TILE)_20101117_gmted_mea150.tif)

ecoregions/ECOREGIONS-%.tif: elevation/GMTED2010/%.tif ecoregions/Ecoregions2017.zip
	gdalsrsinfo -o wkt $< > GMTED.wkt
	gdalinfo $< | grep 'Lower Left\|Upper Right' | sed -E 's/[^\(]+\(([^,]+),([^\)]+)\).+/\1 \2/' | paste --serial > $*-extent.txt
	gdal_rasterize -l Ecoregions2017 -ts 7200 4800 -a OBJECTID -a_nodata 999 -a_srs GMTED.wkt -ot UInt16 -of GTiff /vsizip/ecoregions/Ecoregions2017.zip -te `cat $*-extent.txt` $@

ALL.json COAST.json LAND.jar:
	python code/earth.py

DISTANCES: $(foreach TILE,$(tiles),distances-$(TILE).tif)

RIVER-WADES: $(foreach TILE,$(tiles),x-rivers-$(TILE).tif)

db/rivers-%.sqlite rivers-%.tif: rivers/GloRiC_v10_shapefile.zip elevation/GMTED2010/%_20101117_gmted_mea150.tif code/by_river.py
	python code/by_river.py sqlite:///db/rivers-$*.sqlite $*
	touch db/rivers-$*.sqlite

db/rivers.sqlite:  $(foreach TILE,$(tiles),rivers-$(TILE).sqlite)
	for input in $^; \
	  do sqlite3 $$input .dump | \
	  sed -e 's/CREATE TABLE/CREATE TABLE IF NOT EXISTS/' | \
	  sqlite3 /tmp/$@ ; \
	  echo $$input done ; \
	done
	mv /tmp/$@ $@

db/core-points-%.sqlite: distances-%.tif x-rivers-%.tif code/core.py
	python code/core.py sqlite:///db/core-points-$*.sqlite $*
	touch db/core-points-$*.sqlite

db/core-points.sqlite: $(foreach TILE,$(tiles),db/core-points-$(TILE).sqlite)
	for input in $^; \
	  do sqlite3 $$input .dump | \
	  sed -e 's/CREATE TABLE/CREATE TABLE IF NOT EXISTS/' | \
	  sqlite3 /tmp/$@ ; \
	  echo $$input done ; \
	done
	mv /tmp/$@ $@

db/all-points.sqlite: db/core-points.sqlite db/rivers.sqlite
	for input in $^; \
	  do sqlite3 $$input .dump | \
	  sed -e 's/CREATE TABLE/CREATE TABLE IF NOT EXISTS/' | \
	  sqlite3 /tmp/$@ ; \
	done
	mv /tmp/$@ $@

db/sea_distance.sqlite: db/all-points.sqlite
	cp $< /tmp/$@
	python code/by_sea.py sqlite:////tmp/$@
	mv /tmp/$@ $@

voronoi-%.tif min_distances-%.tif db/x-pairwise-%.sqlite: distances-%.tif x-rivers-%.tif db/all-points.sqlite
	cp db/all-points.sqlite /tmp/pairwise-$*.sqlite
	python code/main.py sqlite:////tmp/pairwise-$*.sqlite $*
	mv /tmp/pairwise-$*.sqlite db/x-pairwise-$*.sqlite

# This way of generating the pairwise distances has less extensive requirements,
# so it can be executed for an individual tile before all core points have been
# generated. But it may have issues with core points near the boundary.
db/pairwise-%.sqlite: distances-%.tif x-rivers-%.tif db/rivers-%.sqlite db/core-points-%.sqlite
	cp all-points.sqlite /tmp/pairwise-$*.sqlite
	for input in db/rivers-$*.sqlite db/core-points-$*.sqlite ; \
	  do sqlite3 $$iinput .dump | \
	  sed -e 's/CREATE TABLE/CREATE TABLE IF NOT EXISTS/' | \
	  sqlite3 /tmp/$@ ; \
	done
	python code/main.py sqlite:////tmp/pairwise-$*.sqlite $*
	mv /tmp/pairwise-$*.sqlite db/pairwise-$*.sqlite

db/all-distances.sqlite: $(foreach TILE,$(tiles),db/x-pairwise-$(TILE).sqlite) db/sea_distance.sqlite
	for input in $^; \
	  do sqlite3 $$input .dump | \
	  sed -e 's/CREATE TABLE/CREATE TABLE IF NOT EXISTS/' | \
	  sqlite3 /tmp/$@ ; \
	  echo $$input ; \
	done
	mv /tmp/$@ $@

db/database-with-build-boats.sqlite: all-distances.sqlite
	cp db/all_distances.sqlite /tmp/database-with-build-boats.sqlite
	echo "UPDATE edges SET travel_time = travel_time + 8*3600 WHERE node2 < 100000000 AND source = 'grid'" | sqlite3 /tmp/database-with-build-boats.sqlite
	echo "UPDATE edges SET travel_time = travel_time + 8*3600 WHERE node1 > 100000000 AND source = 'sea'" | sqlite3 /tmp/database-with-build-boats.sqlite
	mv /tmp/database-with-build-boats.sqlite $@
