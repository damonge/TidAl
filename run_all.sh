#!/bin/bash

for cat in 2MASS_XSC 2MPZ_DA 2MPZ_raw
do
    for sm in 0.5 1.0 2.0
    do
	for crd in EQ GAL
	do
	    python3 get_tidal_maps.py --coords ${crd} --catalog data/${cat}.fits --k-threshold 13.9 --plot-stuff --nside 64 --output-prefix outputs/out_${cat}_${crd} --smooth-scale ${sm}
	done
	python3 get_tidal_maps.py --coords GAL --catalog data/${cat}.fits --k-threshold 13.9 --plot-stuff --nside 64 --output-prefix outputs/out_${cat}_ROT --smooth-scale ${sm} --rotate
    done
done
