#! /bin/bash

python3 inserter.py -f ./Data/dimoi.csv -c dimoi -s ','

python3 inserter.py -f ./Data/staseis_dimoi.csv -c staseis_dimoi  -s ','

python3 inserter.py -f ./Data/staseis_proexoxes.csv -c staseis_proexoxes -s ';'

python3 inserter.py -f ./Data/staseis_taxi.csv -c staseis_taxi -s ';'

python3 inserter.py -f ./Data/staseis_metro.csv -c staseis_metro -s ','

python3 inserter.py -f ./Data/line_categories.csv -c line_categories  -s ';'

python3 inserter.py -f ./Data/landmarks.csv -c landmarks -s ','

python3 inserter.py -f ./Data/stops_by_line.csv -c stops_by_line  -s ','

python3 inserter.py -f ./Data/stops.csv -c stops -s ','

python3 inserter.py -f ./Data/vehicle_capacities.csv -c vehicle_capacities -s ','
