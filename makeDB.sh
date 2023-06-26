#! /bin/bash

python3 inserter.py -f dimoi.csv -c dimoi -s ','

python3 inserter.py -f staseis_dimoi.csv -c staseis_dimoi  -s ','

python3 inserter.py -f staseis_proexoxes.csv -c staseis_proexoxes -s ';'

python3 inserter.py -f staseis_taxi.csv -c staseis_taxi -s ';'

python3 inserter.py -f staseis_metro.csv -c staseis_metro -s ','

python3 inserter.py -f line_categories.csv -c line_categories  -s ';'

python3 inserter.py -f landmarks.csv -c landmarks -s ','

python3 inserter.py -f stops_by_line.csv -c stops_by_line  -s ','

python3 inserter.py -f stops.csv -c stops -s ','

python3 inserter.py -f vehicle_capacities.csv -c vehicle_capacities -s ','
