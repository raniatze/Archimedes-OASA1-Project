#! /bin/bash

python3 inserter.py -f ake.csv -c ake
python3 inserter.py -f events.csv -c events
python3 inserter.py -f dimoi.csv -c dimoi -s ','


wget https://catalog.hcapdata.gr/dataset/956ed989-1e5c-4766-8289-a27373071f0f/resource/56634236-37b3-4c20-bda9-98838b2d6f93/download/stops.txt
python3 inserter.py -f stops.txt -c stops -s ','

wget https://catalog.hcapdata.gr/dataset/956ed989-1e5c-4766-8289-a27373071f0f/resource/8ffcd304-0cb9-4ae5-a918-ae429eae9319/download/routes.txt
python3 inserter.py -f routes.txt -c routes -s ','

wget https://catalog.hcapdata.gr/dataset/956ed989-1e5c-4766-8289-a27373071f0f/resource/b53f97a4-1dbd-44ea-880a-3d56580bedfe/download/calendar_dates.txt 
python3 inserter.py -f calendar_dates.txt -c calendar_dates -s ','

wget https://catalog.hcapdata.gr/dataset/956ed989-1e5c-4766-8289-a27373071f0f/resource/267c3ffc-b477-4cee-ae4b-d827cb602113/download/calendar.txt 
python3 inserter.py -f calendar.txt -c calendar -s ','

wget https://catalog.hcapdata.gr/dataset/956ed989-1e5c-4766-8289-a27373071f0f/resource/edff8832-92d6-4c7f-b066-53f220f0d24a/download/trips.rar
unrar x trips.rar
python3 inserter.py -f trips.txt -c trips -s ','

wget https://catalog.hcapdata.gr/dataset/956ed989-1e5c-4766-8289-a27373071f0f/resource/04f8dc80-a3b1-491a-a2fe-bc70f0b6668b/download/shapes.rar
unrar x shapes.rar
python3 inserter.py -f shapes.txt -c shapes -s ','

wget https://catalog.hcapdata.gr/dataset/956ed989-1e5c-4766-8289-a27373071f0f/resource/26c76d2b-2f43-4110-988f-e3ac3ec09a9d/download/stop_times.rar
unrar x stop_times.rar
python3 inserter.py -f stop_times.txt -c stop_times -s ','

wget https://catalog.hcapdata.gr/dataset/a7691904-f9a0-4d7e-a6e1-88e24bf0df8a/resource/0763e95f-c9e7-488d-be03-daf31c59b8a5/download/stops_230306.rar
unrar x stops_230306.rar
python3 kmltocsv.py
python3 inserter.py -f staseis_dimoi.csv -c staseis_dimoi  -s ','

wget https://catalog.hcapdata.gr/dataset/7deec407-6799-4975-beb2-af983e454e02/resource/2791d19c-a08b-4643-8a45-90a3ad1ec445/download/report1c_vasikodiktyo_dimon_230110.csv
python3 inserter.py -f report1c_vasikodiktyo_dimon_230110.csv -c grammes_dimoi -s ','
