# Input:  .csv file from ONE day (ake_data.csv)
# Output: ake_updated.csv file for that day
import os
import csv
import pymongo
import holidays
import pandas as pd
from datetime import datetime, timedelta

# Set up the connection to the MongoDB server
client = pymongo.MongoClient('mongodb://localhost:27017/')

# Choose the database and collection to work with
db = client['OASA1']

def find_csv_files(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def make_dataset(csv_file):

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file, sep=';')

    # Group the data by Line_descr, Rtype, Direction, Sched, and Vehicle_no
    grouped = df.groupby(['Line_descr', 'Rtype', 'Direction', 'Sched', 'Vehicle_no'])
    
    # Split the file path into directory and filename components
    dir_path, _ = os.path.split(csv_file)
    
    updated_csv_file = os.path.join(dir_path, 'ake_updated.csv')

    with open(updated_csv_file, 'w') as output_file:
         
         csv_writer = csv.writer(output_file, delimiter=';')
         
         # Write header row in output_file
         csv_writer.writerow(['Route_id', 'Rtype', 'Stop_id', 'Stop_order', 'Minute_of_day', 'Day_of_month', 'Day_of_week', 'Week_of_year', 'Day_of_year', 'Is_holiday', 'Temperature', 'Precipitation', 'T_pa_in_veh'])
         
         # Iterate over each group
         for name, group in grouped:
             print(name)
             
             prev_stop_time = None
             prev_stop_order = None
             
             # Sort the entries of the group by S_order column in ascending order
             group_sorted = group.sort_values('S_order', ascending=True)
             group_sorted = group_sorted.reset_index()
             
             # Iterate over each row of the sorted group
             for i, row in group_sorted.iterrows():
             
                 line_descr, rtype, stop_order, arrival_datetime, stop_id, t_pa_in_veh = row['Line_descr'], row['Rtype'], row['S_order'], row['Arrival_datetime'], row['Stop_id'], row['T_pa_in_veh']
                 
                 # Find route_id
                 split_string = line_descr.split(" - ")
                 route_short_name, route_long_name = split_string[0], " - ".join(split_string[1:])
                 result = db.routes.find({"route_short_name": route_short_name, "route_long_name": route_long_name}, {"route_id": 1})
                 for doc in result:
                    route_id = doc["route_id"]
                    
                 # Find municipality of the stop_id
                 result = db.staseis_dimoi.find({"stop_id": str(stop_id)}, {"dimos": 1})
                 for doc in result:
                    dimos = doc["dimos"]
                    
                 # Stop has no arrival_datetime
                 if pd.isna(arrival_datetime):
                 
                    # Previous stop has arrival datetime
                    if prev_stop_time is not None:
                       stop_order = int(row['S_order'])
                       
                       # There is next stop
                       if i != len(group_sorted) - 1:
                          next_stop = group_sorted.iloc[i+1]
                          
                          # Next stop has arrival datetime
                          if not pd.isna(next_stop['Arrival_datetime']):
                             #print("1")
                             next_stop_time = datetime.strptime(next_stop['Arrival_datetime'], '%Y-%m-%d %H:%M:%S')
                             next_stop_order = int(next_stop['S_order'])
                             time_diff = (next_stop_time - prev_stop_time) / (next_stop_order - prev_stop_order)
                             stop_time = prev_stop_time + (time_diff * (stop_order - prev_stop_order))
                             arrival_datetime = stop_time
                             
                          # Next stop has no arrival datetime
                          else: 
                             #print("2")
                             arrival_datetime = prev_stop_time
                             prev_stop_order = stop_order
                        
                       # There is no next stop
                       else:
                          #print("3")
                          arrival_datetime = prev_stop_time
                          prev_stop_order = stop_order
                          
                    # Previous stop has no arrival datetime
                    else:
                       #print("4")
                       arrival_datetime = ''
                       
                 # Stop has arrival datetime
                 else:
                    #print("5")
                    prev_stop_time = datetime.strptime(arrival_datetime, '%Y-%m-%d %H:%M:%S')
                    prev_stop_order = int(row['S_order'])
                    
                 # Convert to arrival_datetime to datetime object
                 if isinstance(arrival_datetime, str):
                    dt_object = datetime.strptime(arrival_datetime, "%Y-%m-%d %H:%M:%S")
                 else:
                    dt_object = arrival_datetime
                    
                 # Get time parameters
                 week_of_year, day_of_week = dt_object.isocalendar()[1:]
                 day_of_month, day_of_year, year = dt_object.timetuple().tm_mday, dt_object.timetuple().tm_yday, dt_object.timetuple().tm_year
                 minute_of_day = (dt_object.time().hour * 60) + dt_object.time().minute
                 
                 # Check if day was public holiday
                 holiday = list(holidays.GR(years = year).keys())
                 if dt_object.date() in holiday:
                    is_holiday = 1
                 else:
                    is_holiday = 0
                       
                 # Get historical weather data
                 # Round up or down to the closest hour
                 rounded_time = (dt_object + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
                 result = db.weather.find({"municipality": dimos, "timestamp": str(rounded_time)}, {"temperature": 1, "precipitation": 1})
                 for doc in result:
                    temperature, precipitation = doc["temperature"], doc["precipitation"]
                 
                 csv_writer.writerow([route_id, rtype, stop_id, stop_order, minute_of_day, day_of_month, day_of_week, week_of_year, day_of_year, is_holiday, temperature, precipitation, t_pa_in_veh])
              
         subprocess.run(['python3', './inserter.py', '-f', updated_csv_file, '-c', 'ake', '-s', ';'])
		       
def process_files(directory):
    csv_files = find_csv_files(directory)
    for csv_file in csv_files:
        print(csv_file)
        make_dataset(csv_file)
        x = input()

if __name__ == '__main__':
    directory = '/home/raniatze/AKE/' #input('Enter the directory path to search for .csv files: ')
    process_files(directory)
