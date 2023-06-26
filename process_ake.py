# Input:  .csv file from ONE day (ake_data.csv)
# Output: ake_updated.csv file for that day
import os
import csv
import pymongo
import holidays
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from bisect import bisect_left,bisect_right
from time import time
from cityhash import CityHash64
import multiprocessing
import fcntl


num_processors = 6

directory = './AKE/' # input('Enter the directory path to search for .csv files: ')

# Set up the connection to the MongoDB server
client = pymongo.MongoClient('mongodb://localhost:27017/')

# Choose the database and collection to work with
db = client['OASA1']

route_ids = {}
#rtype_ids = {}
stop_ids = {}
value_route_id = 0
#value_rtype = 0
value_stop = 0

files_read = set()

lock_dict = {}
lock_dict['Line_descr_encodings'] = multiprocessing.Lock()
lock_dict['Stop_id_encodings'] = multiprocessing.Lock()
lock_dict['Files_read.txt'] = multiprocessing.Lock()

def find_csv_files(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                if not file.endswith('ake_updated.csv'):
                    csv_files.append(os.path.join(root, file))
    print('Found', len(csv_files), 'files.')
    return csv_files

def file_exists(filename):
    try:
        with open(filename, 'r') as file:
            return True
    except FileNotFoundError:
        return False

def load_encoding_file_data(filename):
    data = {}
    counter = 0
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split(':')
                value = int(value.strip())
                data[key.strip()] = value
                if value > counter:
                    counter = value
    return data, counter

def init_encoding_file(filename):
    if file_exists(filename):
        return load_encoding_file_data(filename)
    else:
        with open(filename, 'w') as file:
            pass
        return {}, 0

def load_files_file_data(filename):
    data = set() 
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                data.add(line)
    return data

#file that remembers which files we have already checked
def init_files_file(filename):
    if file_exists(filename):
        return load_files_file_data(filename)
    else:
        with open(filename, 'w') as file:
            pass
        return set()


def append_to_file(filename, line):
    with open(filename, 'a') as file:
        file_lock = lock_dict[filename]

        with file_lock:
            file.write(line + '\n')

def interpolation(start_datetime, end_datetime, num_points):
    """
    Interpolates between two datetimes with a specified number of points.

    Args:
    start_datetime (datetime): The starting datetime object.
    end_datetime (datetime): The ending datetime object.
    num_points (int): The number of interpolated points, not including start and end points.

    Returns:
    list: A list of interpolated datetime objects.
    """
    if num_points < 2:
        interpolated_datetimes = start_datetime + (end_datetime - start_datetime)/2
        interpolated_datetimes = interpolated_datetimes.strftime('%Y-%m-%d %H:%M:%S.%f')[:-7]
    else:
        delta = (end_datetime - start_datetime) / (num_points + 1)
        interpolated_datetimes = [(start_datetime + i * delta).strftime('%Y-%m-%d %H:%M:%S.%f')[:-7] for i in range(1,num_points+1)]
        #interpolated_datetimes = sorted(interpolated_datetimes)
    return interpolated_datetimes

def extrapolation(datetimes, between_points, extra_points, before):
    #Args:
    #start_datetime: starting datetime
    #end_datetime: ending datetime
    #between_points: number of points between start and end datimes, not including them
    #extra_points: how many points to extrapolate
    #before: true to generate points before start, end to generate points after end

    start_datetime = datetimes[0]
    end_datetime = datetimes[1]

    #hacks
    extra_points = len(extra_points)

    delta = (end_datetime - start_datetime) / (between_points + 1)
    if before:
        extrapolated_datetimes = sorted([(start_datetime - i * delta).strftime('%Y-%m-%d %H:%M:%S.%f')[:-7] for i in range(1,extra_points+1)])
    else:
        extrapolated_datetimes = [(end_datetime + i * delta).strftime('%Y-%m-%d %H:%M:%S.%f')[:-7] for i in range(1,extra_points+1)]

    return extrapolated_datetimes if len(extrapolated_datetimes) > 1 else extrapolated_datetimes[0]

def interpolation_int(start, end, num_points):
    """
    Interpolates between two datetimes with a specified number of points.

    Args:
    start_datetime (datetime): The starting datetime object.
    end_datetime (datetime): The ending datetime object.
    num_points (int): The number of interpolated points, excluding start and end points.

    Returns:
    list: A list of interpolated datetime objects.
    """
    if num_points < 2:
        interpolated = start + (end - start)/2
    else:
        delta = (end - start) / (num_points + 1)
        interpolated = [(start + i * delta) for i in range(1,num_points+1)]
    return interpolated

def extrapolation_int(ints, between_points, extra_points, before):
    #Args:
    #start_datetime: starting datetime
    #end_datetime: ending datetime
    #between_points: number of points between start and end datimes, including them
    #extra_points: how many points to extrapolate
    #before: true to generate points before start, end to generate points after end
    start = ints[0]
    end = ints[1]

    #hacks
    extra_points = len(extra_points)

    delta = (end - start) / (between_points + 1)
    if before:
        extrapolated = [max(0, (start - i * delta)) for i in range(1,extra_points+1)]
        extrapolated.reverse()
    else:
        extrapolated = [max(0, (end + i * delta)) for i in range(1,extra_points+1)]

    return extrapolated if len(extrapolated) > 1 else extrapolated[0]

def find_lt(a, x):
    # Find rightmost value less than x
    i = bisect_left(a, x)
    if i:
        return a[i-1]
    raise ValueError

def find_gt(a, x):
    # Find leftmost value greater than x
    i = bisect_right(a, x)
    if i != len(a):
        return a[i]
    raise ValueError

'''
def update_values(data, values_intp, indices, type_intp, col_name):
    # Updates the dataframe with the interpolated/extrapolated values
    start_idx = indices[0]
    end_idx = indices[1]
    df = pd.DataFrame({col_name: values_intp}, index=range(start_idx,end_idx))
    data.loc[start_idx: end_idx - 1, col_name] = df[col_name]
'''

def update_values(data, values_intp, indices, col_name):
    # Updates the dataframe with the interpolated/extrapolated values
    start_idx = indices[0]
    end_idx = indices[1]
    df = pd.DataFrame({col_name: values_intp}, index=range(start_idx,end_idx))
    data.loc[start_idx: end_idx - 1, col_name] = df[col_name]


# Iterates over each group and interpolates/extrapolates for missing values
flag_exception = False
def group_loop(group_sorted, col_name):
    global flag_exception
    #mask null/not null values of grouped dataframe
    m = group_sorted[col_name].isnull()
    m2 = group_sorted[col_name].notnull()
    m2_indices = np.where(m2)[0]

    # Check if there are enought (>1) values in dataframe column else return
    if len(m2_indices) < 2:
        #raise Exception('Not enough values')
        flag_exception = True
        return

    intp = False
    extp = False
    counter = 0

    # Iterate to interpolate/extrapolate for missing values
    for i, idx in enumerate(group_sorted[m].index):
        # Skip next missing lines
        if counter:
            counter -= 1
            continue
        # Missing value at beggining of group column - extrapolation for first values
        if idx == 0 :
            #print('extp1')
            extp = True
            # Find the index of the previous and next non-null values
            prev = find_gt(m2_indices, idx)
            nxt = find_gt(m2_indices, prev)
        # Missing value at end of group column - extrapolation for last values
        elif idx == m2_indices[-1] + 1:
            # print('extp2')
            nxt = find_lt(m2_indices, idx)
            prev = find_lt(m2_indices,nxt)
        # Missing value in the middle of group column - interpolation
        else:
            #print('intp')
            intp = True
            nxt = find_gt(m2_indices, idx)
            prev = find_lt(m2_indices, idx)

        nxt_dt = group_sorted.loc[nxt, col_name]
        prev_dt = group_sorted.loc[prev,col_name]
        n = nxt - prev-1
        if col_name == 'Arrival_datetime' and prev_dt >= nxt_dt:
            #raise Exception('Wrong data sequence')
            flag_exception = True
            return

        # Interpolate between the previous and next values
        # or extrapolate between the next/last two non-null values
        if col_name == 'Arrival_datetime':
            interpolated = interpolation(prev_dt, nxt_dt, n) if intp else extrapolation([prev_dt,nxt_dt],n,range(0,prev),extp) if extp else extrapolation([prev_dt,nxt_dt],n,range(nxt+1, len(group_sorted)),extp)
        else:
            interpolated = interpolation_int(prev_dt, nxt_dt, n) if intp else extrapolation_int([prev_dt,nxt_dt],n,range(0,prev),extp) if extp else extrapolation_int([prev_dt,nxt_dt],n,range(nxt+1, len(group_sorted)),extp)

        if intp:
            type_intp = 'intp'
            indices = [prev+1, nxt]
            counter += n-1
            intp = False
        elif extp:
            type_intp = 'extp_start'
            indices = [0, prev]
            counter += prev-1
            extp = False
        else:
            type_intp = 'extp_end'
            indices = [nxt+1,len(group_sorted)]
            counter += len(group_sorted)-nxt
        update_values(group_sorted, interpolated, indices, col_name)

# Define a function to query the database for route_id
def get_route_id(route_short_name, route_long_name):
    result = db.routes.find({"route_short_name": route_short_name, "route_long_name": route_long_name}, {"route_id": 1})
    for doc in result:
        return doc["route_id"]
    return None

#not used
route_cache = {}
def cache_route_id():
    global route_cache
    quer = db.routes.find({}, {'route_short_name': 1, 'route_long_name': 1,'route_id': 1})
    for entry in quer:
        name = entry['route_short_name'] + ' - ' + entry['route_long_name'] 
        clean_name = clean_line_descr(name)
        route_cache[clean_name] = entry['route_id']

#not used
# Define a function to query the database for municipality
def get_dimos(stop_id):
    result = db.staseis_dimoi.find({"stop_id": str(stop_id)}, {"dimos": 1})
    for doc in result:
        return doc["dimos"]
    return None

dimos_cache = {}
def cache_dimos():
    global dimos_cache
    quer = db.staseis_dimoi.find({}, {"stop_id": 1, "dimos": 1})
    for entry in quer:
        dimos_cache[entry['stop_id']] = entry['dimos']


# Define a function to get historical weather data
def get_weather_data(dimos, timestamp):
    result = db.weather.find({"municipality": dimos, "timestamp": str(timestamp)}, {"temperature": 1, "precipitation": 1})
    for doc in result:
        return doc
    return None

# Define a function to check if a datetime is a public holiday
def is_holiday_check(dt, holidays_list):
    return dt.date() in holidays_list

#not used
def fill_na_dimoi(d):
    # Find non-null values in the dictionary
    non_null_values = [v for v in d.values() if v is not None]
    if len(non_null_values) == 0:
        raise Exception('Dimoi list is null')
    # Iterate over the items of the dictionary and replace None values with non-null values
    for k, v in d.items():
        if v is None:
            d[k] = non_null_values[0]

def clean_line_descr(line_descr):
    #return ''.join(s.strip() for s in line_descr.split('-'))
    return line_descr.replace(' ', '')

def make_dataset(csv_file):

    global route_ids
    #global rtype_ids
    global stop_ids
    global value_route_id
    #global value_rtype
    global value_stop
    global flag_exception

    column_types = {
            "Line_descr": str,
            "Rtype": str,
            "Arrival_datetime": str,
            "Stop_id": str
    }

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file, sep=';', dtype=column_types)
    df['Clean_Line_descr'] = df['Line_descr'].apply(clean_line_descr)

    # Group the data by Line_descr, Rtype, Direction, Sched, and Vehicle_no
    grouped = df.groupby(['Line_descr', 'Direction', 'Sched', 'Vehicle_no'])

    # Split the file path into directory and filename components
    dir_path, _ = os.path.split(csv_file)
    updated_csv_file = os.path.join(dir_path, 'ake_updated.csv')

    with open(updated_csv_file, 'w') as output_file:

        # Write header row in output_file
        headers = pd.DataFrame(columns=['Line_descr', 'Direction', 'Stop_id', 'Stop_order', 'Minute_of_day', 'Day_of_month', 'Day_of_week', 'Week_of_year', 'Day_of_year', 'Is_holiday', 'Temperature', 'Precipitation', 'T_pa_in_veh', 'Sched', 'Year'])
        headers.to_csv(output_file, header = 'True', index=False, mode = 'w')

        # Iterate over each group
        # Iterate over the groups and process each row
        #counterrr = 0
        for name, group in grouped:
            #counterrr += 1
            #if counterrr > 4:
            #    break
            print(name)

            # Sort the entries of the group by S_order column in ascending order
            group_sorted = group.sort_values('S_order', ascending=True)
            group_sorted = group_sorted.reset_index()
            group_sorted['Arrival_datetime'] = pd.to_datetime(group_sorted['Arrival_datetime'], format='%Y-%m-%d %H:%M:%S')

            
            group_sorted['Stop_id'] = group_sorted['Stop_id'].str.strip().str.lstrip('0')

            # Drop null rows in 'Arrival_datetime' column
            group_sorted_full = group_sorted.dropna(subset=['Arrival_datetime'])

            # Check if the dataframe is sorted in ascending order on 'Arrival_datetime' column
            correct_sequence = group_sorted_full['Arrival_datetime'].equals(group_sorted_full['Arrival_datetime'].sort_values())

            if not correct_sequence:
                print("WRONG GROUP")
                continue

            group_loop(group_sorted,'Arrival_datetime')
            if flag_exception:
                flag_exception = False
                continue

            group_loop(group_sorted, 'T_pa_in_veh')
            if flag_exception:
                flag_exception = False
                continue

            #print(group_sorted[['Line_descr', 'Clean_Line_descr']])
            # Get unique values for line_descr and stop_id
            unique_line_descr = group_sorted['Clean_Line_descr'].unique()
            unique_stop_id = group_sorted['Stop_id'].unique()

            #print(unique_line_descr)
            # Query the database for route_id and dimos
            #route_id_dict = {ld: get_route_id(*ld.split(" - ",1)) for ld in unique_line_descr}
            #print(route_id_dict)

            if unique_line_descr[0] not in route_ids:
                route_ids[unique_line_descr[0]] = value_route_id
                #print(unique_line_descr[0], value_route_id)
                line_to_write = unique_line_descr[0] + ': ' + str(value_route_id)
                append_to_file('Line_descr_encodings', line_to_write)
                value_route_id += 1
            
            """
            if group_sorted['Rtype'][0] not in rtype_ids:
                rtype_ids[group_sorted['Rtype'][0]] = value_rtype
                value_rtype += 1
            """

            for stop in unique_stop_id:
                if stop not in stop_ids:
                    stop_ids[stop] = value_stop
                    line_to_write = str(stop) + ': ' + str(value_stop)
                    append_to_file('Stop_id_encodings', line_to_write)
                    value_stop += 1

            group_sorted['Custom_stop_id'] = group_sorted['Stop_id'].replace(stop_ids)
            #print(group_sorted[['Stop_id','Custom_stop_id']])
            # print('Route OK')
            #print(unique_stop_id)
            #print(dimos_cache)
            dimos_dict = {si: get_dimos(si) for si in unique_stop_id}
            #print(dimos_dict)
            #dimos_dict = { for si in unique_stop_id}
            #print(dimos_dict)
            '''
            try:
                fill_na_dimoi(dimos_dict)
            except Exception:
                continue
            '''
            # print('Dimos OK')

            # Get time parameters
            datetimes = pd.to_datetime(group_sorted['Arrival_datetime'])
            day_of_week = datetimes.dt.isocalendar().day
            week_of_year = datetimes.dt.isocalendar().week
            day_of_month = datetimes.dt.day
            day_of_year = datetimes.dt.dayofyear
            year = datetimes.dt.year
            # total_secs = datetimes.dt.second + datetimes.dt.minute * 60
            # total_secs_rounded = (total_secs / 60).apply(lambda x: round(x)) * 60
            # minute_of_day = datetimes + pd.to_timedelta(total_secs_rounded - total_secs, unit='s')
            minute_of_day = datetimes.dt.hour * 60 + datetimes.dt.minute
            # print('Dates OK')
            # Check if day was public holiday
            holidays_list = holidays.GR(years=year).keys()
            is_holiday = datetimes.apply(lambda x: is_holiday_check(x, holidays_list)).astype(int)
            #is_holiday = 1 if is_holiday else 0
            # print('Holidays OK')
            # Get historical weather data
            rounded_time = datetimes.apply(lambda x: pd.to_datetime(x) + timedelta(minutes=30))
            rounded_time = rounded_time.apply(lambda x: x.replace(minute=0, second=0, microsecond=0))
            # Get unique dimos and rounded timestamps
            unique_dimos = set(val for val in dimos_dict.values())
            unique_timestamps = rounded_time.unique()
            unique_timestamps = [str(pd.to_datetime(ts)) for ts in unique_timestamps]

            # Query the database for all relevant weather data
            weather_data_dict = {}
            for doc in db.weather.find({"municipality": {"$in": list(unique_dimos)}, "timestamp": {"$in": unique_timestamps}}, {"municipality": 1, "timestamp": 1, "temperature": 1, "precipitation": 1}):
                key = (doc['municipality'], pd.Timestamp(doc['timestamp']))
                weather_data_dict[key] = {'temperature': doc['temperature'], 'precipitation': doc['precipitation']}
                
            if weather_data_dict == {}:
               print("EMPTY WEATHER DATA DICT")
               exit()

            # Get the weather data for each row
            weather_data = []
            for si, rt in zip(group_sorted['Stop_id'], rounded_time):
                key = (dimos_dict[si], rt)
                weather_data.append(weather_data_dict.get(key, None))
            # Get the temperature and precipitation values
            temperature = [wd['temperature'] if wd else None for wd in weather_data]
            precipitation = [wd['precipitation'] if wd else None for wd in weather_data]
            # print('Weather OK')

            #dataframe_string = group.to_string()
            dataframe_string = str(name[:3])
            #unique_hash = CityHash64(dataframe_string)

            # Get the data in the desired format and write to CSV
            data = {
                'Line_descr': route_ids[unique_line_descr[0]],
                #'Rtype': rtype_ids[group_sorted['Rtype'][0]],
                'Direction': group_sorted['Direction'].astype('int'),
                'Stop_id': group_sorted['Custom_stop_id'],
                'Stop_order': group_sorted['S_order'],
                'Minute_of_day': minute_of_day,
                'Day_of_month': day_of_month,
                'Day_of_week': day_of_week,
                'Week_of_year': week_of_year,
                'Day_of_year': day_of_year,
                'Is_holiday': is_holiday,
                'Temperature': temperature,
                'Precipitation': precipitation,
                'T_pa_in_veh': group_sorted['T_pa_in_veh'].astype('int'),
                'Sched': group_sorted['Sched'],
                'Year': year
            }
            # Convert the data to a DataFrame and write to CSV
            df = pd.DataFrame(data)
            df.to_csv(output_file, mode='a', index=False, header=False)

    # subprocess.run(['python3', './inserter.py', '-f', updated_csv_file, '-c', 'ake', '-s', ';'])

def process_files(csv_files):
    for csv_file in csv_files:
        print(csv_file)
        if csv_file in files_read:
            print("Already read", csv_file)
            continue
        make_dataset(csv_file)
        append_to_file('Files_read.txt', csv_file)
        #x = input()

def process_file(csv_file):
    print(csv_file)
    if csv_file in files_read:
        print("Already read", csv_file)
        return
    make_dataset(csv_file)
    append_to_file('Files_read.txt', csv_file)

if __name__ == '__main__':
    #cache_route_id()
    #cache_dimos()
    route_ids, value_route_id = init_encoding_file('Line_descr_encodings')
    stop_ids, value_stop = init_encoding_file('Stop_id_encodings')

    files_read = init_files_file('Files_read.txt')
    
    '''
    manager = multiprocessing.Manager()
    s_route_ids = manager.dict(route_ids)
    s_stop_ids = manager.dict(stop_ids)
    
    s_value_route_id = manager.Value('i', value_route_id)
    s_value_stop = manager.Value('i', value_stop)
    '''

    csv_files = find_csv_files(directory)


    pool = multiprocessing.Pool(processes=num_processors)
    pool.map(process_file, csv_files)
    #process_file(csv_files)

    pool.close()
    pool.join()
