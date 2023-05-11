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

# Set up the connection to the MongoDB server
client = pymongo.MongoClient('mongodb://localhost:27017/')

# Choose the database and collection to work with
db = client['OASA1']

route_ids = {}
rtype_ids = {}
value_route_id = 0
value_rtype = 0

def find_csv_files(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                if not file.endswith('ake_updated.csv'):
                    csv_files.append(os.path.join(root, file))
    return csv_files
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

def update_values(data, values_intp, indices, type_intp, col_name):
    # Updates the dataframe with the interpolated/extrapolated values
    start_idx = indices[0]
    end_idx = indices[1]
    df = pd.DataFrame({col_name: values_intp}, index=range(start_idx,end_idx))
    data.loc[start_idx: end_idx - 1, col_name] = df[col_name]

# Iterates over each group and interpolates/extrapolates for missing values
def group_loop(group_sorted, col_name):

    #mask null/not null values of grouped dataframe
    m = group_sorted[col_name].isnull()
    m2 = group_sorted[col_name].notnull()
    m2_indices = np.where(m2)[0]

    # Check if there are enought (>1) values in dataframe column else return
    if len(m2_indices) < 2:
        raise Exception('Not enough values')
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
            raise Exception('Wrong data sequence')
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
        update_values(group_sorted, interpolated, indices,  type_intp, col_name)

# Define a function to query the database for route_id
def get_route_id(route_short_name, route_long_name):
    result = db.routes.find({"route_short_name": route_short_name, "route_long_name": route_long_name}, {"route_id": 1})
    for doc in result:
        return doc["route_id"]
    return None

# Define a function to query the database for municipality
def get_dimos(stop_id):
    result = db.staseis_dimoi.find({"stop_id": str(stop_id)}, {"dimos": 1})
    for doc in result:
        return doc["dimos"]
    return None

# Define a function to get historical weather data
def get_weather_data(dimos, timestamp):
    result = db.weather.find({"municipality": dimos, "timestamp": str(timestamp)}, {"temperature": 1, "precipitation": 1})
    for doc in result:
        return doc
    return None

# Define a function to check if a datetime is a public holiday
def is_holiday_check(dt, holidays_list):
    return dt.date() in holidays_list

def fill_na_dimoi(d):
    # Find non-null values in the dictionary
    non_null_values = [v for v in d.values() if v is not None]
    if len(non_null_values) == 0:
        raise Exception('Dimoi list is null')
    # Iterate over the items of the dictionary and replace None values with non-null values
    for k, v in d.items():
        if v is None:
            d[k] = non_null_values[0]

def make_dataset(csv_file):

    global route_ids
    global rtype_ids
    global value_route_id
    global value_rtype

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file, sep=';')

    # Group the data by Line_descr, Rtype, Direction, Sched, and Vehicle_no
    grouped = df.groupby(['Line_descr', 'Rtype', 'Direction', 'Sched', 'Vehicle_no'])

    # Split the file path into directory and filename components
    dir_path, _ = os.path.split(csv_file)
    updated_csv_file = os.path.join(dir_path, 'ake_updated.csv')

    with open(updated_csv_file, 'w') as output_file:

        # Write header row in output_file
        headers = pd.DataFrame(columns=['Route_id', 'Rtype', 'Stop_id', 'Stop_order', 'Minute_of_day', 'Day_of_month', 'Day_of_week', 'Week_of_year', 'Day_of_year', 'Is_holiday', 'Temperature', 'Precipitation', 'T_pa_in_veh', 'Unique_hash'])
        headers.to_csv(output_file, header = 'True', index=False, mode = 'w')

        # Iterate over each group
        # Iterate over the groups and process each row
        for name, group in grouped:
            print(name)

            # Sort the entries of the group by S_order column in ascending order
            group_sorted = group.sort_values('S_order', ascending=True)
            group_sorted = group_sorted.reset_index()
            group_sorted['Arrival_datetime'] = pd.to_datetime(group_sorted['Arrival_datetime'])
            try:
                group_loop(group_sorted,'Arrival_datetime')
            except Exception:
                continue
            try:
                group_loop(group_sorted, 'T_pa_in_veh')
            except Exception:
                continue
            # Get unique values for line_descr and stop_id
            unique_line_descr = group_sorted['Line_descr'].unique()
            unique_stop_id = group_sorted['Stop_id'].unique()

            # Query the database for route_id and dimos
            route_id_dict = {ld: get_route_id(*ld.split(" - ",1)) for ld in unique_line_descr}

            if unique_line_descr[0] not in route_ids:
                route_ids[unique_line_descr[0]] = value_route_id
                value_route_id += 1

            if group_sorted['Rtype'][0] not in rtype_ids:
                rtype_ids[group_sorted['Rtype'][0]] = value_rtype
                value_rtype += 1
            # print('Route OK')

            dimos_dict = {si: get_dimos(si) for si in unique_stop_id}
            try:
                fill_na_dimoi(dimos_dict)
            except Exception:
                continue
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
            dataframe_string = str(name)
            unique_hash = CityHash64(dataframe_string)

            # Get the data in the desired format and write to CSV
            data = {
                'Route_id': route_ids[unique_line_descr[0]],
                'Rtype': rtype_ids[group_sorted['Rtype'][0]],
                'Stop_id': group_sorted['Stop_id'],
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
                'Unique_hash': unique_hash
            }
            # Convert the data to a DataFrame and write to CSV
            df = pd.DataFrame(data)
            df.to_csv(output_file, mode='a', index=False, header=False)

    # subprocess.run(['python3', './inserter.py', '-f', updated_csv_file, '-c', 'ake', '-s', ';'])

def process_files(directory):
    csv_files = find_csv_files(directory)
    for csv_file in csv_files:
        print(csv_file)
        make_dataset(csv_file)
        #x = input()

if __name__ == '__main__':
    directory = './AKE/2021/2021/04/2021-04-13_akedata/' #input('Enter the directory path to search for .csv files: ')
    process_files(directory)
