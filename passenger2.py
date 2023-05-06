#! /bin/bash
# Input:  .csv file from ONE day (ake_data.csv)
# Output: ake_updated.csv file for that day
import os
import csv
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.impute import SimpleImputer

def find_csv_files(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def interpolation(start_datetime, end_datetime, num_points):
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
    #between_points: number of points between start and end datimes, including them
    #extra_points: how many points to extrapolate
    #before: true to generate points before start, end to generate points after end
    start_datetime = datetimes[0]
    end_datetime = datetimes[1]

    #hacks
    between_points = between_points[1] - between_points[0] + 1
    extra_points = len(extra_points)

    delta = (end_datetime - start_datetime) / (between_points - 1)
    if before:
        extrapolated_datetimes = [(start_datetime - i * delta).strftime('%Y-%m-%d %H:%M:%S.%f')[:-7] for i in range(1,extra_points+1)]
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
    between_points = between_points[1] - between_points[0] + 1
    extra_points = len(extra_points)

    delta = (end - start) / (between_points - 1)
    if before:
        extrapolated = [max(0, (start - i * delta)) for i in range(1,extra_points+1)]
        extrapolated.reverse()
    else:
        extrapolated = [max(0, (end + i * delta)) for i in range(1,extra_points+1)]



    return extrapolated if len(extrapolated) > 1 else extrapolated[0]

# 1: write updated rows to csv
# 2: fill the missing values in df
def update_values(df, stops_missing, values, typee):
    length = len(stops_missing)
    for count, value  in enumerate(stops_missing):
        if length == 1:
            df.loc[value-1, 'T_pa_in_veh'] = int(values)
        else:
            df.loc[value-1, 'T_pa_in_veh'] = int(values[count])
        df.loc[value-1, 'Type_intp'] = typee

def make_dataset(csv_file):

    #Determine target csv_file
    target = './outputs2/test.csv'# + str(csv_file.split('.')[0].split('/')[-2]) + '_output.csv'

    column_types = {
        "Line_descr": str,
        "Rtype": str,
        "Arrival_datetime": str,
        "Stop_id": str
    }

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file, sep=';', dtype=column_types)
    # Group the data by Line_descr, Rtype, Direction, Sched, and Vehicle_no
    grouped = df.groupby(['Line_descr', 'Rtype', 'Direction', 'Sched', 'Vehicle_no'])

    if not os.path.isfile(target):

        with open(target, 'w',newline='') as output_file:

            headers = pd.DataFrame(columns=['Name','Stop_order','Arrival_datetime','Type_intp','Passengers','Changed'])
            headers.to_csv(target, header = 'True', index=False, mode = 'w')
            # Write header row in output_file
            # Iterate over each group
            for name, group in grouped:
                #print(name)
                # Sort the entries of the group by S_order column in ascending order
                group_sorted = group.sort_values('S_order', ascending=True)
                group_sorted = group_sorted.reset_index()
                group_sorted['Type_intp'] = ''

                extp = 0
                intp = False
                next = None
                prev = None
                stop_counter_extp = 0
                stop_counter_intp = 0
                n = len(group)
                stop_i = 0

                if len(list(group_sorted.iterrows())) < 2:
                    continue

                group_sorted['Changed'] = group_sorted['T_pa_in_veh'].isnull()

                # Iterate over each row of the sorted group
                for i, row in group_sorted.iterrows():
                    now = row['T_pa_in_veh']
                    now_res = now
                    #Stop has no arrival_datetime
                    if pd.isna(now):
                        #first stop(s)
                        if i == 0 or extp == 1:
                            stop_counter_extp += 1
                            extp = 1
                        #last stop(s)
                        elif i == n - 1 :
                            #print('extp_end')
                            stop_counter_extp = stop_counter_intp + 1
                            if next == prev:
                                try:
                                    prev = pd.to_datetime(group_sorted.loc[stop_i-2,'T_pa_in_veh'])
                                except:
                                    print('not enough values to extrapolate 1', name[0])
                                    break
                            now_res = extrapolation_int([prev,next],[stop_i-2, stop_i-1], range(n - stop_counter_extp +1, n + 1), False)
                            if now_res != -1:
                                #fill all missing values of interpolation (end of route)
                                update_values(group_sorted, range(stop_i+1, n+1), now_res,'extp_end')
                            else:
                                print('not enough values to extrapolate 2a', name[0])
                                break
                        #middle stop(s)
                        else:
                            stop_counter_intp += 1
                            intp = True
                    else:
                        if extp:
                            if prev == None:
                                #print('extp_start1')
                                prev = now
                                extp = 2
                                stop_i += stop_counter_extp + 1
                            else:
                                #print('extp_start2')
                                next = now
                                stop_ord_1 = stop_counter_extp + 1
                                stop_ord_2 = stop_ord_1 + stop_counter_intp + 1
                                now_res = extrapolation_int([prev,next],[stop_ord_1, stop_ord_2], range(0,stop_counter_extp), True)
                                if now_res != -1:
                                    #fill all missing values of extrapolation (start of route)
                                    update_values(group_sorted,  range(1,stop_counter_extp+1), now_res, 'extp_start')
                                else:
                                    print('not enough values to extrapolate 2b', name[0])
                                    break

                                stop_counter_extp = 0
                                extp = 0
                                stop_i += 1
                                if intp:
                                    #print('intp-extp')
                                    now_res = interpolation_int(prev, next, stop_counter_intp)
                                    #fill all missing values of interpolation exactly after extrapolation (start of route)
                                    update_values(group_sorted,  range(stop_i, stop_i+stop_counter_intp), now_res, 'intp-extp')
                                    stop_i += stop_counter_intp
                                    stop_counter_intp = 0
                                    prev = now
                                    intp = False
                        elif intp:
                            #print('intp')
                            stop_i += stop_counter_intp + 1
                            next = now
                            now_res = interpolation_int(prev, next, stop_counter_intp)
                            #fill all missing values of interpolation (middle values)
                            update_values(group_sorted,  range(stop_i - stop_counter_intp, stop_i), now_res, 'intp')
                            stop_counter_intp = 0
                            prev = now
                            intp = False
                        else:
                            #print('standard')
                            next = now
                            prev = now
                            stop_i += 1

                        #fill with current stop values
                        update_values(group_sorted,  range(stop_i,stop_i+1), now, 'standard')

                group_sorted = group_sorted.sort_values('S_order', ascending=True)
                group_sorted = group_sorted.reset_index()
                group_sorted['T_pa_in_veh'] = group_sorted['T_pa_in_veh'].astype(int)
                group_sorted[['Line_descr','S_order','Arrival_datetime','Type_intp','T_pa_in_veh','Changed']].to_csv(target, mode='a', index=False, header=False)
                #group_sorted.to_csv(target, mode='a', index=False, header=False)
    else:
        print('File already exists..')


def process_files():
    csv_files = find_csv_files(directory)
    #a = './AKE/2021/2021/12/2021-12-22_akedata/ake_data.txt'
    a = './AKE/2021/2021/12/2021-12-08_akedata/ake_data.txt'
    make_dataset(a)
    """
    for csv_file in csv_files:
        print(csv_file)
        make_dataset(csv_file)
        # x = input()
    """

if __name__ == '__main__':
    directory = './AKE/' #input('Enter the directory path to search for .csv files: ')
    process_files()
