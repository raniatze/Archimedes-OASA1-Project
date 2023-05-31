import os
import csv
import pandas as pd
import numpy as np
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")  # Replace with your MongoDB connection details
db = client["OASA1"]  # Replace with your database name

def copy_rows(df, num_missing_rows):

    print("Missing rows: ", num_missing_rows)
    # Copy the missing rows by duplicating the last available row
    last_row = df.loc[df.index[-1]]
    copied_rows = pd.concat([last_row] * num_missing_rows, axis=1).transpose()

    # Append the copied rows to the dataframe
    df = pd.concat([df, copied_rows], ignore_index=True)
    return df

def get_previous_stops(m, index, df):
    result_df = pd.DataFrame()
    # Find current_day_of_year and current_year
    current_day_of_year, current_year = df.iloc[index]['Day_of_year'], df.iloc[index]['Year']
    print("Current Row")
    print(pd.DataFrame(df.iloc[index]).transpose())
    for i in range(m):
        previous_row = df.iloc[index - (i+1)]
        previous_day_of_year, previous_year = previous_row['Day_of_year'], previous_row['Year']
        if previous_day_of_year == current_day_of_year and previous_year == current_year:
           result_df = pd.concat([result_df, pd.DataFrame([previous_row])], ignore_index=True)

    return result_df

def get_previous_days(n, index, df, stops_dict):
    result_df = pd.DataFrame()
    # Find current_day_of_year and current_year
    current_stop_id, current_stop_order = df.iloc[index]['Stop_id'], df.iloc[index]['Stop_order']

    if (current_stop_id, current_stop_order) not in stops_dict:
       stops_dict[(current_stop_id, current_stop_order)] = df[(df['Stop_id'] == current_stop_id) & (df['Stop_order'] == current_stop_order)]

    filtered_df = stops_dict[(current_stop_id, current_stop_order)]
    filtered_df = filtered_df.reset_index()
    #print("Current Row")
    #print(pd.DataFrame(df.iloc[index]).transpose())
    #print("Filtered Df")
    #print(filtered_df)
    new_index = filtered_df.index[filtered_df['index'] == index].tolist()[0]
    filtered_df.drop(['index'], axis=1, inplace=True)
    if new_index >= n:
       for i in range(n):
         previous_row = filtered_df.iloc[new_index - (i+1)]
         result_df = pd.concat([result_df, pd.DataFrame([previous_row])], ignore_index=True)

    return result_df

def create_input_sequences(num_line_descr, line_descr_df, m, n):

    folder_path = 'LSTM_Dataset_' + num_line_descr
    os.makedirs(folder_path)

    with open(os.path.join(folder_path, 'inputs.csv'), 'w') as input_file, open(os.path.join(folder_path, 'targets.csv'), 'w') as target_file:

         csv_input_writer, csv_target_writer = csv.writer(input_file, delimiter=','), csv.writer(target_file)

         # Write header row in output_file
         csv_input_writer.writerow(['Line_descr', 'Direction', 'Stop_id', 'Stop_order', 'Minute_of_day', 'Day_of_month', 'Day_of_week', 'Week_of_year', 'Day_of_year', 'Is_holiday', 'Temperature', 'Precipitation', 'T_pa_in_veh'])
         csv_target_writer.writerow(['T_pa_in_veh'])

         # Group the data by Day_of_year
         grouped_data = line_descr_df.groupby(['Direction', 'Sched'])

         for name, group in grouped_data:
            print("################################################### NEW GROUP #####################################")
            print(name)
            stops_dict = {} # Dictionary with (stop_id, stop_order) keys and filtered_df values
            group_sorted = group.sort_values(['Year', 'Day_of_year', 'Stop_order'], ascending=[True, True, True])
            group_sorted = group_sorted.reset_index(drop=True)
            # Get the sequence length for the current Day_of_year group
            group_length = len(group_sorted)
            if group_length > m:

                # Iterate over the group, starting from sequence_length index
                for i in range(m, group_length):

                    # Get the previous stops and days for the current instance
                    previous_stops, previous_days = get_previous_stops(m, i, group_sorted), get_previous_days(n, i, group_sorted, stops_dict)
                    num_previous_stops, num_previous_days = len(previous_stops), len(previous_days)
                    if num_previous_stops == 0 or num_previous_days == 0:
                       print("Skip current row")
                       continue
                    else:
                       previous_stops = previous_stops.sort_values(['Stop_order'], ascending = True).reset_index(drop=True)
                       previous_days = previous_days.sort_values(['Year', 'Day_of_year'], ascending = [True, True]).reset_index(drop=True)
                       num_missing_stops, num_missing_days = m - num_previous_stops, n - num_previous_days

                       if num_missing_stops != 0:
                          new_previous_stops = copy_rows(previous_stops, num_missing_stops)
                       else:
                          new_previous_stops = previous_stops

                       if num_missing_days != 0:
                          new_previous_days = copy_rows(previous_days, num_missing_days)
                       else:
                          new_previous_days = previous_days
                       #print("Previous stops")
                       #print(new_previous_stops)
                       #print("Previous days")
                       #print(new_previous_days)

                    # Combine the previous stops and previous days' stops
                    inputs = pd.concat([new_previous_stops, new_previous_days], axis=0).reset_index(drop=True)
                    inputs.drop(['Sched', 'Year'], axis=1, inplace=True)
                    print("Input")
                    print(inputs)
                    if inputs.shape != (5,13):
                       print(" WRONG ")
                       print(inputs)
                       x = input()

                    # Get the target value (T_pa_in_veh) for the current instance
                    target = group_sorted.iloc[i]['T_pa_in_veh']
                    print("Target")
                    print(target)

                    csv_input_writer.writerows(inputs.values)
                    csv_target_writer.writerow([target])
            else:
                continue
    return

def filter_line_descr(line_descr):
    # Find the matching documents
    pipeline = [
        {
            "$match": {
                "Line_descr": line_descr
            }
        }
    ]
    result = db.ake.aggregate(pipeline)

    # Return the query result
    return list(result)

# Example usage
m = 3 # previous stop_orders
n = 2 # previous days
num_line_descr = "1"

filtered_data = filter_line_descr(num_line_descr)
line_descr_df = pd.DataFrame(filtered_data)
line_descr_df.drop(['_id'], axis=1, inplace=True)
line_descr_df['Stop_id'] = line_descr_df['Stop_id'].astype(int)
line_descr_df['Stop_order'] = line_descr_df['Stop_order'].astype(int)
line_descr_df['Day_of_year'] = line_descr_df['Day_of_year'].astype(int)
line_descr_df['Minute_of_day'] = line_descr_df['Minute_of_day'].astype(int)
line_descr_df['T_pa_in_veh'] = line_descr_df['T_pa_in_veh'].astype(int)
line_descr_df['Year'] = line_descr_df['Year'].astype(int)

create_input_sequences(num_line_descr, line_descr_df, m, n)
