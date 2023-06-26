import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
from tensorflow.keras.models import load_model
import joblib
import random
import holidays
import matplotlib.pyplot as plt

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")  # Replace with your MongoDB connection details
db = client["OASA1"]  # Replace with your database name

def copy_rows(df, num_missing_rows):

    # Copy the missing rows by duplicating the last available row
    last_row = df.loc[df.index[-1]]
    copied_rows = pd.concat([last_row] * num_missing_rows, axis=1).transpose()

    # Append the copied rows to the dataframe
    df = pd.concat([df, copied_rows], ignore_index=True)
    return df


def get_previous_days(n, current_stop_order, current_stop_id, df, stops_dict):
    result_df = pd.DataFrame()
    if (current_stop_id, current_stop_order) not in stops_dict:
        stops_dict[(current_stop_id, current_stop_order)] = df[(df['Stop_id'] == current_stop_id) & (df['Stop_order'] == current_stop_order)]

    filtered_df = stops_dict[(current_stop_id, current_stop_order)]
    filtered_df = filtered_df.reset_index()
    filtered_df.drop(['index'], axis=1, inplace=True)
    if filtered_df.shape[0] == 1:
        result_df = copy_rows(filtered_df, n-1)
    else:
        for i in range(n):
            previous_row = filtered_df.iloc[-(i+1)]
            result_df = pd.concat([result_df, pd.DataFrame([previous_row])], ignore_index=True)
    return result_df

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

def filter_line_encoding(line_encoding):
    # Find the matching documents
    pipeline = [
            {
                "$match": {
                    "Line_encoding": line_encoding
                    }
                }
            ]
    result = db.stops_by_line.aggregate(pipeline)

    # Return the query result
    return list(result)

def is_holiday(current_date):
    if current_date in holidays_list:
        return 1
    else:
        return 0

# Define a function to query the database for municipality
def get_dimos(stop_encoding):
    result = db.staseis_dimoi.find({"stop_encoding": str(stop_encoding)}, {"dimos": 1})
    for doc in result:
        return doc["dimos"]
    return None

# Define a function to get historical weather data
def get_weather_data(dimos, timestamp):
    result = db.weather_today.find({"municipality": dimos, "timestamp": str(timestamp)}, {"temperature": 1, "precipitation": 1})
    for doc in result:
        return doc
    return None

# Example usage
m = 3 # previous stop_orders
n = 2 # previous days

category_0 = ['0', '1', '139', '140', '4', '5', '143', '6', '144', '8', '9', '188', '145', '191', '172', '15', '20', '173', '22', '205']

category_1 = ['2', '217', '10', '12', '16', '18', '174', '193', '33', '39', '194', '58', '68', '70', '225', '90', '92', '179', '113']

category_2 = ['11', '189', '13', '190', '192', '14', '181', '146', '307', '19', '21', '200', '209', '24', '175', '148', '182', '210', '38', '40']

category_3 = ['114', '115', '116', '169', '117', '118', '170', '119', '120', '230', '122', '123', '124', '125', '199', '127', '128', '129', '130']

category_4 = ['171', '17', '49', '50', '152', '65', '74', '82', '196', '107', '108']

category_5 = ['142', '37', '226', '223', '331', '237', '206', '204', '231', '131', '132', '133', '134', '135', '244'] # 332 was empty

merged_list = []

merged_list.extend(category_0)
#merged_list.extend(category_1)
#merged_list.extend(category_2)
#merged_list.extend(category_3)
#merged_list.extend(category_4)
#merged_list.extend(category_5)

current_date = datetime.now()
day_of_week = current_date.isocalendar()[2]
week_of_year = current_date.isocalendar()[1]
day_of_month = current_date.day
day_of_year = current_date.timetuple().tm_yday
year = current_date.year
holidays_list = holidays.GR(years=year).keys()
current_date = pd.to_datetime(current_date.strftime('%Y-%m-%d'))

for line_encoding in merged_list:

    result = db.line_categories.find({"Line_encoding": line_encoding}, {"Category": 1})
    for doc in result:
        category = doc["Category"]

    print("Line encoding {line_encoding} belongs to category {category}.".format(line_encoding=line_encoding, category=category))

    joblib_path = 'filter_ake_' + line_encoding + '.joblib'
    # Check if files exist
    if os.path.exists(joblib_path):
        # If files exist, load them
        filtered_data = joblib.load(joblib_path)
    else:
        filtered_data = filter_line_descr(line_encoding)
        joblib.dump(filtered_data, joblib_path)

    line_ake_df = pd.DataFrame(filtered_data)
    line_ake_df.drop(['_id'], axis=1, inplace=True)
    line_ake_df['Stop_id'] = line_ake_df['Stop_id'].astype(int)
    line_ake_df['Stop_order'] = line_ake_df['Stop_order'].astype(int)
    line_ake_df['Day_of_year'] = line_ake_df['Day_of_year'].astype(int)
    line_ake_df['Day_of_week'] = line_ake_df['Day_of_week'].astype(int)
    line_ake_df['Minute_of_day'] = line_ake_df['Minute_of_day'].astype(int)
    line_ake_df['T_pa_in_veh'] = line_ake_df['T_pa_in_veh'].astype(int)
    line_ake_df['Year'] = line_ake_df['Year'].astype(int)

    filtered_data = filter_line_encoding(line_encoding)
    line_stops_df = pd.DataFrame(filtered_data)
    line_stops_df.drop(['_id'], axis=1, inplace=True)
    line_stops_df['Stop_order'] = line_stops_df['Stop_order'].astype(int)
    line_stops_df['Stop_id'] = line_stops_df['Stop_id'].astype(int)


    # Group the data by Day_of_year
    grouped_data = line_ake_df.groupby(['Direction','Sched'])

    file = './Checkpoints/Category_{category}_best/best_model.h5'.format(category=category)
    model = load_model(file)
    predictions = pd.DataFrame()

    for name, group in grouped_data:
        print(name)
        flag = False
        direction = group.iloc[0]['Direction']

        stops_dict = {}
        stops_predicted = []
        stops_predicted_df = pd.DataFrame()

        previous_days, previous_stops = pd.DataFrame(), pd.DataFrame()

        group_sorted = group.sort_values(['Year', 'Day_of_year', 'Stop_order'], ascending=[True, True, True])
        group_sorted = group_sorted.reset_index(drop=True)

        group_counts = group_sorted.groupby(['Stop_order', 'Stop_id']).size().reset_index(name='count')
        max_counts = group_counts.groupby('Stop_order')['count'].transform(max)
        filtered_group = group_counts[group_counts['count'] == max_counts]
        unique_combinations = filtered_group[['Stop_id', 'Stop_order']].drop_duplicates()

        #find unique stop encodings
        unique_stop_encodings = group_sorted['Stop_id'].astype(float).unique()

        # create dimos_dict for unique stops
        dimos_dict = {si: get_dimos(si) for si in unique_stop_encodings}

        # Iterate over the group, starting from the first stop
        for combination in unique_combinations.values:
            stop_encoding, stop_order = combination[0], combination[1] # int, int
            print("Stop_order: ", stop_order)

            if stop_order not in [1,2,3]:
               # get previous days
               previous_days = get_previous_days(n, stop_order, stop_encoding, group_sorted, stops_dict)
               previous_days = previous_days.sort_values(['Year', 'Day_of_year'], ascending = [True, True]).reset_index(drop=True)
               previous_days.drop(['Sched', 'Year'], axis=1, inplace=True)
               previous_stops = [stops_predicted[-m+j] for j in range(0,m)]
               previous_stops = pd.DataFrame(previous_stops)
               previous_stops.columns = previous_days.columns
               passengers = -1
            else:
               previous_days = group_sorted[(group_sorted['Stop_id'] == int(stop_encoding)) & (group_sorted['Stop_order'] == stop_order) & (group_sorted['Day_of_week'] == day_of_week)]
               if previous_days.empty:
                   previous_days = group_sorted[(group_sorted['Stop_id'] == int(stop_encoding)) & (group_sorted['Stop_order'] == stop_order)]
                   if previous_days.empty:
                       print('Did not find information from the historical data on the same day_of_week for the first stop order.')
                       flag = True
                       break
               passengers = previous_days['T_pa_in_veh'].median()

            minute_of_day = round(previous_days['Minute_of_day'].mean())
            hour = round(minute_of_day/60)
            current_timestamp = current_date + pd.DateOffset(hours = hour)
            current_dimos = dimos_dict[float(stop_encoding)]
            result = db.weather_today.find({'municipality':current_dimos,'timestamp':str(current_timestamp)})
            for doc in result:
                temperature, precipitation = doc['temperature'], doc['precipitation']
            current_row = [line_encoding,direction,stop_encoding,stop_order,minute_of_day,day_of_month, day_of_week,week_of_year,day_of_year,is_holiday(current_date),temperature, precipitation,passengers]

            if stop_order not in [1,2,3]:
                # Combine the previous stops and previous days' stops
                inputs = pd.concat([previous_stops, previous_days], axis=0).reset_index(drop=True) # (5,13)
                current_row_df = pd.DataFrame([current_row], columns=inputs.columns) # (1,13)
                concat_input= pd.concat([inputs, current_row_df], axis=0).reset_index(drop=True) # (6,13)
                print("input")
                print(concat_input)
                if concat_input.shape != (6,13):
                    print(" WRONG ")
                    x = input()

                # Specify the columns to convert and their respective dtypes
                column_dtypes = {
                        'Line_descr': 'int32',
                        'Direction': 'int32',
                        'Stop_id': 'int32',
                        'Stop_order': 'int32',
                        'Minute_of_day': 'int32',
                        'Day_of_month': 'int32',
                        'Day_of_week': 'int32',
                        'Week_of_year': 'int32',
                        'Day_of_year': 'int32',
                        'Is_holiday': 'int32',
                        'Temperature': 'float32',
                        'Precipitation': 'float32',
                        'T_pa_in_veh': 'int32'
                        }

                # Iterate over the columns and their specified dtypes
                for column, dtype in column_dtypes.items():
                    concat_input[column] = pd.to_numeric(concat_input[column], errors='coerce').astype(dtype)

                X_test = np.asarray(concat_input).reshape(1,6,13)

                current_prediction = model.predict(X_test)
                print('current_prediction', round(current_prediction[0][0]))
                current_row[-1] = round(current_prediction[0][0])

            stops_predicted.append(current_row)

        if flag:
            continue

        stops_predicted_df = pd.DataFrame(stops_predicted)
        predictions = pd.concat([predictions, stops_predicted_df], axis = 0).reset_index(drop=True)
        passengers = stops_predicted_df.iloc[:,-1]
        plt.plot(passengers, label='Predictions')
        plt.xlabel('')
        plt.ylabel('Ridership')
        plt.legend()
        plt.show()

    column_names = line_ake_df.columns[0:13]
    predictions.columns = column_names
    # print(predictions)

    collection = db['predictions']

    # Convert the DataFrame to a list of dictionaries
    data = predictions.to_dict("records")

    # Insert the data into your MongoDB collection
    collection.insert_many(data)
