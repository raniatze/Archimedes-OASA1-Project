import pandas as pd
import numpy as np
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")  # Replace with your MongoDB connection details
db = client["OASA1"]  # Replace with your database name

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
    
    print("Previous stop orders")       
    print(result_df)
    return result_df

def create_input_sequences(line_descr_df, m, n):
    input_sequences = [] # X_train
    target_values = [] # Y_train

    # Group the data by Day_of_year
    grouped_data = line_descr_df.groupby(['Direction', 'Sched'])

    for name, group in grouped_data:
        print("################################################### NEW GROUP #####################################")
        print(name)
        group_sorted = group.sort_values(['Year', 'Day_of_year', 'Stop_order'], ascending=[True, True, True])
        group_sorted = group_sorted.reset_index(drop=True)
        # Get the sequence length for the current Day_of_year group
        group_length = len(group_sorted)
        if group_length >= (m + n):
            # Iterate over the group, starting from sequence_length index
            for i in range(m, group_length):
                # Get the previous stops for the current instance
                previous_stops = get_previous_stops(m, i, group_sorted)
                
                if len(previous_stops) != m:
                   continue

                # Get the previous days' stops for the current instance
                #previous_days = get_previous_days(data, group['Day_of_year'].iloc[i], sequence_length)

                # Combine the previous stops and previous days' stops
                #inputs = previous_stops + previous_days

                # Get the target value (T_pa_in_veh) for the current instance
                #target = group['T_pa_in_veh'].iloc[i]

                #input_sequences.append(inputs)
                #target_values.append(target)
        else:
            continue

    # Convert the input_sequences and target_values to numpy arrays
    input_sequences = np.array(input_sequences)
    target_values = np.array(target_values)

    return input_sequences, target_values

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

def get_previous_days(n, stop_id, stop_order, day_of_year, df):
    # Filter a specific stop_id and stop_order
    filtered_df = df[(df['Stop_id'] == stop_id) & (df['Stop_order'] == int(stop_order))]

    # Sort filtered dataframe
    sorted_df = filtered_df.sort_values(by='Day_of_year', ascending=True).reset_index(drop=True)

    # Find the index of the row with the specific day of year
    index = sorted_df[sorted_df['Day_of_year'] == int(day_of_year)].index[0]

    # Use the index to slice the dataframe and retrieve the previous m rows
    previous_days_df = sorted_df.loc[index-n:index-1]
    result_length = len(previous_days_df)

    # Fill missing values with NaN
    if result_length < n:
       missing_rows = n - result_length
       missing_data = pd.DataFrame(np.nan, index=np.arange(missing_rows), columns=previous_days_df.columns)
       previous_days_df = pd.concat([previous_days_df, missing_data]).reset_index(drop=True)

    return previous_days_df


# Example usage
m = 3 # previous stop_orders
n = 2 # previous days
line_descr = "1"
direction = "1"
stop_id = "38"
stop_order = "18"
sched = "1900-01-01 05:50:00"
day_of_year = "96"
year = "2021"


filtered_data = filter_line_descr(line_descr)
line_descr_df = pd.DataFrame(filtered_data)
line_descr_df.drop(['_id'], axis=1, inplace=True)
line_descr_df['Stop_order'] = line_descr_df['Stop_order'].astype(int)
line_descr_df['Day_of_year'] = line_descr_df['Day_of_year'].astype(int)
line_descr_df['Minute_of_day'] = line_descr_df['Minute_of_day'].astype(int)
line_descr_df['Year'] = line_descr_df['Year'].astype(int)

create_input_sequences(line_descr_df, m, n)
