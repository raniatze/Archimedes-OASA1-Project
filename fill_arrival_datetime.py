# Input:  .csv file from ONE day

import datetime
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('ake_data.csv', sep=';')

# Group the data by Line_descr, Rtype, Direction, Sched, and Vehicle_no
grouped = df.groupby(['Line_descr', 'Rtype', 'Direction', 'Sched', 'Vehicle_no'])

# Iterate over each group
for name, group in grouped:
    
    prev_stop_time = None
    prev_stop_order = None

    # Sort the entries of the group by S_order column in ascending order
    group_sorted = group.sort_values('S_order', ascending=True)
    group_sorted = group_sorted.reset_index()
    
    # Iterate over each row of the sorted group
    for i, row in group_sorted.iterrows():
        
        # Stop has no arrival_datetime
        if pd.isna(row['Arrival_datetime']):
           
           # Previous stop has arrival datetime
           if prev_stop_time is not None:
              stop_order = int(row['S_order'])
            
              # There is next stop
              if i != len(group_sorted) - 1:
                next_stop = group_sorted.iloc[i+1]
                
                # Next stop has arrival datetime
                if not pd.isna(next_stop['Arrival_datetime']):
                   next_stop_time = datetime.datetime.strptime(next_stop['Arrival_datetime'], '%Y-%m-%d %H:%M:%S')
                   next_stop_order = int(next_stop['S_order'])
                   time_diff = (next_stop_time - prev_stop_time) / (next_stop_order - prev_stop_order)
                   stop_time = prev_stop_time + (time_diff * (stop_order - prev_stop_order))
                   #db.ake_test.update_one({'_id': stop['_id']}, {'$set': {'Arrival_datetime': stop_time.strftime('%Y-%m-%d %H:%M:%S')}})
               
                # Next stop has no arrival datetime
                else: 
                   #db.ake_test.update_one({'_id': stop['_id']}, {'$set': {'Arrival_datetime': prev_stop_time.strftime('%Y-%m-%d %H:%M:%S')}})
                   prev_stop_order = stop_order
            
              # There is no next stop
              else:
                #db.ake_test.update_one({'_id': stop['_id']}, {'$set': {'Arrival_datetime': prev_stop_time.strftime('%Y-%m-%d %H:%M:%S')}})
                prev_stop_order = stop_order
            
           # Previous stop has no arrival datetime
           else:
             continue
             #db.ake_test.update_one({'_id': stop['_id']}, {'$set': {'Arrival_datetime': ''}})
        
        # Stop has arrival datetime
        else:
          prev_stop_time = datetime.datetime.strptime(row['Arrival_datetime'], '%Y-%m-%d %H:%M:%S')
          prev_stop_order = int(row['S_order'])
