import csv
import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client['OASA1']
collection = db['ake']

with open("Stop_id_encodings.txt", "r") as file:
      lines = file.readlines()

stops_data_dict = {}
for line in lines:
    line = line.strip()
    if line:
       parts = line.split(":")
       stop_id = parts[0].strip()
       stop_encoding = int(parts[1].strip())
       stops_data_dict[stop_encoding] = stop_id
       
with open("Line_descr_encodings.txt", "r") as file:
      lines = file.readlines()

lines_data_dict = {}
for line in lines:
    line = line.strip()
    if line:
       parts = line.split(":")
       line_descr = parts[0].strip()
       line_encoding = int(parts[1].strip())
       lines_data_dict[line_encoding] = line_descr        
          

pipeline = [
    {
        '$group': {
            '_id': {
                'Line_descr': '$Line_descr',
                'Direction': '$Direction',
                'Stop_id': '$Stop_id',
                'Stop_order': '$Stop_order'
            }
        }
    },
    {
        '$project': {
            '_id': 0,
            'Line_descr': '$_id.Line_descr',
            'Direction': '$_id.Direction',
            'Stop_id': '$_id.Stop_id',
            'Stop_order': '$_id.Stop_order'
        }
    }
]

# Create an empty DataFrame
df = pd.DataFrame(columns=['Line_descr', 'Line_encoding', 'Direction', 'Stop_id', 'Stop_encoding', 'Stop_order', 'Stop_descr'])

# Perform the aggregation query
result = db.ake.aggregate(pipeline)

# Iterate over the results and extract the values
for doc in result:
    line_encoding = int(doc['Line_descr'])
    direction = doc['Direction']
    stop_encoding = int(doc['Stop_id'])
    stop_order = doc['Stop_order']
    stop_id = stops_data_dict[stop_encoding]
    line_descr = lines_data_dict[line_encoding]
    
    query = db.stops.find({"stop_id": stop_id}, {"stop_descr": 1})
    for doc in query:
      stop_descr = doc["stop_descr"]
      
    # Create a new row to append to the DataFrame
    new_row = pd.DataFrame({'Line_descr': [line_descr], 'Line_encoding': [line_encoding], 'Direction': [direction], 'Stop_id': [stop_id], 'Stop_encoding': [stop_encoding], 'Stop_order': [stop_order], 'Stop_descr': [stop_descr]})
    
    # Append the new row to the DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
                    
# Convert the Stop_order column to integer
df['Line_encoding'] = df['Line_encoding'].astype(int)
df['Stop_order'] = df['Stop_order'].astype(int)

# Sort the results by Line_descr and Stop_order
df = df.sort_values(['Line_encoding', 'Stop_order'])

# Write the results to a CSV file
df.to_csv('stops_by_line.csv', index=False)
