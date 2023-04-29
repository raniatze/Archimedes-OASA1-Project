import datetime
import pymongo

# Set up the connection to the MongoDB server
client = pymongo.MongoClient('mongodb://localhost:27017/')

# Choose the database and collection to work with
db = client['OASA1']

# sort the collection by S_order
docs = db.ake_test.find()
sorted_docs = sorted(docs, key=lambda doc: int(doc['S_order']))

# iterate over the stops and fill in the missing arrival times
prev_stop_time = None
prev_stop_order = None
for stop in sorted_docs: 
 
    # Stop has no arrival_datetime
    if stop['Arrival_datetime'] == '':
    
        # There is previous stop time available
        if prev_stop_time is not None:
            stop_order = int(stop['S_order'])
            
            # There is next stop
            if stop_order != len(sorted_docs):
               next_stop = sorted_docs[stop_order]
               
               # Next stop has arrival datetime
               if next_stop['Arrival_datetime'] != '':
                  next_stop_time = datetime.datetime.strptime(next_stop['Arrival_datetime'], '%Y-%m-%d %H:%M:%S')
                  next_stop_order = int(next_stop['S_order'])
                  time_diff = (next_stop_time - prev_stop_time) / (next_stop_order - prev_stop_order)
                  stop_time = prev_stop_time + (time_diff * (stop_order - prev_stop_order))
                  db.ake_test.update_one({'_id': stop['_id']}, {'$set': {'Arrival_datetime': stop_time.strftime('%Y-%m-%d %H:%M:%S')}})
               
               # Next stop has no arrival datetime
               else: 
                  db.ake_test.update_one({'_id': stop['_id']}, {'$set': {'Arrival_datetime': prev_stop_time.strftime('%Y-%m-%d %H:%M:%S')}})
                  prev_stop_order = stop_order
            
            # There is no next stop
            else:
                db.ake_test.update_one({'_id': stop['_id']}, {'$set': {'Arrival_datetime': prev_stop_time.strftime('%Y-%m-%d %H:%M:%S')}})
                prev_stop_order = stop_order
            
        # There is no previous stop time available
        else:
            db.ake_test.update_one({'_id': stop['_id']}, {'$set': {'Arrival_datetime': ''}})
        
    # Stop has arrival datetime
    else:
        prev_stop_time = datetime.datetime.strptime(stop['Arrival_datetime'], '%Y-%m-%d %H:%M:%S')
        prev_stop_order = int(stop['S_order'])
