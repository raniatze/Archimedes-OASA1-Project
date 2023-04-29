import datetime
import pymongo

# Set up the connection to the MongoDB server
client = pymongo.MongoClient('mongodb://localhost:27017/')

# Choose the database and collection to work with
db = client['OASA1']

# sort the collection by S_order
docs = db.ake_test.find()
sorted_docs = sorted(docs, key=lambda doc: int(doc['S_order']))
sorted_docs = iter(sorted_docs)

# iterate over the stops and fill in the missing arrival times
prev_stop_time = None
prev_stop_order = None
for stop in sorted_docs: # kane to stop -> doc
    print(stop)
    if stop['Arrival_datetime'] == '':
        print("empty arrival datetime")
        print(stop)
        if prev_stop_time is not None:
            next_stop = next(sorted_docs, None)
            if next_stop is not None and next_stop['Arrival_datetime'] != '':
                print("previous and next stop times available")
                print("previous stop time: ", prev_stop_time)
                print("previous stop order: ", prev_stop_order)
                next_stop_time = datetime.datetime.strptime(next_stop['Arrival_datetime'], '%Y-%m-%d %H:%M:%S')
                print("next stop time: ", next_stop_time)
                next_stop_order = int(next_stop['S_order'])
                print("next stop order: ", next_stop_order)
                time_diff = (next_stop_time - prev_stop_time) / (next_stop_order - prev_stop_order)
                stop_time = prev_stop_time + (time_diff * (int(stop['S_order']) - prev_stop_order))
                #stop['Arrival_datetime'] = stop_time.strftime('%Y-%m-%d %H:%M:%S')
                db.ake_test.update_one({'_id': stop['_id']}, {'$set': {'Arrival_datetime': stop_time.strftime('%Y-%m-%d %H:%M:%S')}})
            else:
                print("no next stop or next stop arrival datetime null")
                #stop['Arrival_datetime'] = prev_stop_time.strftime('%Y-%m-%d %H:%M:%S')
                db.ake_test.update_one({'_id': stop['_id']}, {'$set': {'Arrival_datetime': prev_stop_time.strftime('%Y-%m-%d %H:%M:%S')}})
        else:
            #stop['Arrival_datetime'] = ''
            db.ake_test.update_one({'_id': stop['_id']}, {'$set': {'Arrival_datetime': ''}})
        print(stop['Arrival_datetime'])
    else:
        prev_stop_time = datetime.datetime.strptime(stop['Arrival_datetime'], '%Y-%m-%d %H:%M:%S')
        prev_stop_order = int(stop['S_order'])
