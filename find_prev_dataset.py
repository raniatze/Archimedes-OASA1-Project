
import pymongo
import pandas as pd
import numpy as np

# Set up the connection to the MongoDB server
client = pymongo.MongoClient('mongodb://localhost:27017/')

# Choose the database and collection to work with
db = client['OASA1']

def query1(line_descr):

    pipeline = [
        {
            "$match": {
                "Line_descr": line_descr,
            }
        },
    {
        "$group": {
            "_id": {
                "Line_descr": "$Line_descr",
                "Direction": "$Direction",
                "Sched": "$Sched",
                "Day_of_year": "$Day_of_year"
            },
            "docs": {"$push": "$$ROOT"}
        }
    },
    {
        "$project": {
            "_id": 0,
            "docs._id": 0
        }
    }
    ]

    return pipeline

def query2(line_descr, direction, sched, day_of_year,stops):
    pipeline = [
        {
            "$addFields": {
                "Day_of_year_int": {"$toInt": "$Day_of_year"}
            }
        },
        {
            "$match": {
                "Line_descr": line_descr,
                "Direction": direction,
                "Sched": sched,
                "Day_of_year_int": {"$lt": int(day_of_year)}
            }
        },
        {
            "$sort": {
                "Day_of_year_int": -1
            }
        },
        {
            "$limit": 2 * stops
        },
        {
            "$group": {
                "_id": {
                    "Stop_order": "$Stop_order"
                },
                "docs": {"$push": "$$ROOT"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "docs._id": 0,
                "Day_of_year_int": 0,
                "Stop_order_int": 0
            }
        }
    ]


    return pipeline


def query_db(pipeline):
    try:
        results = db.ake.aggregate(pipeline)
        result_docs = list(results)
        print(f"Number of documents: {len(result_docs)}")
        return result_docs
    except Exception as e:
        print(f"An error occurred during aggregation: {e}")
        return []

def find_prev_stops(df_sorted, i):

    if i < 3:
        data = (3-i)*[[np.nan] * 15]
        zeros_df = pd.DataFrame(data)
        stops_prev = pd.DataFrame(zeros_df, columns = df_sorted.columns)
        stops_prev = pd.concat([stops_prev, df_sorted.iloc[0:i, :]], axis = 0, ignore_index= True)
    else:
        stops_prev = df_sorted.iloc[i-3:i, :]

    return stops_prev

def to_df_previous_days(res_docs):
    prev_days = []

    for j in range(len(res_docs)):
        pday = pd.DataFrame(res_docs[j]['docs'])
        prev_days.append(pday)

    prev_days_total = pd.concat(prev_days)
    prev_days_total.reset_index(drop=True, inplace=True)

    return prev_days_total


pipeline1 = query1('129')
res_docs1 = query_db(pipeline1)


# Convert dictionaries to string representations
doc_strings = [str(doc) for doc in res_docs1]

if len(doc_strings) != len(set(doc_strings)):
    print("Duplicates found in res_docs1.")
else:
    print("No duplicates found in res_docs1.")


dataset = []
sample = res_docs1[0:1]
counter = 0


for doc in sample:

    dataframes = []
    df_prev_stops = pd.DataFrame(doc['docs'],dtype=str)

    line = df_prev_stops.loc[0,'Line_descr']
    direction = df_prev_stops.loc[0,'Direction']
    year = df_prev_stops.loc[0,'Year']
    day = df_prev_stops.loc[0,'Day_of_year']
    sched = df_prev_stops.loc[0,'Sched']

    print(line,direction,sched,day,year)
    x = int(df_prev_stops['Stop_order'].count())

    pipeline2 = query2(line, direction, sched, day, x)
    res_docs2 = query_db(pipeline2)

    prev_days = to_df_previous_days(res_docs2)

    df_prev_stops['Stop_order'] = df_prev_stops['Stop_order'].astype('int')
    df_prev_stops.sort_values('Stop_order', ascending = True, inplace = True)
    df_prev_stops.reset_index(drop=True, inplace=True)

    for i,row in df_prev_stops.iterrows():

        current_stop = row['Stop_order']
        prev_stops_i = find_prev_stops(df_prev_stops, current_stop-1)
        prev_days_i = prev_days.loc[prev_days['Stop_order'] == current_stop]

        if len(prev_days_i) < 2 :
            data = (2-i)*[[np.nan] * 15]
            zeros_df = pd.DataFrame(data, columns=df_prev_stops.columns)
            prev_days_i = pd.DataFrame(zeros_df)

        dataframes.append(pd.DataFrame(row).T)
        dataframes.append(prev_stops_i)
        dataframes.append(prev_days_i)

    dataset_temp = pd.concat(dataframes, axis=0)
    dataset.append(dataset_temp)


dataset_final = pd.concat(dataset, axis = 0, ignore_index = True)

# Reconvert not NaN 'Stop_order' to int
dataset_final.loc[dataset_final['Stop_order'].notna(),'Stop_order'] = dataset_final.loc[dataset_final['Stop_order'].notna(),'Stop_order'].astype('int')

# Conver all values to string
dataset_final = dataset_final.astype('str')


with open('output.txt', 'w') as file:
    dataset_final.to_csv(file, header=0, index=None, sep=' ', mode='a')
