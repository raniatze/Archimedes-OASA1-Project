import csv
import argparse
from pymongo import MongoClient

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--collection', type=str, required=True, help='MongoDB collection to insert into')
args = parser.parse_args()


client = MongoClient('mongodb://localhost:27017/')
db = client['OASA1']
collection = db[args.collection]

for doc in collection.find():
    print(doc)
