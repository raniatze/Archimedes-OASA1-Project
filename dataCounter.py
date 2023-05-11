import os
import pymongo

# Set up the connection to the MongoDB server
client = pymongo.MongoClient('mongodb://localhost:27017/')

# Choose the database and collection to work with
db = client['OASA1']

def find_csv_files(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def read_csv_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        content = file.readlines()
    return content

def process_files(directory):
    csv_files = find_csv_files(directory)
    counts = {}
    vehicles = set()
    directions = set()
    stop_ids = set()
    routes = set()
    for file in csv_files:
        content = read_csv_file(file)
        for line in content[1:]:
            # Get the string
            string = line.split(';')[0]
            # Get veh_no
            vehicle_no = line.split(';')[5]
            # Get direction
            direction = line.split(';')[2]
            # Get stop id
            stop_id = line.split(';')[7]
            # Increment the count for this string
            counts[string] = counts.get(string, 0) + 1
            # Add string to routes
            routes.add(string)
            # Add vehicle_no to vehicles
            vehicles.add(vehicle_no)
            # Add direction to directions
            directions.add(direction)
            # Add stop_id to stop_ids
            stop_ids.add(stop_id)

    # Print the counts for each string
    #for string, count in counts.items():
        #print(f'{string}: {count}')
        
    # Print the counts for each string
    #for vehicle_no in vehicles:
      #print(vehicle_no)
      #result = db.capacities.find({"veh_no": vehicle_no},{"vehtyp_capacity": 1})
      #for doc in result:
        #if doc["vehtyp_capacity"] == '':
           #print(vehicle_no)
           
    #for stop_id in stop_ids:
       #result = db.staseis_dimoi.find({"stop_id": stop_id}, {"dimos": 1})
       #for doc in result:
         #if doc["dimos"] == '':
            #print(stop_id)
           
    #print(directions)
    #print(routes)
    print("#routes: ", len(routes))
    print("#vehicles: ", len(vehicles))
    print("#stop_ids: ", len(stop_ids))

if __name__ == '__main__':
    directory = '/home/raniatze/AKE/'#input('Enter the directory path to search for .txt files: ')
    process_files(directory)

