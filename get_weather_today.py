import csv
import json
import requests
import subprocess
from pymongo import MongoClient
from datetime import date, datetime

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection details

# Access the database
db = client['OASA1']  # Replace with your database name

# Drop the collection
db['weather_today'].drop()

# Close the connection
client.close()

def get_data(lat, long):
    url = "https://api.open-meteo.com/v1/forecast?"
    
    current_date = date.today()
    formatted_date = current_date.strftime('%Y-%m-%d')
   
    params = {
            "latitude": lat,
            "longitude": long,
            "start_date": formatted_date,
            "end_date": formatted_date,
            "timezone": "Europe/Athens",
            "hourly": ["temperature_2m", "precipitation"]
            }
    
    # Make API request
    response = requests.get(url, params=params)

    # Check if request was successful
    if response.status_code == 200:
        # Extract weather information from response JSON
        data = response.json()

        """
        temperature_data = data["hourly"]["temperature_2m"]
        precipitation_data = data["hourly"]["precipitation"]
        return temperature_data[timestamp.hour], precipitation_data[timestamp.hour]
        """

        return data

    else:
        print(f"Error retrieving weather information: {response.status_code} {response.reason}")
        #raise Exception("api error")
        sleep(5)
        get_data(lat, long)


csv_out = "weather_today.csv"

def main():
    path = "./Data/dimoi.csv"

    dimoi = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            dimoi.append({
                "name": row['municipality_name'],
                "lat": float(row['municipality_latitude']),
                "long": float(row['municipality_longitude'])
            })


    for_csv = []

    for dimos in dimoi:
        print("Getting data for municipality:", dimos['name'])
        weather = get_data(dimos['lat'], dimos['long'])['hourly']


        times = []
        for t in weather['time']:
            times.append(datetime.strptime(t, '%Y-%m-%dT%H:%M'))

        matched = list(zip(times, weather['temperature_2m'], weather['precipitation']))

        dicts = []
        for m in matched:
            d = {}
            d['municipality'] = dimos['name']
            d['timestamp'] = m[0]
            d['temperature'] = m[1]
            d['precipitation'] = m[2]
            dicts.append(d)

        for_csv.extend(dicts)


    with open(csv_out, 'w', newline='', encoding='utf-8') as csvfile2:
        #fieldnames = ['municipality', 'timestamp', 'temperature', 'precipitaion']
        fieldnames = for_csv[0].keys()
        writer = csv.DictWriter(csvfile2, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(for_csv)

    subprocess.run(['python3', './inserter.py', '-f', csv_out, '-c', 'weather_today', '-s', ','])

if __name__ == "__main__":
    main()
