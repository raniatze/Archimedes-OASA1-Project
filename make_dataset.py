import csv
import pymongo
import holidays
import requests
from datetime import datetime, timedelta

# Set up the connection to the MongoDB server
client = pymongo.MongoClient('mongodb://localhost:27017/')

# Choose the database to work with
db = client['OASA1']

def historical_weather_data(timestamp, latitude, longitude):

  # Open Meteo API endpoint and parameters
  url = "https://archive-api.open-meteo.com/v1/archive?"

  # Construct API request parameters
  params = {
      "latitude": latitude,
      "longitude": longitude,
      "start_date": timestamp.date(),
      "end_date": timestamp.date(),
      "timezone": "Europe/Athens",
      "hourly": ["temperature_2m","precipitation"]
  }

  # Make API request
  response = requests.get(url, params=params)

  # Check if request was successful
  if response.status_code == 200:
      # Extract weather information from response JSON
      data = response.json()
      temperature_data = data["hourly"]["temperature_2m"]
      precipitation_data = data["hourly"]["precipitation"]
      return temperature_data[timestamp.hour], precipitation_data[timestamp.hour]
  else:
      print(f"Error retrieving weather information: {response.status_code} {response.reason}")
      return
      
def main():
    with open('ake.csv') as input_file, open('ake_updated.csv', 'w') as output_file:
    
      csv_reader = csv.reader(input_file, delimiter=';')
      csv_writer = csv.writer(output_file, delimiter=';')
      
      # Write header row in output_file
      csv_writer.writerow(['Route_id', 'Rtype', 'Stop_id', 'Stop_order', 'Minute_of_day', 'Day_of_month', 'Day_of_week', 'Week_of_year', 'Day_of_year', 'Is_holiday', 'Temperature', 'Precipitation', 'T_pa_in_veh'])
      
      # Skip header row in input_file
      next(csv_reader)
      
      for row in csv_reader:
       line_descr, rtype, stop_order, arrival_datetime, stop_id, t_pa_in_veh = row[0], row[1], row[4], row[6], row[7], row[12]
       
       # Find route_id
       split_string = line_descr.split(" - ")
       route_short_name, route_long_name = split_string[0], " - ".join(split_string[1:])
       result = db.routes.find_one({"route_short_name": route_short_name, "route_long_name": route_long_name})
       if result:
       	route_id = result["route_id"]
       else:
       	print("No matching document found.")

       # Find municipality geolocation for the stop_id
       result = db.staseis_dimoi.find({"stop_id": stop_id}, {"dimos": 1})
       for doc in result:
         dimos = doc["dimos"]
       result = db.dimoi.find({'municipality_name': dimos},{'municipality_latitude': 1, 'municipality_longitude': 1})
       for doc in result:
         municipality_lat, municipality_lon = doc["municipality_latitude"], doc["municipality_longitude"]
       
       if arrival_datetime != '':
         # Convert to datetime object
         dt_object = datetime.strptime(arrival_datetime, "%Y-%m-%d %H:%M:%S")
         
         # Get time parameters
         week_of_year, day_of_week = dt_object.isocalendar()[1:]
         day_of_month, day_of_year, year = dt_object.timetuple().tm_mday, dt_object.timetuple().tm_yday, dt_object.timetuple().tm_year
         minute_of_day = (dt_object.time().hour * 60) + dt_object.time().minute
         
         # Check if day was public holiday
         holiday = list(holidays.GR(years = year).keys())
         if dt_object.date() in holiday:
         	is_holiday = 1
         else:
         	is_holiday = 0
         
         # Get historical weather data
	 # Round up or down to the closest hour	
         rounded_time = (dt_object + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
         temperature, precipitation = historical_weather_data(rounded_time, municipality_lat, municipality_lon)
         
         csv_writer.writerow([route_id, rtype, stop_id, stop_order, minute_of_day, day_of_month, day_of_week, week_of_year, day_of_year, is_holiday, temperature, precipitation, t_pa_in_veh])
       else:
         continue
       	#result = db.stop_times.find_one({"trip_id": trip_id, "stop_id": stop_id})
       	#if result:
       	#	arrival_datetime = result["arrival_time"]
       	#else:
       	#	print("No matching document found. ")

if __name__ == '__main__':
    main()
