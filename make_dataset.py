import csv
import pymongo
import holidays
import requests
from datetime import datetime, timedelta

# Set up the connection to the MongoDB server
client = pymongo.MongoClient('mongodb://localhost:27017/')

# Choose the database and collection to work with
db = client['OASA1']
collection = db["ake"]

      
def main():
    cnt = 0
    cursor = collection.find()
    for row in cursor:
      
      line_descr, rtype, sched, stop_order, arrival_datetime, stop_id, t_pa_in_veh = row['Line_descr'], row['Rtype'], row['Sched'], row['S_order'], row['Arrival_datetime'], row['Stop_id'], row['T_pa_in_veh']
     
      # Find route_id
      split_string = line_descr.split(" - ")
      route_short_name, route_long_name = split_string[0], " - ".join(split_string[1:])
      result = db.routes.find({"route_short_name": route_short_name, "route_long_name": route_long_name}, {"route_id": 1})
      for doc in result:
        route_id = doc["route_id"]

      # Find municipality of the stop_id
      result = db.staseis_dimoi.find({"stop_id": stop_id}, {"dimos": 1})
      for doc in result:
        dimos = doc["dimos"]
       
      
      # Convert to arrival_datetime to datetime object
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
      result = db.weather.find({"municipality": dimos, "timestamp": str(rounded_time)}, {"temperature": 1, "precipitation": 1})
      for doc in result:
        temperature, precipitation = doc["temperature"], doc["precipitation"]
 
      #csv_writer.writerow([route_id, rtype, stop_id, stop_order, minute_of_day, day_of_month, day_of_week, week_of_year, day_of_year, is_holiday, temperature, precipitation, t_pa_in_veh])

if __name__ == '__main__':
    main()
