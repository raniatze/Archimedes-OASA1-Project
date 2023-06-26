from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pymongo
import folium
import numpy as np
import datetime

app = Flask(__name__)

ld = pd.read_csv('../Data/landmarks.csv')
tax = pd.read_csv('../Data/staseis_taxi.csv',delimiter=";")
met = pd.read_csv('../Data//staseis_metro.csv',delimiter=",")

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['OASA1']
collection = db['stops_by_line']
collection1 = db['stops']
predictions = db['ake']
choices = collection.distinct("Line_descr")

selected_option = None
dir = None


@app.route('/', methods=['GET', 'POST'])
def load_page():
    #if request.method == 'POST':
    #   return redirect(url_for('rinfo'))
    
    #return redirect(url_for('form'))
    return redirect(url_for('home'))

@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return redirect(url_for('dir'))
    return render_template('home.html', choices=choices)

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        return redirect(url_for('dir'))
    return render_template('form.html', choices=choices)

@app.route('/direction', methods = ['POST'])
def dir():
    global selected_option
    selected_option = request.form.get('dropdown')
    res2 = collection.find({"Line_descr": selected_option},{"Direction":1, "_id":0})
    res2 = list(res2)
    res2 = pd.DataFrame(res2)
    res2 = np.unique(res2)
    if len(res2) < 2:
        redirect(url_for('rinfo'))
    return render_template('direction.html', selected_option = selected_option, choices = res2)

@app.route('/rinfo', methods=['POST'])
def rinfo():
    global dir
    dir = request.form.get('dropdown1')
    print(dir)
    print(selected_option)
    results = collection.find({"Line_descr": str(selected_option), "Direction": int(dir)}, {"Stop_descr":1, "_id": 0})
    data = list(results)
    stopss = pd.DataFrame(data)
    print(stopss)
    stops = stopss['Stop_descr'].tolist()
    return render_template('routeinfo.html', selected_option=selected_option, stops=stops, dir = dir)

#@app.route('/rinfo', methods=['POST'])
#def rinfoo():
#    selected_option = request.form.get('dropdown')
#     return redirect(url_for('stopinfo', selected_option=selected_option, _external=True))


@app.route('/stopinfo', methods = ['GET','POST'])
def stopinfo():
    #selected_option = request.args.get('selected_option')
    #print(selected_option)
    selected_stop = request.form.get('stop-dropdown')
    #print(selected_stop)
    res = collection.find({"Line_descr":selected_option, "Stop_descr":str(selected_stop), "Direction": int(dir)}, {"Stop_lat":1,"Stop_lon":1,"_id":0})
    res = list(res)
    print(res)
    res = pd.DataFrame(res)
    latitude = float(res.iloc[0,0])  # Replace with the actual latitude
    longitude = float(res.iloc[0,1])  # Replace with the actual longitude
    zoom_level = 16  # Adjust the zoom level as needed
    map = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)
    folium.Marker(location=[latitude, longitude], popup=selected_stop,icon=folium.Icon(color = "red", icon="bus", prefix="fa")).add_to(map)
    #tooltip = "Click me!"
    for i in range(len(ld)):
        marker = folium.Marker(
            location=[ld['latitude'][i], ld['longitude'][i]],
            popup=f"<i>{ld['name'][i]}</i>",
            icon=folium.Icon(color = "purple",),
            #tooltip=tooltip
        )
        marker.add_to(map)

    for i in range(len(tax)):
        marker = folium.Marker(
            location=[tax['latitude'][i], tax['longitude'][i]],
            popup=f"<i>{tax['name'][i]}</i>",
            icon=folium.Icon(color = "darkgreen", icon="taxi",prefix="fa"),
            #tooltip=tooltip
        )
        marker.add_to(map)

    for i in range(len(met)):
        marker = folium.Marker(
            location=[met['latitude'][i], met['longitude'][i]],
            popup=f"<i>{met['s_name'][i]}</i>",
            icon=folium.Icon(color = "blue", icon="subway", prefix="fa"),
            #tooltip=tooltip
        )
        marker.add_to(map)

    # Get current datetime
    current_datetime = datetime.datetime.now()

    # Extract year, day of the year, and minute of the day
    year = current_datetime.year
    Day_of_year = current_datetime.timetuple().tm_yday
    minute_of_day = current_datetime.hour * 60 + current_datetime.minute
    Day_of_year = '76'
    year = '2023'
    minute_of_day = 324
    encoding = collection.find({"Line_descr":selected_option, "Stop_descr":str(selected_stop), "Direction": int(dir)},{"Stop_encoding":1,"Line_encoding":1,"_id":0})
    encoding = list(encoding)
    encoding = pd.DataFrame(encoding)

    res3 = predictions.find({"Line_descr":str(encoding.iloc[0,0]), "Stop_id":str(encoding.iloc[0,1]), "Direction": str(dir),"Day_of_year":Day_of_year,"Year":year}, {"Minute_of_day":1,"T_pa_in_veh":1,"_id":0})
    res3 = list(res3)
    print(res3)
    final_result = 0
    for d in res3:
        if int(d['Minute_of_day']) >= minute_of_day:
            final_result = d['T_pa_in_veh']
            break

    return render_template('stop.html', selected_option = selected_option,selected_stop=selected_stop, passengers=final_result, map=map._repr_html_())


if __name__ == '__main__':
    app.run()
