from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pymongo
import folium
import datetime
import numpy as np

app = Flask(__name__)

def get_output_string(strings, direction):
    if direction == '1':
        output = '-'.join(strings)
    elif direction == '2':
        output = '-'.join(strings[::-1])
    else:
        output = "Invalid direction value. Please choose 1 or 2."

    return output

def get_direction(df,name):
    direction = np.where(df['Name'] == name)
    direction = direction[0][0]
    return df.iloc[direction,0]


ld = pd.read_csv('../landmarks.csv')
tax = pd.read_csv('../staseis_taxi.csv',delimiter=";")
met = pd.read_csv('../staseis_metro.csv',delimiter=",")

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['OASA1']
collection = db['stops_by_line']
collection1 = db['stops']
predictions = db['predictions']
accessibility = db['staseis_proexoxes']
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
    print(res2)
    res2 = np.unique(res2)
    print(len(res2))
    nam = selected_option
    nam = nam.split("-")
    nam = nam[1:]
    global df
    df = pd.DataFrame(columns=['Direction','Name'])
    for i in range(len(res2)):
        df.loc[i] = [res2[i],get_output_string(nam,res2[i])]
    print(df)
    #print(nam)
    #if len(res2) < 2:
    #    redirect(url_for('rinfo'))
    return render_template('direction.html', selected_option = selected_option, choices = df['Name'])

@app.route('/rinfo', methods=['POST'])
def rinfo():
    global dir
    dir1 = request.form.get('dropdown1')
    dir = get_direction(df,dir1)
    print(dir)
    #print(selected_option)
    results = collection.find({"Line_descr": str(selected_option), "Direction": str(dir)}, {"Stop_descr":1, "_id": 0})
    data = list(results)
    stopss = pd.DataFrame(data)
    print(stopss)
    #print(stopss)
    stops = stopss['Stop_descr'].tolist()
    return render_template('routeinfo.html', selected_option=selected_option, stops=stops, dir = dir1)

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
    res = collection.find({"Line_descr":selected_option, "Stop_descr":str(selected_stop), "Direction": str(dir)}, {"Stop_lat":1,"Stop_lon":1,"_id":0})
    res = list(res)
    print(res)
    res = pd.DataFrame(res)
    latitude = float(res.iloc[0,0])  # Replace with the actual latitude
    longitude = float(res.iloc[0,1])  # Replace with the actual longitude
    zoom_level = 16  # Adjust the zoom level as needed
    map = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)
    folium.Marker(location=[latitude, longitude], popup=selected_stop,icon=folium.Icon(color = "red", icon="bus", prefix="fa")).add_to(map)
    #tooltip = "Click me!"
    access = accessibility.find_one({"stop_descr":selected_stop},{"stop_descr":1,"_id":0})
    if access:
        access = True
        print(access)
    else : 
        access = False
        print(access)
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
    Day_of_year = '249'
    year = '2021'
    stop_encoding = collection.find({"Line_descr":selected_option, "Stop_descr":str(selected_stop), "Direction": str(dir)},{"Stop_encoding":1,"_id":0})
    stop_encoding = list(stop_encoding)
    stop_encoding = pd.DataFrame(stop_encoding)
    print(stop_encoding.iloc[0,0])

     # Get current datetime
    current_datetime = datetime.datetime.now()

    # Extract year, day of the year, and minute of the day
    year = current_datetime.year
    Day_of_year = current_datetime.timetuple().tm_yday
    minute_of_day = current_datetime.hour * 60 + current_datetime.minute
    print(Day_of_year)
    print(minute_of_day)
    print(selected_option)
    #Day_of_year = '76'
    #year = '2023'
    #minute_of_day = 324
    encoding = collection.find({"Line_descr":selected_option, "Stop_descr":str(selected_stop), "Direction": str(dir)},{"Stop_encoding":1,"Line_encoding":1,"_id":0})
    encoding = list(encoding)
    encoding = pd.DataFrame(encoding)
    print(encoding)

    res3 = predictions.find({"Line_descr":str(encoding.iloc[0,0]), "Stop_id":int(encoding.iloc[0,1]), "Direction": str(dir),"Day_of_year":Day_of_year}, {"Minute_of_day":1,"T_pa_in_veh":1,"_id":0})
    res3 = list(res3)
    print(res3)
    final_result = 0
    for d in res3:
        if int(d['Minute_of_day']) >= minute_of_day:
            final_result = round(d['T_pa_in_veh'])
            break

    #res3 = predictions.find({"Line_descr":selected_option, "Stop_id":str(stop_encoding.iloc[0,0]), "Direction": str(dir),"Day_of_year":Day_of_year,"Year":year}, {"Minute_of_day":1,"T_pa_in_veh":1,"_id":0})
    #res3 = list(res3)
    #print(res3)
    return render_template('stop.html', selected_option = selected_option,selected_stop=selected_stop, final_result= final_result,access = access ,map=map._repr_html_())


if __name__ == '__main__':
    app.run()
