from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pymongo
import folium
import numpy as np

app = Flask(__name__)

ld = pd.read_csv('/home/alex/Documents/Archimedes-OASA1-Project/landmarks.csv')

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['OASA1']
collection = db['stops_by_line']
collection1 = db['stops']
choices = collection.distinct("Line_descr")

selected_option = None
dir = None


@app.route('/', methods=['GET', 'POST'])
def load_page():
    #if request.method == 'POST':
    #   return redirect(url_for('rinfo'))
    
    return redirect(url_for('form'))

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
    print(selected_stop)
    res = collection.find({"Line_descr":selected_option, "Stop_descr":str(selected_stop), "Direction": int(dir)}, {"Stop_lat":1,"Stop_lon":1,"_id":0})
    res = list(res)
    print(res)
    res = pd.DataFrame(res)
    latitude = float(res.iloc[0,0])  # Replace with the actual latitude
    longitude = float(res.iloc[0,1])  # Replace with the actual longitude
    zoom_level = 16  # Adjust the zoom level as needed
    map = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)
    folium.Marker(location=[latitude, longitude], popup=selected_stop,icon=folium.Icon(color = "red")).add_to(map)
    #tooltip = "Click me!"
    for i in range(len(ld)):
        marker = folium.Marker(
            location=[ld['latitude'][i], ld['longitude'][i]],
            popup=f"<i class = 'fa-solid fa-landmark'>{ld['name'][i]}</i>",
            icon=folium.Icon(color = "lightgray", icon="glyphicon glyphicon-home"),
            #tooltip=tooltip
        )
        marker.add_to(map)

    return render_template('stop.html', selected_option = selected_option,selected_stop=selected_stop, map=map._repr_html_())


if __name__ == '__main__':
    app.run()
