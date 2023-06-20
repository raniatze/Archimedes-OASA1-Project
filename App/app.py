from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pymongo

app = Flask(__name__)

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['OASA1']
collection1 = db['ake']
c2 = db['stops']
   

merge_df2 = pd.read_csv('/home/alex/Documents/routess.csv')
choices = merge_df2['route_short_name'].tolist()

merge_df = pd.read_csv('/home/alex/Documents/stop.csv')


@app.route('/', methods=['GET', 'POST'])
def load_page():
    if request.method == 'POST':
        return redirect(url_for('rinfo'))
    
    return redirect(url_for('form'))

@app.route('/form', methods=['GET'])
def form():
    return render_template('form.html', choices=choices)

@app.route('/rinfo', methods=['POST'])
def rinfo():
    selected_option = request.form.get('dropdown')
    filtered_df1 = merge_df2[merge_df2['route_short_name'] == selected_option]
    x = filtered_df1['Line_descr'].values[0]
    #print(x)
    dit = collection1.distinct("Stop_id",{"Line_descr":str(x)})
    #print(dit)   
    stops= []
    for i in range(len(dit)):
        filtered_df = merge_df[merge_df['Stop_encoding'] == int(dit[i])]
        if not filtered_df.empty:
            stops.append(filtered_df['stop_name'].values[0])
        else:
            stops.append(None) 
    stops = list(filter(lambda value: value is not None, stops))
    #print(stops)
    
    # Handle form submission logic
    return render_template('routeinfo.html', selected_option=selected_option, stops=stops)


if __name__ == '__main__':
    app.run()
