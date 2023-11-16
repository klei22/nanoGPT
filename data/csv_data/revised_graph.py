import csv
import plotly.express as px
from flask import Flask, jsonify
import argparse
from datetime import datetime

def parseargs():
    parser = argparse.ArgumentParser(description='Plot data from CSV file.')
    parser.add_argument("-i", "--input_file", type=str, help="Input CSV file")
    return parser.parse_args()

args = parseargs()

app = Flask(__name__)

x = []
y_series = []

with open(args.input_file) as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        # Add the timestamp to the x-axis data list
        x.append(datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
        
        # For each column, add the data point to the respective y-series list
        for i in range(1, len(row)):
            # Expand y_series list to hold new columns as they are encountered
            if len(y_series) < i:
                y_series.append([])
            
            # Append data to the appropriate y-series list, handling empty strings
            y_series[i-1].append(float(row[i]) if row[i] != '' else None)

# Plotting each y-series as a separate line on the same graph
fig = px.line()
for i, ys in enumerate(y_series):
    fig.add_scatter(x=x, y=ys, mode='lines', name=f'Column {i+1}')

@app.route('/')
def index():
    return '''
        <html>
            <head>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <div id='plot'></div>
                <script>
                   var graph = ''' + fig.to_json() + ''';
                   Plotly.plot('plot',graph);
                </script>
            </body>
        </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, port=6006)

