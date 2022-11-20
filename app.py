from flask import Flask, request, abort, render_template, send_from_directory, jsonify
from lib.helper import *
from lib.tsdn_main import TSDN
import pandas as pd
import json

app = Flask(__name__)
tsdn = TSDN()


@app.route('/assets/<path:path>')
def assets_path(path):
    return send_from_directory('assets', path)


@app.route('/files/<path:path>')
def file_path(path):
    return send_from_directory('data', path)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/data", methods=['GET'])
def data_url():
    data = request.args.to_dict()
    if (len(data.items()) > 0):
        if (data.get('query', None) == 'dataset'):
            dataset = pd.read_csv('files/segm_pca.csv', index_col=0)
            dataset['Gender'] = dataset['Gender'].apply(
                lambda x:  'Male' if x == 0 else 'Female')
            dataset.drop(['Component 1', 'Component 2', 'Component 3', 'Segment K-means PCA'],
                         axis=1, inplace=True)
            return json.dumps(dataFrameToArray(dataset), cls=NpEncoder)
        if (data.get('query', None) == 'graph'):
            dataset = pd.read_csv('files/segm_pca.csv', index_col=0)

            graph_data = pd.DataFrame()
            graph_data['Component 2'] = dataset['Component 2']
            graph_data['Component 3'] = dataset['Component 3']
            graph_data['Segment'] = dataset['Segment K-means PCA']

            return json.dumps(dataFrameToArray(graph_data), cls=NpEncoder)
        return {}
    else:
        return render_template('data.html')


@app.route("/chart")
def chart_url():
    return render_template('chart.html')


@app.route("/pred", methods=['POST'])
def predict():
    if not is_valid_content(request):
        abort(400)
    data: dict = convert_request(request)
    if all(x in data for x in ['gender', 'age', 'income', 'spending_score']):
        result = tsdn.predict(int(data['gender']), int(data['age']), int(
            data['income']), int(data['spending_score']))
        return jsonify(dict(result))
    else:
        abort(400)
