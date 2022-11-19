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
            dataset = pd.read_csv('files/Mall_Customers.csv', index_col=0)
            # dataset['Gender'] = dataset['Gender'].apply(
            #     lambda x:  0 if x == 'Male' else 1)
            return json.dumps(dataFrameToArray(dataset), cls=NpEncoder)
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
