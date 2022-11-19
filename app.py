from flask import Flask, request, abort, render_template, send_from_directory, jsonify
from lib.helper import *
from lib.tsdn_main import TSDN

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


@app.route("/data")
def data_url():
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
