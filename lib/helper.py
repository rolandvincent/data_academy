import json
import numpy as np


def convert_request(request):
    if request.headers.get('Content-Type') == 'application/json':
        return dict(request.get_json())
    else:
        return dict(request.form)


def is_valid_content(request, content_type=['application/json', 'multipart/form-data', 'application/x-www-form-urlencoded']) -> bool:
    if any(request.headers.get('Content-Type', 'NULL').startswith(x) for x in content_type):
        return True
    return False


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def dataFrameToArray(dataFrame: dict):
    totalRow = len(dataFrame.get(dataFrame.keys()[0]).keys())
    tmp = list()
    for i in range(1, totalRow):
        row = list()
        for key in dataFrame.keys():
            row.append(dataFrame.get(key).get(i))
        tmp.append(row)

    return tmp
