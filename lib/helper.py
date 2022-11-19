def convert_request(request):
    if request.headers.get('Content-Type') == 'application/json':
        return dict(request.get_json())
    else:
        return dict(request.form)


def is_valid_content(request, content_type=['application/json', 'multipart/form-data', 'application/x-www-form-urlencoded']) -> bool:
    if any(request.headers.get('Content-Type', 'NULL').startswith(x) for x in content_type):
        return True
    return False
