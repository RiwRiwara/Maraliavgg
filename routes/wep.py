from . import defaultAPI
from flask import  jsonify, render_template, request, session, redirect, url_for, send_file, send_from_directory
from config.db import db

@defaultAPI.route('/testapi', methods=['GET'])
def api_entry():
    collection_names = db.list_collection_names()
    response = {
        'data': "API Running",
        'collection_names': collection_names
    }
    return jsonify(response)


@defaultAPI.route('/')
def index():
    return render_template('index.html' , title='Home')

@defaultAPI.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', title='Home', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', title='Home', error='No selected file')

    if file:
        file.save('./static/img/' + file.filename)
        print("---------------------")
        predictions = "sd"
        return render_template('result.html', title='Result', img=file, predictions=predictions)
    else:
        return render_template('index.html', title='Home', error='Error uploading file')



@defaultAPI.route('/img/<filename>')
def send_img(filename):
    return send_from_directory('./static/img', filename)
@defaultAPI.route('/robots.txt')
def send_robots():
    return send_from_directory('./static', 'robots.txt')

@defaultAPI.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('index'))
