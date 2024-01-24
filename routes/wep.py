from . import defaultAPI
from flask import  jsonify, render_template, request, session, redirect, url_for, send_file, send_from_directory
from config.db import db
from PIL import Image
import base64
from io import BytesIO
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


Classes = ['Uninfected_Patients', 'Plasmodium_falciparum', 'Plasmodium_Vivax']

class VGGModel(torch.nn.Module):
    def __init__(self):
        super(VGGModel, self).__init__()

    def forward(self, x):
        pass

@defaultAPI.route('/testapi', methods=['GET'])
def api_entry():
    response = {
        'status': 200,
        'model_status': 'Model loaded successfully!',
        'data': "API Running",
    }
    return jsonify(response)


@defaultAPI.route('/')
def index():
    return render_template('index.html' , title='Home')
@defaultAPI.route('/about')
def about():
    return render_template('aboutus.html' , title='About Us')
@defaultAPI.route('/tutorial')
def tutorial():
    return render_template('tutorial.html' , title='Tutorial')


@defaultAPI.route('/upload', methods=['POST'])
def upload():


    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    model_path = './Maraliavgg16.pt'  
    model = torch.jit.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    print("Model loaded successfully!")

    image = Image.open(file)
    image2 = Image.open(file)

    single_image_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])
    transformed_image = single_image_transform(image)

    transformed_image = transformed_image.unsqueeze(0)

    outputs = model(transformed_image)
    _, predicted = torch.max(outputs, 1)

    image = transformed_image.squeeze().cpu().permute(1, 2, 0).numpy()
    predicted_label = predicted.item()
    predicted_class_name = Classes[predicted_label]

    buffered = BytesIO()
    image2.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template('result.html', img=img_str, predicted_class=predicted_class_name, title='Result')

@defaultAPI.route('/img/<filename>')
def send_img(filename):
    return send_from_directory('./static/img', filename)
@defaultAPI.route('/robots.txt')
def send_robots():
    return send_from_directory('./static', 'robots.txt')

@defaultAPI.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('index'))
