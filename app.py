# -*- coding: utf-8 -*-
from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop


app = Flask(__name__)
CORS(app)
# CORS(app, origins=['https://bench-map.herokuapp.com'])

MAX_CONTENT_LENGTH = 2*1024*1024 # 2*1MB
ALLOWED_EXTENSION = set(['jpg', 'jpeg', 'png'])
image_size = 50


@app.route('/api/predict', methods=['POST'])
def predict():
    data_uri = request.form['data_uri']
    if not data_uri:
        return jsonify(dict(
            code=400,
            error='Bad Request',
            description='画像が送信されませんでした',
        ))
    header, image_base64 = data_uri.split(",", 1)
    print(header)
    print(image_base64[0:8])
    extension = header.split('/', 1)[1].split(';', 1)[0].lower()
    print(extension)
    if extension not in ALLOWED_EXTENSION:
        return jsonify(dict(
            code=415,
            error='UnsupportedMediaType',
            description='拡張子に誤りがあります',
        ))
    binary_image = base64.b64decode(image_base64)
    if len(binary_image) > MAX_CONTENT_LENGTH:
        return jsonify(dict(
            code=413,
            error='RequestEntityTooLarge',
            description='画像のサイズが大きすぎます',
        ))
    pillow_image = Image.open(BytesIO(binary_image))
    pillow_image = pillow_image.convert('RGB')
    pillow_image = pillow_image.resize((image_size, image_size))
    data = np.asarray(pillow_image)
    X = []
    X.append(data)
    X = np.array(X)

    # model file load without optimizer
    model = load_model('./bench_cnn_aug.h5')
    # Before, require compile with optimizer
    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])
    result = model.predict([X])[0]
    predicted = result.argmax() # label index (0: image, 1: bench) of max percentage
    percentage = int(result[predicted] * 100)
    response = dict(
        code=200,
        label=str(predicted),
        percentage=str(percentage),
    )
    return jsonify(response)

if __name__ == "__main__":
    app.run()
