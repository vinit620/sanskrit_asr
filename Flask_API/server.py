import os
import random
from stt import stt
from flask import Flask, request, jsonify

SAMPLE_AUDIO = './Test_Audio/sp017-000041_001.wav'

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # get audio file and save it
    audio_file = request.files['file']
    file_name = str(random.randint(0, 10000))
    audio_file.save(file_name)

    # load model
    model = stt()
    predictions = model.predict([SAMPLE_AUDIO, file_name])

    # remove saved audio
    os.remove(file_name)

    # return prediction
    data = {'Prediction': predictions[-1]}
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=False, port=8000)
