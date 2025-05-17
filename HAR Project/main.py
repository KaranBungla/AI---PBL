from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import imutils
import os
from werkzeug.utils import secure_filename

MODEL_PATH = 'resnet-34_kinetics.onnx'
CLASSES_PATH = 'Actions.txt'
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112
UPLOAD_FOLDER = 'uploads'

with open(CLASSES_PATH) as f:
    ACT = f.read().strip().split('\n')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_video(filepath):
    cap = cv2.VideoCapture(filepath)
    predictions = []
    ACT = open(CLASSES_PATH).read().strip().split("\n")
    SAMPLE_DURATION = 16
    SAMPLE_SIZE = 112
    gp = cv2.dnn.readNet(MODEL_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Video FPS:", fps)
    while True:
        frames = []
        originals = []
        for i in range(SAMPLE_DURATION):
            grabbed, frame = cap.read()
            if not grabbed:
                break
            originals.append(frame)
            frame = imutils.resize(frame, width=400)
            frames.append(frame)
        if len(frames) < SAMPLE_DURATION:
            break
        blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)
        gp.setInput(blob)
        outputs = gp.forward()
        label = ACT[np.argmax(outputs)]
        predictions.append(label)
    cap.release()
    if predictions:
        from collections import Counter
        most_common_pred, _ = Counter(predictions).most_common(1)[0]
        emoji_map = {
            'Archery': '🏹',
            'Basketball': '⛹️',
            'Blowing Candles': '🕯️',
            'Brushing Teeth': '🪥',
            'BOWLING': '🎳',
            'Drumming': '🥁',
            'Knitting': '🧶',
            'Lunges': '🦵',
            'Mopping Floor': '🧹',
            'Playing Cello': '🎻',
            'Playing Guitar': '🎸',
            'Playing Tabla': '🥁',
            'Table Tennis Shot': '🏓',
            'Typing': '⌨️',
            'PUSH UP': '💪',
            'SKIPPING ROPE': '🤾🪢',
            'BLOWING OUT CANDLES': '🎂',
        }
        pred = most_common_pred
        # Map 'USING COMPUTER' to 'Typing' with emoji
        if 'using computer' in pred.lower():
            pred = 'Typing ⌨️'
        elif 'playing cymbals' in pred.lower():
            pred = 'Playing Tabla 🥁'
        else:
            # Only append emoji if not already present
            for k, v in emoji_map.items():
                if k.lower() in pred.lower() and v not in pred:
                    pred = f"{pred} {v}"
                    break
        # Return as HTML to preserve emoji rendering
        return [pred]
    else:
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    file_size = os.path.getsize(filepath)
    print(f"[DEBUG] Saved file: {filepath}, size: {file_size} bytes")
    if file_size == 0:
        return jsonify({'error': 'Uploaded file is empty or corrupted.'}), 400
    predictions = predict_video(filepath)
    os.remove(filepath)
    if predictions:
        return jsonify({'predictions': [predictions[-1]]})
    else:
        return jsonify({'error': 'Could not process video'}), 500

if __name__ == '__main__':
    app.run(debug=True)

# local server
# http://localhost:5000