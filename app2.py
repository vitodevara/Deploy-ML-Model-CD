from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from PIL import Image
import os
import pandas as pd
from ultralytics import YOLO

app = Flask(__name__)
run_with_ngrok(app)  # ngrok to create a public URL
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model_dataset = YOLO("best50n-single.pt")

def process_image(image_path):
    # Perform inference
    results = model_dataset(image_path)
    
    # Convert results to DataFrame
    boxes_list = results[0].boxes.data.tolist()
    columns = ['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id']
    
    for i in boxes_list:
        i[:4] = [round(i, 1) for i in i[:4]]
        i[5] = int(i[5])
        i.append(results[0].names[i[5]])

    columns.append('class_name')
    result_df = pd.DataFrame(boxes_list, columns=columns)
    
    # Filter out 'X' class_name and convert class_name to string
    result_df = result_df[result_df['class_name'] != 'X']
    result_df['class_name'] = result_df['class_name'].astype(str)
    
    return result_df

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        result_df = process_image(filename)
        
        # Convert DataFrame to JSON format
        result_json = result_df.to_dict(orient='records')
        
        return jsonify(result_json)

if __name__ == '__main__':
    app.run()
