from flask import Flask, render_template, request
import os

# Webserver gateway interface
app = Flask(__name__)

# Paths
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')
ROI_PATH = os.path.join(BASE_PATH, 'static/roi/')
PREDICT_PATH = os.path.join(BASE_PATH, 'static/predict/')

# Ensure folders exist
os.makedirs(UPLOAD_PATH, exist_ok=True)
os.makedirs(ROI_PATH, exist_ok=True)
os.makedirs(PREDICT_PATH, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        if upload_file:
            filename = upload_file.filename
            save_path = os.path.join(UPLOAD_PATH, filename)
            upload_file.save(save_path)

            # Process image with deeplearning.py (call prediction function)
            from deeplearning import process_image
            predicted_text, roi_path, prediction_path = process_image(save_path)

            return render_template('index.html', uploaded_image=save_path, roi_image=roi_path, prediction_image=prediction_path, prediction_text=predicted_text)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
