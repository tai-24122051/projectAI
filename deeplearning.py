import os
from PIL import Image

def process_image(image_path):
    """
    This function takes the uploaded image, runs the OCR and NPD pipeline,
    and saves the ROI and prediction images.
    """
    # Placeholder for ML model loading
    # model = load_your_model()

    # Placeholder for image processing
    # e.g., detect_plate(image_path)
    roi_path = os.path.join('static/roi/', 'example_roi.jpg')  # Save ROI here
    prediction_path = os.path.join('static/predict/', 'example_prediction.jpg')  # Save final output here
    predicted_text = "Example Number Plate Text"


    Image.open(image_path).convert("RGB").save(roi_path)
    Image.open(image_path).convert("RGB").save(prediction_path)

    return predicted_text, roi_path, prediction_path
