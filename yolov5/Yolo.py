import cv2
import numpy as np
import pytesseract as pt

# Đọc mô hình ONNX (YOLO)
net = cv2.dnn.readNetFromONNX('./yolov5/runs/train/Model5/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Cấu hình kích thước đầu vào
INPUT_HEIGHT = 640
INPUT_WIDTH = 640

def get_detections(img, net):
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # Chuẩn bị đầu vào cho mô hình YOLO
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    return input_image, detections

def non_maximum_supression(input_image, detections):
    boxes = []
    confidences = []
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.2:
            class_score = row[5]
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

    # Áp dụng NMS (Non-Maximum Suppression)
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)
    return boxes_np, confidences_np, index

def drawings(image, boxes_np, confidences_np, index):
    # Vẽ hộp giới hạn và thông tin
    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)

        license_text = extract_text(image, boxes_np[ind])

        # Vẽ hộp giới hạn và thêm các thông tin
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), thickness=2)
        image = cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 0, 255), thickness=2)
        image = cv2.rectangle(image, (x, y + h), (x + w, y + h + 25), (0, 0, 0), thickness=2)

        image = cv2.putText(image, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        image = cv2.putText(image, license_text, (x, y + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 3)

    return image

def yolo_predictions(img, net):
    input_image, detections = get_detections(img, net)
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    result_img = drawings(input_image, boxes_np, confidences_np, index)
    return result_img

def extract_text(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y + h, x:x + w]

    if 0 in roi.shape:
        return 'no number'

    else:
        text = pt.image_to_string(roi)
        text = text.strip()
        return text


import cv2

cap = cv2.VideoCapture('TEST/TEST.mp4')

nFrame = 10 # Show nFrame only
cFrame = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret == False:
        print('Unable to read video')
        break

    results = yolo_predictions(frame,net)

    # cv2.namedWindow('YOLO',cv2.WINDOW_KEEPRATIO)
    cv2_imshow(results)
    if cv2.waitKey(1) == 27 :
        break
    cFrame = cFrame+1
    if (cFrame > nFrame):
        break
cv2.destroyAllWindows()
cap.release()