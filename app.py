from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import numpy as np
import sys

app = Flask(__name__)


# Define paths to YOLO model files
model_weights = r"C:\Users\pavit\OneDrive\Documents\imgpath\yolov3.weights"
model_config = r"C:\Users\pavit\OneDrive\Documents\imgpath\yolov3.cfg"
class_names = r"C:\Users\pavit\OneDrive\Documents\imgpath\coco.names"

# Function to perform object detection
def detect_objects(image_path=None, video_path=None):
    # Load YOLO
    net = cv2.dnn.readNet(model_weights, model_config)
    classes = []
    with open(class_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Determine if detecting objects in an image, video, or webcam
    is_image = image_path is not None
    is_video = video_path is not None

    if is_image:
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            return jsonify({'error': f"Unable to load image at {image_path}"})
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
    elif is_video:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': f"Unable to open video file at {video_path}"})
    else:
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({'error': "Unable to open webcam"})

    while True:
        if is_video or not is_image:
            ret, frame = cap.read()
            if not ret:
                break
            if is_video:
                # Resize the frame to 640x480 for better visualization in video
                frame = cv2.resize(frame, (640, 480))

            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)

        # Run forward pass
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Process detection results
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Apply non-maximum suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Display names of detected objects
        for i in range(len(boxes)):
            if i in indices:
                box = boxes[i]
                x, y, w, h = box
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                text = f"{label} {confidence:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the result
        cv2.imshow("Object Detection", frame)

        # Check for 'Esc' key or 'x' button to exit
        key = cv2.waitKey(1)
        if key == 27:  # 'Esc' key
            break

    # Release resources
    if is_video or not is_image:
        cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    # Handle the start detection form submission
    image_path = request.files['image_path']
    image_path.save('uploaded_image.jpg')  # Save the uploaded image
    detect_objects(image_path='uploaded_image.jpg')
    video_path = request.files['video_path']
    video_path.save('uploaded_video.mp4')  # Save the uploaded image
    detect_objects(video_path='uploaded_video.mp4')
    return redirect(url_for('index'))  # Redirect to the index page
   

@app.route('/start_simulation', methods=['POST'])
def start_simulation_route():

    return redirect(url_for('index'))
    
@app.route('/exit', methods=['POST'])
def exit():
    # Handle the exit form submission
    # Here you can perform any necessary processing
    print("Exiting the program...")
    sys.exit()


if __name__ == '__main__':
    app.run(debug=True) 