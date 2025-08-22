from ultralytics import YOLO
import cv2
import os

def detect_on_webcam(model_path="models/yolov8n-face.pt", cam_index=0):
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    # Load model
    model = YOLO(model_path)

    # Open webcam
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return

    print("✅ Press 'q' to quit webcam window")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Run YOLO detection
        results = model(frame)

        # Annotate
        annotated_frame = results[0].plot()

        # Show output
        cv2.imshow("YOLO Face Detection", annotated_frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_on_webcam()
