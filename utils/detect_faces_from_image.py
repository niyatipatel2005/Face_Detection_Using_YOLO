from ultralytics import YOLO
import cv2
import os
import time

def detect_on_image(
    model_path="models/yolov8n-face.pt",
    image_path="sample.webp",
    save_dir="outputs"
):
    # Check paths
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"❌ Image not found at {image_path}")
        return

    # Load model
    model = YOLO(model_path)

    # Run detection
    results = model(image_path)

    # Create output directories
    os.makedirs(save_dir, exist_ok=True)
    faces_dir = os.path.join(save_dir, "faces")
    os.makedirs(faces_dir, exist_ok=True)

    # Get annotated image (with bounding boxes)
    annotated_image = results[0].plot()

    # Save annotated image with timestamp
    timestamp = int(time.time())
    output_path = os.path.join(save_dir, f"output_{timestamp}.jpg")
    cv2.imwrite(output_path, annotated_image)
    print(f"[INFO] Annotated image saved at: {output_path}")

    # Crop and save faces
    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
    img = cv2.imread(image_path)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        w, h = x2 - x1, y2 - y1

        # Ignore very small detections
        if w < 20 or h < 20:
            continue

        face = img[y1:y2, x1:x2]

        face_path = os.path.join(faces_dir, f"face_{i+1}_{timestamp}.jpg")
        cv2.imwrite(face_path, face)
        print(f"[INFO] Face {i+1} saved at: {face_path}")

    # Show annotated image (only for normal Python, not Streamlit)
    cv2.imshow("YOLO Face Detection - Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results


if __name__ == "__main__":
    detect_on_image()
