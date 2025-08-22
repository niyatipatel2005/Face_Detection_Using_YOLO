import os
import cv2
import time

def save_cropped_faces(image_path, results, output_dir="outputs/faces"):
    """
    Crop and save detected faces from an image.
    Args:
        image_path (str): path to original image
        results (YOLO result): detection result
        output_dir (str): folder where cropped faces will be saved
    """
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return []

    if len(results[0].boxes.xyxy) == 0:
        print("[INFO] No faces detected.")
        return []

    saved_faces = []
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = int(time.time())

    face_count = 0
    for box in results[0].boxes.xyxy:  # loop through detected bounding boxes
        x1, y1, x2, y2 = map(int, box.tolist())
        cropped_face = img[y1:y2, x1:x2]

        face_path = os.path.join(output_dir, f"{base_name}_face_{face_count}_{timestamp}.jpg")
        cv2.imwrite(face_path, cropped_face)
        saved_faces.append(face_path)
        face_count += 1

    print(f"[INFO] Saved {face_count} faces to {output_dir}")
    return saved_faces
