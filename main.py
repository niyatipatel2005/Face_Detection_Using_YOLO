from utils.detect_faces_from_image import detect_on_image
from utils.detect_face_from_webcam import detect_on_webcam

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Detect faces in Image")
    print("2. Detect faces in Webcam")

    try:
        choice = input("Enter choice (1/2, default=1): ").strip() or "1"
    except KeyboardInterrupt:
        print("\n[INFO] Exiting program...")
        exit()

    if choice == "1":
        detect_on_image()
    elif choice == "2":
        detect_on_webcam()
    else:
        print("Invalid choice! Please enter 1 or 2.")
