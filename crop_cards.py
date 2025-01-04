import os
import cv2
from ultralytics import YOLO
import numpy as np

TARGET_ASPECT_RATIO = 2.5 / 3.5  # MTG card aspect ratio
DISQUALIFYING_CLASSES = ["person"]


def load_model(model_path="yolov8m-seg.pt"):
    """Load the YOLOv8 model."""
    return YOLO(model_path)


def filter_best_box(results, target_aspect_ratio, disqualifying_classes, model):
    """Filter the best bounding box based on confidence, aspect ratio, and disqualifying classes."""
    best_box = None
    best_score = 0

    for result in results:
        for box in result.boxes:

            # Skip disqualifying classes
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]

            if class_name in disqualifying_classes:
                continue

            # Extract bounding box details
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].item()

            # Calculate bounding box width, height, and aspect ratio
            width, height = x2 - x1, y2 - y1
            aspect_ratio = width / height

            # Score based on confidence and aspect ratio similarity
            aspect_ratio_score = 1 - abs(aspect_ratio - target_aspect_ratio)
            score = confidence * aspect_ratio_score

            # Update the best box
            if score > best_score:
                best_score = score
                best_box = (int(x1), int(y1), int(x2), int(y2))

    return best_box


def crop_card(image, box):
    """Crop the card using the bounding box."""
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


def save_annotated_image(result, output_path):
    """Save the annotated image using result.plot()."""
    annotated_image = result.plot()  # Generates an annotated image with bounding boxes
    cv2.imwrite(output_path, annotated_image)


def process_image(model, image_path, cropped_output_dir, annotated_output_dir, target_aspect_ratio):
    """Process a single image, crop the card, and save both the cropped and annotated images."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Run YOLO model
    results = model(image_path, task="segment")

    for i, result in enumerate(results):
        # Save annotated image
        annotated_output_path = os.path.join(annotated_output_dir, f"{os.path.basename(image_path)}")
        save_annotated_image(result, annotated_output_path)

        # Filter the best bounding box
        best_box = filter_best_box([result], target_aspect_ratio, DISQUALIFYING_CLASSES, model)
        if best_box:
            # Crop the card and save the result
            cropped_card = crop_card(image, best_box)
            cropped_output_path = os.path.join(cropped_output_dir, f"{os.path.basename(image_path)}")
            cv2.imwrite(cropped_output_path, cropped_card)
        else:
            print(f"No valid card detected in: {image_path}")


def process_directory(model, input_dir, cropped_output_dir, annotated_output_dir, target_aspect_ratio):
    """Batch process all images in a directory."""
    os.makedirs(cropped_output_dir, exist_ok=True)
    os.makedirs(annotated_output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, file_name)
        if os.path.isfile(image_path):
            process_image(
                model, image_path, cropped_output_dir, annotated_output_dir, target_aspect_ratio
            )


if __name__ == "__main__":
    # Paths
    model_path = "yolov8m-seg.pt"
    input_dir = "data/images/wild/"
    cropped_output_dir = "data/images/cropped/"
    annotated_output_dir = "data/images/annotated"

    # Load YOLO model
    model = load_model(model_path)

    # Batch process the directory
    process_directory(model, input_dir, cropped_output_dir, annotated_output_dir, TARGET_ASPECT_RATIO)
