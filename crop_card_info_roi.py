import os
import cv2


def crop_image_ROI(input_dir, output_dir, target_width=512, roi_height_ratio=0.2, roi_width_ratio=0.2):
    """
    Preprocess images by resizing them to a standard width and isolating the bottom-left text region.

    Args:
        input_dir (str): Path to the directory containing raw cropped images.
        output_dir (str): Path to the directory to save processed images.
        target_width (int): Target width for resizing images (default: 512 pixels).
        roi_height_ratio (float): Height ratio for bottom-left ROI (default: 0.2).
        roi_width_ratio (float): Width ratio for bottom-left ROI (default: 0.4).

    Returns:
        None: Processed images are saved to the `output_dir`.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each image in the input directory
    for file_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, file_name)

        if os.path.isfile(image_path):
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {file_name}")
                continue

            # Step 1: Resize the image to a standard width while maintaining aspect ratio
            h, w = image.shape[:2]
            aspect_ratio = h / w
            target_height = int(target_width * aspect_ratio)
            resized_image = cv2.resize(
                image, (target_width, target_height), interpolation=cv2.INTER_AREA
            )

            # Step 2: Crop the bottom-left region
            h, w = resized_image.shape[:2]
            roi_h = int(h * roi_height_ratio)  # Bottom 20% of height
            roi_w = int(w * roi_width_ratio)  # Left 40% of width
            roi = resized_image[h - roi_h :, :roi_w]  # Bottom-left region

            # Step 3: Save the processed ROI
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, roi)


if __name__ == "__main__":
    # Input and output directories
    input_dir = "data/images/cropped-cards/"
    output_dir = "data/images/roi/"

    # Preprocess images
    crop_image_ROI(
        input_dir=input_dir,
        output_dir=output_dir,
        target_width=512,
        roi_height_ratio=0.15,
        roi_width_ratio=0.25,
    )
