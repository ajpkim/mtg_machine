import cv2
import pytesseract
from pytesseract import Output

image_path = "data/images/ltr-95-mirkwood-bats.jpg"
image = cv2.imread(image_path)

# Preprocess the image
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to binarize the image
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Run OCR
custom_config = r"--psm 6"  # PSM 6 is for text blocks with uniform alignment
ocr_result = pytesseract.image_to_string(binary, config=custom_config)

print("OCR Result:")
print(ocr_result)
