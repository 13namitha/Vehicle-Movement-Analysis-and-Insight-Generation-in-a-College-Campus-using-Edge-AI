import cv2
import easyocr
import imghdr

class LicensePlateRecognizer:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader for English language

    def preprocess_image(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to enhance text visibility
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return opening

    def recognize(self, frame, detection):
        # Perform license plate recognition on the detection
        if frame is None:
            print("Error: Frame is None. Check image loading and detection.")
            return None
        x, y, x_end, y_end = detection
        if x is None or y is None or x_end is None or y_end is None:
            print("Error: Detection is invalid. Check detection algorithm.")
            return None
        crop = frame[y:y_end, x:x_end]
        
        # Preprocess the cropped image
        preprocessed_image = self.preprocess_image(crop)
        
        # Perform OCR using EasyOCR
        result = self.reader.readtext(preprocessed_image)
        
        license_plate = ""
        for res in result:
            license_plate += res[1] + " "
        
        return license_plate.strip()

# Example usage
recognizer = LicensePlateRecognizer()

# Load an image containing a license plate
image = cv2.imread("C:/Users/namit/OneDrive/Pictures/Desktop/yolov8parkingspace-main/vehicleLicensePlate_Images")
# Detect the license plate region (assuming you have a detection algorithm)
detection = (10, 10, 200, 50)  # x, y, x_end, y_end coordinates of the detection

# Recognize the license plate
license_plate = recognizer.recognize(image, detection)
print("Recognized license plate:", license_plate)