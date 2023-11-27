import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'.\tesseract.exe'

# Load the image
img = cv2.imread("img1.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blur = cv2.GaussianBlur(gray, (9, 9), 0)

# Apply Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# Dilate the edges to connect nearby edges
dilated_edges = cv2.dilate(edges, None, iterations=2)
eroded_edges = cv2.erode(dilated_edges, None, iterations=1)

# Invert the image to have dark digits on light background
inverted = cv2.bitwise_not(eroded_edges)

# Find contours in the image
contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area in descending order
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Iterate through contours to find the region of interest (ROI)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    if w > 10 and h > 10:  # Adjust the threshold based on your specific case
        # Calculate the position and size of the 4 rectangles
        x_roi1 = min(x + 140, img.shape[1] - 1)  # Ensure x_roi1 is within the image width
        y_roi = y + int(2 / 2.54 * 96)  # 2cm converted to pixels at 96 DPI
        w_roi = int(2.5 / 2.54 * 96)
        h_roi = int(0.8 / 2.54 * 96)  # 2cm converted to pixels at 96 DPI

        # Extract the 4 ROIs from the image
        roi1 = img[y_roi:y_roi + h_roi, x_roi1:x_roi1 + w_roi]

        # Perform OCR on each ROI
        text1 = pytesseract.image_to_string(roi1, config="--psm 6 digits")

        print("OCR Result:", text1)

        # Draw rectangles around the detected regions for visualization
        cv2.rectangle(img, (x_roi1, y_roi), (x_roi1 + w_roi, y_roi + h_roi), (0, 255, 0), 2)
        # cv2.rectangle(img, (x_roi2, y_roi), (x_roi2 + w_roi, y_roi + h_roi), (0, 255, 0), 2)
        # cv2.rectangle(img, (x_roi3, y_roi), (x_roi3 + w_roi, y_roi + h_roi), (0, 255, 0), 2)
        # cv2.rectangle(img, (x_roi4, y_roi), (x_roi4 + w_roi, y_roi + h_roi), (0, 255, 0), 2)

        break  # Stop after processing the first contour (largest contour)

# Save the final result
cv2.imwrite("result.jpg", img)

# Display the results
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

