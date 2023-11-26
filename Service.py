#!/usr/bin/env python3

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
import urllib.request 
from PIL import Image 
import cv2
import numpy as np
import pytesseract

app = FastAPI()

class RequestBody(BaseModel):
    url: str

def downloadImg(url: str):
    urllib.request.urlretrieve(url, "img.png") 
    
    img = Image.open(r"img.png") 
    img.show()

def detect():
    result = ""
    pytesseract.pytesseract.tesseract_cmd = r'./tesseract.exe'

    # Load the image
    img = cv2.imread("img.png")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

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
            x_roi1 = min(x + 124, img.shape[1] - 1)  # Ensure x_roi1 is within the image width
            y_roi = y + int(2.65 / 2.54 * 96)  # 2cm converted to pixels at 96 DPI
            w_roi = int(0.5 / 2.54 * 96)
            h_roi = int(0.8 / 2.54 * 96)  # 2cm converted to pixels at 96 DPI

            x_roi2 = min(x_roi1 + w_roi + 6, img.shape[1] - 1)  # Ensure x_roi2 is within the image width
            x_roi3 = min(x_roi2 + w_roi + 6, img.shape[1] - 1)  # Ensure x_roi3 is within the image width
            x_roi4 = min(x_roi3 + w_roi + 6, img.shape[1] - 1)  # Ensure x_roi4 is within the image width

            # Extract the 4 ROIs from the image
            roi1 = img[y_roi:y_roi + h_roi, x_roi1:x_roi1 + w_roi]
            roi2 = img[y_roi:y_roi + h_roi, x_roi2:x_roi2 + w_roi]
            roi3 = img[y_roi:y_roi + h_roi, x_roi3:x_roi3 + w_roi]
            roi4 = img[y_roi:y_roi + h_roi, x_roi4:x_roi4 + w_roi]

            # Perform OCR on each ROI
            text1 = pytesseract.image_to_string(roi1, config="--psm 6 digits")
            text2 = pytesseract.image_to_string(roi2, config="--psm 6 digits")
            text3 = pytesseract.image_to_string(roi3, config="--psm 6 digits")
            text4 = pytesseract.image_to_string(roi4, config="--psm 6 digits")

            print("OCR Result for ROI 1:", text1)
            print("OCR Result for ROI 2:", text2)
            print("OCR Result for ROI 3:", text3)
            print("OCR Result for ROI 4:", text4)
            
            result += text1 + text2 + text3 + text4
            return result.replace("\n","")
            # Draw rectangles around the detected regions for visualization
            cv2.rectangle(img, (x_roi1, y_roi), (x_roi1 + w_roi, y_roi + h_roi), (0, 255, 0), 2)
            cv2.rectangle(img, (x_roi2, y_roi), (x_roi2 + w_roi, y_roi + h_roi), (0, 255, 0), 2)
            cv2.rectangle(img, (x_roi3, y_roi), (x_roi3 + w_roi, y_roi + h_roi), (0, 255, 0), 2)
            cv2.rectangle(img, (x_roi4, y_roi), (x_roi4 + w_roi, y_roi + h_roi), (0, 255, 0), 2)

            break  # Stop after processing the first contour (largest contour)

    # Save the final result
    cv2.imwrite("result.jpg", img)

    # Display the results
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result

@app.post("/")
def handle_detection(requestBody: RequestBody):
    downloadImg(requestBody.url)
    res = detect()
    print("res: ", res)
    return res


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=4000)

