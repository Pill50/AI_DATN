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
    url = url.replace("https","http")
    urllib.request.urlretrieve(url, "img.jpg") 
    
    img = Image.open(r"img.jpg") 
    img.show()

def detect():
    pytesseract.pytesseract.tesseract_cmd = r'.\tesseract.exe'

    # Load the image
    img = cv2.imread("img.jpg")

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

        if w > 10 and h > 10:  
            x_roi1 = 143 
            y_roi = 75
            w_roi = int(2.5 / 2.54 * 96)
            h_roi = int(0.7 / 2.54 * 96)  

            roi1 = img[y_roi:y_roi + h_roi, x_roi1:x_roi1 + w_roi]

            text1 = pytesseract.image_to_string(roi1, config="--psm 6 digits")

            print("OCR Result:", text1)
            return text1.replace("\n","")


    cv2.imwrite("result.jpg", img)
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



@app.post("/")
def handle_detection(requestBody: RequestBody):
    downloadImg(requestBody.url)
    res = detect()
    print("res: ", res)
    return res


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=4000)

