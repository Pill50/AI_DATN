import urllib.request 
from PIL import Image 
  
# Retrieving the resource located at the URL 
# and storing it in the file name a.png 
url = "https://res.cloudinary.com/dlwbf1u2g/image/upload/v1700900972/img2_fisnmo.png" 
urllib.request.urlretrieve(url, "geeksforgeeks.png") 
  
img = Image.open(r"geeksforgeeks.png") 
img.show()