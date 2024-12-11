import streamlit as st
import cv2
import numpy as np

# Function to apply edge detection and extract the boundaries of the anime character
def extract_anime_boundaries(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty image to draw contours
    contour_image = np.zeros_like(image)
    
    # Draw the contours on the empty image
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)
    
    return contour_image

# Streamlit app
st.title("Anime Character Boundary Extraction")

uploaded_file = st.file_uploader("Drag and drop an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Displaying the uploaded image
    st.image(image_rgb, caption='Uploaded Image', use_container_width=True)
    
    if st.button("Extract Boundaries"):
        #Extracting boundaries
        result_img = extract_anime_boundaries(image_rgb)
        if result_img is not None:
            st.image(result_img, caption='Anime Character Boundaries', use_container_width=True)
        else:
            st.write("No boundaries detected in the image.")
