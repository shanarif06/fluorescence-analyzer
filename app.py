import streamlit as st
import numpy as np
from PIL import Image

st.title("🔬 Fluorescence Image Analyzer")
st.write("Upload a fluorescence image to calculate average intensity.")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy array
    img_array = np.array(image)

    # Calculate intensity values
    avg_intensity = np.mean(img_array)
    min_intensity = np.min(img_array)
    max_intensity = np.max(img_array)

    # Display results
    st.subheader("📊 Intensity Analysis")
    st.write(f"**Average Intensity:** {avg_intensity:.2f}")
    st.write(f"**Minimum Intensity:** {min_intensity}")
    st.write(f"**Maximum Intensity:** {max_intensity}")

    # Optional: histogram
    st.subheader("🔎 Histogram of Pixel Intensities")
    st.bar_chart(np.histogram(img_array, bins=50)[0])
