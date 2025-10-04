import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from sklearn.linear_model import LogisticRegression

# ---------------------------
# Title
# ---------------------------
st.set_page_config(page_title="Fluorescence RGB Analyzer", layout="wide")
st.title("🔬 Fluorescence RGB Analyzer with AI & ROI Selection")

st.write("Upload a fluorescence image, drag a rectangle ROI, and analyze RGB + intensity.")

# ---------------------------
# Upload Image
# ---------------------------
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------------------------
    # ROI Selection using Canvas
    # ---------------------------
    st.subheader("🖼️ Draw ROI on Image")
    canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # Transparent red fill
        stroke_width=2,
        background_image=image,
        update_streamlit=True,
        height=image.size[1],
        width=image.size[0],
        drawing_mode="rect",  # User draws rectangle
        key="canvas",
    )

    # ---------------------------
    # ROI Processing
    # ---------------------------
    if canvas.json_data is not None and len(canvas.json_data["objects"]) > 0:
        # Take last drawn rectangle
        obj = canvas.json_data["objects"][-1]
        x, y, w, h = int(obj["left"]), int(obj["top"]), int(obj["width"]), int(obj["height"])

        if w > 0 and h > 0:
            crop = np.array(image)[y:y+h, x:x+w]

            st.image(crop, caption="Selected ROI", use_column_width=True)

            # Calculate average RGB in ROI
            R = crop[:, :, 0]
            G = crop[:, :, 1]
            B = crop[:, :, 2]

            avg_R = np.mean(R)
            avg_G = np.mean(G)
            avg_B = np.mean(B)

            # Intensity using luminance formula
            intensity = 0.2126 * avg_R + 0.7152 * avg_G + 0.0722 * avg_B

            st.subheader("📊 ROI Channel Analysis")
            st.write(f"**Average Red (R):** {avg_R:.2f}")
            st.write(f"**Average Green (G):** {avg_G:.2f}")
            st.write(f"**Average Blue (B):** {avg_B:.2f}")
            st.write(f"**Overall Intensity (weighted):** {intensity:.2f}")

            # Histograms
            st.subheader("📈 ROI RGB Histograms")
            fig, ax = plt.subplots()
            ax.hist(R.ravel(), bins=50, color="red", alpha=0.5, label="Red")
            ax.hist(G.ravel(), bins=50, color="green", alpha=0.5, label="Green")
            ax.hist(B.ravel(), bins=50, color="blue", alpha=0.5, label="Blue")
            ax.set_title("ROI RGB Intensity Distribution")
            ax.set_xlabel("Pixel value")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)

            # ---------------------------
            # Simple AI Analyzer (Demo)
            # ---------------------------
            st.subheader("🤖 AI Fluorescence Analyzer")

            # Example: Logistic Regression trained on fake data
            # X = [[R,G,B,Intensity]] , y = [0=Low, 1=High]
            X_train = [[50, 60, 40, 55], [200, 210, 180, 205], [100, 90, 80, 95], [230, 240, 220, 235]]
            y_train = [0, 1, 0, 1]

            model = LogisticRegression()
            model.fit(X_train, y_train)

            pred = model.predict([[avg_R, avg_G, avg_B, intensity]])[0]
            label = "🌑 Low Fluorescence" if pred == 0 else "🌕 High Fluorescence"

            st.write(f"**AI Prediction:** {label}")
        else:
            st.warning("⚠️ Please draw a valid ROI (non-zero size).")
