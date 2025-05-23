import streamlit as st
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import os
import io
import math # For calculating RMSE

st.set_page_config(layout="wide", page_title="Deep Digit Analyzer")

# Import models from vae_model.py
try:
    from vae_model import VAE, CNNClassifier
except ImportError:
    st.error("Could not import VAE or CNNClassifier. Make sure vae_model.py is in the same directory.")
    st.stop()

# Import the drawing component
from streamlit_drawable_canvas import st_canvas

# --- Configuration ---
VAE_MODEL_PATH = "vae_mnist.pth"
CNN_MODEL_PATH = "cnn_classifier_mnist.pth"
VAE_INPUT_DIM = 784
VAE_H_DIM = 256
VAE_Z_DIM = 20
IMAGE_SIZE = 28 # MNIST image size

# --- Load Models ---
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE
    vae_model = VAE(VAE_INPUT_DIM, VAE_H_DIM, VAE_Z_DIM).to(device)
    if os.path.exists(VAE_MODEL_PATH):
        try:
            vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=device))
            vae_model.eval()
            st.success("VAE model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading VAE model: {e}. Please ensure vae_mnist.pth exists and is valid.")
            st.stop()
    else:
        st.error(f"VAE model file not found at {VAE_MODEL_PATH}. Please train the VAE first by running vae_model.py.")
        st.stop()

    # Load CNN Classifier
    cnn_classifier = CNNClassifier(num_classes=10).to(device)
    if os.path.exists(CNN_MODEL_PATH):
        try:
            cnn_classifier.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
            cnn_classifier.eval()
            st.success("CNN Classifier loaded successfully!")
        except Exception as e:
            st.error(f"Error loading CNN Classifier model: {e}. Please ensure cnn_classifier_mnist.pth exists and is valid.")
            st.stop()
    else:
        st.error(f"CNN Classifier model file not found at {CNN_MODEL_PATH}. Please train the CNN first by running vae_model.py.")
        st.stop()

    return vae_model, cnn_classifier, device

vae_model, cnn_classifier, device = load_models()

# --- Preprocessing for Model Input ---
# This transformation matches the training data normalization for both VAE and CNN
transform_input = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Ensure single channel
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Resize to 28x28
    transforms.ToTensor(), # Convert to tensor (values 0-1)
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] as per model training
])

RECONSTRUCTION_ERROR_THRESHOLD = 0.05 # Example: Mean Squared Error threshold

# --- Streamlit App ---

st.title("Scribbled Digit Analyzer: VAE Reconstruction & CNN Prediction")
st.write("Draw a digit below, and see its VAE reconstruction, a numerical prediction, and an anomaly score!")

col_drawing, col_results = st.columns([1, 1])

with col_drawing:
    st.subheader("Draw your digit here:")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",  # White background
        stroke_width=15, # Thicker line for digit drawing
        stroke_color="black", # Black color for drawing
        background_color="#eee", # Light grey behind the drawing area
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

with col_results:
    st.subheader("Analysis Results:")
    if canvas_result.image_data is not None:
        # Convert RGBA to Grayscale PIL Image
        drawn_image_np = canvas_result.image_data.astype(np.uint8)
        drawn_image_pil = Image.fromarray(drawn_image_np).convert("L") # Convert to grayscale

        # Invert the image: MNIST digits are white on a black background
        inverted_image_pil = ImageOps.invert(drawn_image_pil)

        # Display the processed input for user's clarity
        st.image(inverted_image_pil, caption="Your Drawing (28x28 Processed)", width=150)

        # Preprocess the image for VAE and CNN
        input_tensor = transform_input(inverted_image_pil)
        input_tensor_batch = input_tensor.unsqueeze(0).to(device) # Add batch dimension

        # --- VAE Reconstruction ---
        with torch.no_grad():
            recon_image_tensor, mu, logvar = vae_model(input_tensor_batch)
            recon_image_tensor_disp = recon_image_tensor.cpu().view(1, IMAGE_SIZE, IMAGE_SIZE)

            # Denormalize reconstruction for display (from [-1,1] to [0,1])
            recon_image_display = (recon_image_tensor_disp * 0.5 + 0.5) * 255
            recon_image_display = recon_image_display.squeeze(0).byte().numpy()
            recon_image_pil = Image.fromarray(recon_image_display)

            st.image(recon_image_pil, caption="VAE Reconstruction", width=150)

        # --- CNN Classification Prediction ---
        with torch.no_grad():
            cnn_outputs = cnn_classifier(input_tensor_batch)
            _, predicted_class = torch.max(cnn_outputs, 1)
            st.metric(label="Predicted Digit", value=f"**{predicted_class.item()}**")

        # --- Anomaly Detection ---
        input_for_error = (input_tensor * 0.5) + 0.5 # input_tensor is already [1, 28, 28]
        recon_for_error = (recon_image_tensor.cpu().view(1, IMAGE_SIZE, IMAGE_SIZE) * 0.5) + 0.5

        # Calculate Mean Squared Error (MSE) per pixel, then average over all pixels
        reconstruction_error_mse = torch.mean((input_for_error - recon_for_error)**2).item()
        reconstruction_error_rmse = math.sqrt(reconstruction_error_mse) # RMSE is often more interpretable

        st.markdown(f"**Reconstruction Error (RMSE):** `{reconstruction_error_rmse:.4f}`")

        if reconstruction_error_rmse > RECONSTRUCTION_ERROR_THRESHOLD:
            # Changed from: st.warning(f"**Anomaly Detected!** This drawing might not be a standard digit. (Error > {RECONSTRUCTION_ERROR_THRESHOLD:.4f})")
            st.warning(f"**Potential Anomaly Detected!** (Error > {RECONSTRUCTION_ERROR_THRESHOLD:.4f})")
            st.info("High reconstruction error suggests the drawing is significantly different from typical MNIST digits learned by the VAE.")
        else:
            st.success(f"**Digit-like!** Low reconstruction error indicates a standard digit. (Error < {RECONSTRUCTION_ERROR_THRESHOLD:.4f})")
            st.info("Low reconstruction error indicates the VAE can effectively reconstruct your drawing, suggesting it's similar to the digits it was trained on.")

    else:
        st.info("Draw a digit on the left canvas to see its analysis.")

st.sidebar.header("About Me")
st.sidebar.markdown("""
Hello! I'm **Ashutosh Jena**, an aspiring **Data Scientist & AI Engineer**.

This project is a hands-on exploration into **deep generative models** and **interactive AI applications**. I developed it to demonstrate:

* The power of **Variational Autoencoders (VAEs)** in understanding and reconstructing complex data patterns.
* The effectiveness of **Convolutional Neural Networks (CNNs)** for image classification.
* The versatility of **Streamlit** for building engaging, user-friendly web applications that showcase machine learning models.

I'm passionate about **computer vision, natural language processing, ethical AI, scalable ML** and constantly seeking new challenges to apply and expand my knowledge in the field of artificial intelligence.

Feel free to connect with me or explore my other projects!

* **GitHub:**https://github.com/ajverse
* **LinkedIn:**https://www.linkedin.com/in/ashutosh-jena-67a3b91ba/
""")