import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title="MRI Denoiser", layout="centered")

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# -------------------------------
# MODEL ARCHITECTURE
# -------------------------------
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = DenoisingAutoencoder()
    model.load_state_dict(torch.load("denoising_autoencoder.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("🧠 MRI Denoising AI Tool")
st.markdown("Upload a **.pt, .png, or .jpg MRI image (64x64 recommended)**")

st.info("⚠️ For best results, use grayscale MRI images. Non-medical images may give poor results.")

uploaded_file = st.file_uploader("Upload Image", type=["pt", "png", "jpg", "jpeg"])

# -------------------------------
# PREPROCESS FUNCTION
# -------------------------------
def preprocess_image(file):
    if file.name.endswith(".pt"):
        img = torch.load(file)

        if img.dim() == 2:
            img = img.unsqueeze(0).unsqueeze(0)

        return img.float()

    else:
        image = Image.open(file).convert("L")
        image = image.resize((64, 64), Image.BICUBIC)

        img_np = np.array(image) / 255.0
        img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).float()

        return img_tensor

# -------------------------------
# UPSCALE FOR DISPLAY
# -------------------------------
def upscale_for_display(img, size=256):
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)

    # Remove potential 1-pixel border
    img_uint8 = img_uint8[1:-1, 1:-1]

    return np.array(
        Image.fromarray(img_uint8).resize((size, size), Image.NEAREST)
    )
# -------------------------------
# INFERENCE
# -------------------------------
if uploaded_file is not None:

    noisy_img = preprocess_image(uploaded_file)

    with torch.no_grad():
        output = model(noisy_img)
        denoised_img = output.squeeze().numpy()

    # Upscale for better viewing
    input_display = upscale_for_display(noisy_img.squeeze().numpy())
    output_display = upscale_for_display(denoised_img)

    # -------------------------------
    # DISPLAY
    # -------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Noisy Input")
        st.image(input_display, clamp=True)

    with col2:
        st.subheader("Denoised Output")
        st.image(output_display, clamp=True)

    # -------------------------------
    # DOWNLOAD AS PNG
    # -------------------------------
    if st.button("Download Denoised Image (PNG)"):
        img = (denoised_img * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        pil_img.save("denoised_output.png")

        with open("denoised_output.png", "rb") as f:
            st.download_button(
                "Click to Download",
                f,
                file_name="denoised_output.png"
            )