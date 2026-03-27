import streamlit as st
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
# UI DESIGN
# -------------------------------


st.title("🧠 MRI Denoising AI Tool")
st.markdown("Upload a **.pt or .png MRI image (64x64)** and get a denoised output.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["pt", "png", "jpg", "jpeg"])

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
        image = Image.open(file).convert("L")  # grayscale
        image = image.resize((64, 64))

        img_np = np.array(image) / 255.0
        img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).float()

        return img_tensor

# -------------------------------
# INFERENCE
# -------------------------------
if uploaded_file is not None:

    noisy_img = preprocess_image(uploaded_file)

    with torch.no_grad():
        output = model(noisy_img)
        denoised_img = output.squeeze().numpy()

    # -------------------------------
    # DISPLAY
    # -------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Noisy Input")
        st.image(noisy_img.squeeze().numpy(), clamp=True, use_container_width=True)

    with col2:
        st.subheader("Denoised Output")
        st.image(denoised_img, clamp=True, use_container_width=True)

    # -------------------------------
    # DOWNLOAD BUTTON
    # -------------------------------
    if st.button("Download Denoised Output"):
        torch.save(torch.tensor(denoised_img), "denoised_output.pt")
        with open("denoised_output.pt", "rb") as f:
            st.download_button("Download File", f, file_name="denoised_output.pt")