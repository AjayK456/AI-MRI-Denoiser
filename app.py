import streamlit as st
import torch
import matplotlib.pyplot as plt
import os

# -------------------------------
# DEBUG (optional – remove later)
# -------------------------------
st.write("Files in repo:", os.listdir())

# -------------------------------
# LOAD FULL MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = torch.load("denoising_full_model.pth", map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("🧠 MRI Denoising AI Tool")
st.write("Upload a noisy MRI `.pt` file (64x64) to get a denoised output.")

uploaded_file = st.file_uploader("Upload MRI file", type=["pt"])

# -------------------------------
# INFERENCE
# -------------------------------
if uploaded_file is not None:
    noisy_img = torch.load(uploaded_file)

    # Ensure correct shape
    if noisy_img.dim() == 2:
        noisy_img = noisy_img.unsqueeze(0).unsqueeze(0)

    noisy_img = noisy_img.float()

    with torch.no_grad():
        output = model(noisy_img)
        denoised_img = output.squeeze().numpy()

    # -------------------------------
    # VISUALIZATION
    # -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(noisy_img.squeeze(), cmap='gray')
    axes[0].set_title("Noisy Input")
    axes[0].axis("off")

    axes[1].imshow(denoised_img, cmap='gray')
    axes[1].set_title("Denoised Output")
    axes[1].axis("off")

    st.pyplot(fig)

    # -------------------------------
    # DOWNLOAD OUTPUT
    # -------------------------------
    if st.button("Download Denoised Image"):
        torch.save(torch.tensor(denoised_img), "denoised_output.pt")
        with open("denoised_output.pt", "rb") as f:
            st.download_button("Click to Download", f, file_name="denoised_output.pt")
