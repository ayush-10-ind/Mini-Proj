import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam, LayerAttribution
import torch.nn.functional as F
from torchvision import models

st.markdown("""
 About This System
 Detects Pneumonia from Chest X-rays
 Uses ResNet (Transfer Learning)
 Provides Explainability using Grad-CAM
""")

st.warning(" This is an AI-based prediction and should not replace medical diagnosis.")

st.title(" AI-Based Pneumonia Detection System")
st.markdown("Upload a chest X-ray to detect pneumonia using Deep Learning + Explainable AI")


# Load ResNet model

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("model.pth"))
model.eval()

# Transform

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])


# Upload image

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img = transform(image).unsqueeze(0)

    # Prediction
    with st.spinner("Analyzing X-ray..."):
        with torch.no_grad():
            output = model(img)
            _, pred = torch.max(output,1)

    classes = ["NORMAL", "PNEUMONIA"]

    # Confidence
    probs = F.softmax(output, dim=1)
    confidence = probs[0][pred.item()].item()


    # Grad-CAM
   
    gradcam = LayerGradCam(model, model.layer4)

    attr = gradcam.attribute(img, target=pred.item())

    heatmap = LayerAttribution.interpolate(attr, (224,224))
    heatmap = heatmap.squeeze().detach().numpy()

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    image_np = img.squeeze().detach().numpy().transpose(1,2,0)

    # Plot
    fig, ax = plt.subplots(1,2, figsize=(10,4))

    ax[0].imshow(image_np)
    ax[0].set_title("Original")

    ax[1].imshow(image_np)
    ax[1].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[1].set_title("Grad-CAM")

    for a in ax:
        a.axis('off')

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image")

    with col2:
        st.pyplot(fig)

    # Result
    if classes[pred.item()] == "PNEUMONIA":
        st.error(f"Prediction: {classes[pred.item()]}")
    else:
        st.success(f"Prediction: {classes[pred.item()]}")

    st.write(f"Confidence: {confidence*100:.2f}%")