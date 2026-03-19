import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam, LayerAttribution

st.title("Pneumonia Detection with Explainable AI")

# Model
class BetterCNN(nn.Module):
    def __init__(self):
        super(BetterCNN, self).__init__()

        self.conv1 = nn.Conv2d(1,16,3)
        self.conv2 = nn.Conv2d(16,32,3)

        self.pool = nn.MaxPool2d(2,2)

        self.fc = nn.Linear(32*54*54,2)

    def forward(self,x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x
# Load model
model = BetterCNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Upload image
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")  # FORCE grayscale

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output,1)

    classes = ["NORMAL", "PNEUMONIA"]
    st.subheader(f"Prediction: {classes[pred.item()]}")

    # -------- Grad-CAM --------
    gradcam = LayerGradCam(model, model.conv2)

    attr = gradcam.attribute(img, target=pred.item())

    heatmap = LayerAttribution.interpolate(attr, (224,224))
    heatmap = heatmap.squeeze().detach().numpy()

    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    image_np = img.squeeze().detach().numpy()

    # Plot
    fig, ax = plt.subplots(1,2, figsize=(10,4))

    ax[0].imshow(image_np, cmap='gray')
    ax[0].set_title("Original")

    ax[1].imshow(image_np, cmap='gray')
    ax[1].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[1].set_title("Grad-CAM")

    for a in ax:
        a.axis('off')

    st.pyplot(fig)