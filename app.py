import streamlit as st
import torch
from torchvision import models, transforms, datasets
from PIL import Image
from torch.utils.data import DataLoader

import os

# ==== Load model ====
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("resnet_fracture_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ==== Define transform (same as training) ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.300, 0.300, 0.300],
                         [0.200, 0.200, 0.225])
])

class_names = ['fractured', 'not_fractured']

# ==== Streamlit UI ====
st.title("🩻 Bone Fracture Detection")
st.write("Upload an X-ray image to detect if it's **fractured** or **not fractured**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    image_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)  # Calculate softmax probabilities
        _, pred = torch.max(output, 1)                      # Get predicted class index
        confidence = probs[0][pred.item()].item() * 100     # Confidence for predicted class
        prediction = class_names[pred.item()]               # Convert index to class name

    st.markdown(f"### 🔍 Prediction: `{prediction}`")
    st.markdown(f"### 📌 Confidence: `{confidence:.2f}%`")



