import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import requests
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Page config
# -----------------------------
st.set_page_config(
    page_title="Image Classification with PyTorch & Streamlit",
    layout="centered"
)

st.title("Image Classification Web App (ResNet18)")
st.write("Using **PyTorch ResNet-18 (pretrained on ImageNet)** + Streamlit")

# -----------------------------
# Step 2: Load labels
# -----------------------------
@st.cache_data
def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    return response.text.strip().split("\n")

# -----------------------------
# Step 3 + 4: CPU-only + Load model (eval mode)
# -----------------------------
device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    model.to(device)
    return model

labels = load_imagenet_labels()
model = load_model()

# -----------------------------
# Step 5: Define preprocessing transforms
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Step 6: File uploader UI
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image (jpg/png)",
    type=["jpg", "jpeg", "png"]
)

# Optional: show process path (Step 10)
with st.expander("Show classification process (process path)"):
    st.write("""
    **Process Path (Step-by-step)**  
    1) Upload image  
    2) Convert image to RGB  
    3) Resize → CenterCrop → Convert to tensor  
    4) Normalize using ImageNet mean/std  
    5) Feed tensor into ResNet18 model (CPU)  
    6) Apply Softmax to get probabilities  
    7) Display Top-5 predictions + bar chart  
    """)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -----------------------------
    # Step 7: Convert to tensor and inference (no gradients)
    # -----------------------------
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = F.softmax(outputs[0], dim=0)

    # -----------------------------
    # Step 8: Top-5 predictions
    # -----------------------------
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    st.subheader("Top-5 Predictions")
    for i in range(5):
        st.write(f"**{labels[int(top5_catid[i])]}** — probability: {top5_prob[i].item():.4f}")

    # -----------------------------
    # Predictions Table
    # -----------------------------
    df = pd.DataFrame({
        "Label": [labels[int(i)] for i in top5_catid.cpu().numpy()],
        "Probability": [float(p) for p in top5_prob.cpu().numpy()]
    })

    st.write("### Predictions Table")
    st.dataframe(df, use_container_width=True)

    # -----------------------------
    # Step 9: Bar chart visualization
    # -----------------------------
    st.subheader("Probability Bar Chart (Top-5)")

    labels_list = df["Label"].astype(str).tolist()
    probs_list = df["Probability"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels_list, probs_list)
    ax.set_xlabel("Probability")
    ax.set_title("Top-5 Prediction Probabilities")
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    st.pyplot(fig)

    # -----------------------------
    # Step 10: Discuss results (in-app notes)
    # -----------------------------
    st.info("""
    **Result Discussion (Step 10):**  
    - Clear, single-object images usually give higher confidence predictions.  
    - Images with multiple objects, low lighting, blur, or unusual angles may reduce accuracy.  
    - This happens because ResNet18 was trained on ImageNet classes and generalizes best to similar images.
    """)

else:
    st.info("Upload an image to start classification.")


