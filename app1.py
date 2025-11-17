import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn # Import nn to define the CNN architecture
import torch.nn.functional as F # Import F for activation and pooling

# Set page config (moved to top as per Streamlit requirements)
st.set_page_config(page_title="Animal Classifier App (CNN)", layout="centered") # Using centered layout like app1

# --- Define the Simple CNN Architecture (Must match your Colab code) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Calculate the input size for the first fully connected layer
        # Assuming input image size is 224x224 after transformations:
        # After conv1 and pool1: (224/2) x (224/2) = 112 x 112, 32 channels
        # After conv2 and pool2: (112/2) x (112/2) = 56 x 56, 64 channels
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# ---------------------------------------------------------------------

# Define the same transformations used for validation/testing
def define_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load the trained CNN model
@st.cache_resource # Cache the model loading
def load_my_cnn_model(model_path):
    model = SimpleCNN(num_classes=2) # Create an instance of the SimpleCNN architecture
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu')) # Adjust map_location if using GPU
    model.load_state_dict(state_dict)
    model.eval() # Set to evaluation mode
    return model

# Load your trained CNN model file (assuming it's named 'simple_cnn_model.pth')
cnn_model_path = 'simple_cnn_model.pth' # Make sure this file is in the same directory or provide the full path
my_cnn_model = load_my_cnn_model(cnn_model_path)
my_transforms = define_transforms()
label_map_inverse = {0: 'Cow', 1: 'Buffalo'} # Inverse of your label_map


st.title("🐮🐂 Cow and Buffalo Classifier (CNN)") # Title
st.markdown("Upload an image of a cow or a buffalo to get a classification prediction.") # Description

# File upload UI
uploaded_file = st.file_uploader("📤 Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"]) # File uploader

# Prediction logic
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image.", use_column_width=True) # Display image
    st.write("")

    # Create a spinner while classifying
    with st.spinner("Classifying..."):
        # Preprocess the image
        image_tensor = my_transforms(image).unsqueeze(0) # Add batch dimension
        # No need to move to device if map_location='cpu' was used during load

        # Make prediction using the CNN model
        with torch.no_grad():
            outputs = my_cnn_model(image_tensor) # Use the CNN model
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted_class_index = torch.max(outputs.data, 1)

        predicted_class = label_map_inverse[predicted_class_index.item()]
        confidence = probabilities[0][predicted_class_index].item()

    # Display results with styling similar to app1
    st.markdown("---") # Separator
    st.subheader("🧠 Prediction Result") # Subheader
    st.success(f"**Class:** `{predicted_class}`") # Success box for class
    st.info(f"**Confidence:** `{confidence:.2f}`") # Info box for confidence

    # Optional: Display probabilities for both classes
    st.markdown("### 📊 Confidence Scores") # Markdown header for probabilities
    for i, prob in enumerate(probabilities[0]):
        class_name = label_map_inverse[i]
        st.write(f"{class_name}: `{prob.item():.2f}`") # Display probabilities with backticks

else:
    # Placeholder when no file uploaded
    st.info("Please upload an image to classify.")

# Add sidebar and footer similar to your original app.py
st.sidebar.header("About")
st.sidebar.info("This app uses a trained Simple CNN model to classify images as either a cow or a buffalo.")
st.sidebar.write("Created based on a model trained in Google Colab.")

# Footer
st.markdown("---")
