
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm # Assuming you have timm installed in your Streamlit environment

st.set_page_config(page_title="Animal Classifier App", layout="centered")




# Add some padding to the bottom of the main content area to prevent it from being hidden by the footer
st.markdown("""
<style>
 html, body, .main {
        height: 100%;
        margin: 0;
        padding: 0;
        background: url('data:image/webp;base64,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vh14tGdfTkFu0qz9HXJhK8QYloyK79kP/xKYagTQfVJsWBU5pluiWToZ8eX4FFS6KYoE6cW4vypgAAAAAAAAABRh2diWKBR6zaGq1YfcLsX9kg0cpIuUg6R3HAq/hn2yrnSFIcMcPzDvXfdkmvnVdmeuvG98lW3L+sMAAAAAAAAX2WoUFqFEEWQuY7dJS0ys/aZPu5FlI9pabWX8nkGyCPgPG9URlmJCRVVUJIYkyb6MpCap2j5oUOdycGRhayAAAAAAECothehDyc7xml4lmMZENnrIuS8v6a4Pz/z1Q1dYKvNv7d1pMbY1z82xO5Re2Ps/ANv1vb2bSCAAAAAA/gAFenFkklOgyHUuZwC4Fy2Eq8P6s0MWe7vLAtb2y8SirQyv1MYFrpAlXlfE78hSkbr2YIAAAAAAAAABx8euHK4kRQv6+yRrIHue6KGbhPaCXkYjZgkq5hnutNh60UB9LuXONRvwVg8C/AAAAAAAAAAAAD6FOagDUdaLX1kVSSp7e2virzqt+MTNuA6cC2O4Q5SfLJv1rnr4bEwHliGUywoAAAAAAAAAAAAdI9wu6lYbXQ/+YqYyNAFNxHazqiDqfF5sB4PnljlEL/J344TZxCfD7IaaoAAAAAAAAAHFSJiV4sG8gqEufjkm2qSywQgez6EyC1ffQXuszmNqy4xherdn4YnAMg8XwMaAAAAAAAAAAAAANocd3Okwnhm9InwRtmIZt8zxfDNGWvadlacOoqAs6gHDRfan/r9D1tjAAAAAAAAAAAAAD0FpiQpMplNP7hzUosM0uCICPCxMtv6QkYSxneMVCc4GjzzX51q9zkXztcTuwAAAAAAAAAAAAAAABTcUnN7EHajDpVZGSiEXyZO9Ol2+azAdgR6VSr3f/fcjcoggAAAAAAAAAAAAAAAAGWKC69Cx3nS0gJOKhrM7RtMy0B96UJuPnToD753q12svJbEWslvRWVAAAAAAAAAAAAAAAAAAAEVYSUa6AAAARXhpZgAASUkqAAgAAAAGABIBAwABAAAAAQAAABoBBQABAAAAVgAAABsBBQABAAAAXgAAACgBAwABAAAAAgAAABMCAwABAAAAAQAAAGmHBAABAAAAZgAAAAAAAABIAAAAAQAAAEgAAAABAAAABgAAkAcABAAAADAyMTABkQcABAAAAAECAwAAoAcABAAAADAxMDABoAMAAQAAAP//AAACoAQAAQAAAFwDAAADoAQAAQAAAK4BAAAAAAAAWE1QIIwBAAA8P3hwYWNrZXQgYmVnaW49IiIgaWQ9Ilc1TTBNcENlaGlIenJlU3pOVGN6a2M5ZCI/Pgo8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJHbyBYTVAgU0RLIDEuMCI+PHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj48cmRmOkRlc2NyaXB0aW9uIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIgcmRmOmFib3V0PSIiPjxkYzpjcmVhdG9yPjxyZGY6U2VxPjxyZGY6bGk+VmVjdG9yU3RvY2suY29tLzU4MzY3Njk3PC9yZGY6bGk+PC9yZGY6U2VxPjwvZGM6Y3JlYXRvcj48L3JkZjpEZXNjcmlwdGlvbj48L3JkZjpSREY+PC94OnhtcG1ldGE+Cjw/eHBhY2tldCBlbmQ9InciPz4=') no-repeat center center fixed;
        background-size: cover;
        min-height: 100vh;
        width: 100vw;
    }
    .block-container {
        background: transparent !important;
    }
.main {
    padding-bottom: 70px; /* Adjust this value based on your footer's height */
}
</style>
""", unsafe_allow_html=True)


# Define the model architecture (needs to match your Colab code)
def build_vit_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False) # pretrained=False initially
    num_ftrs = model.head.in_features
    model.head = torch.nn.Linear(num_ftrs, 2)
    return model

# Define the same transformations used for validation/testing
def define_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load the trained model
@st.cache_resource # Cache the model loading
def load_my_model(model_path):
    model = build_vit_model()
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu')) # Adjust map_location if using GPU
    model.load_state_dict(state_dict)
    model.eval() # Set to evaluation mode
    return model

# Load your trained model file (assuming it's named 'animal_classifier.pth')
model_path = 'animal_classifier.pth' # Make sure this file is in the same directory or provide the full path
my_model = load_my_model(model_path)
my_transforms = define_transforms()
label_map_inverse = {0: 'Cow', 1: 'Buffalo'} # Inverse of your label_map


st.title("Cow and Buffalo Classifier")

st.write("Upload an image of a cow or a buffalo to get a classification prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")

    # Create a spinner while classifying
    with st.spinner("Classifying..."):
        # Preprocess the image
        image_tensor = my_transforms(image).unsqueeze(0) # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = my_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted_class_index = torch.max(outputs.data, 1)

        predicted_class = label_map_inverse[predicted_class_index.item()]
        confidence = probabilities[0][predicted_class_index].item()

    st.subheader("🧠 Prediction Result") # Added Brain emoji
    st.success(f"**Class:** `{predicted_class}`") # Used st.success and backticks
    st.info(f"**Confidence:** `{confidence:.2f}`") # Used st.info and backticks

    # Optional: Display probabilities for both classes
    st.write("Probabilities:")
    st.markdown("### 📊 Confidence Scores") 
    for i, prob in enumerate(probabilities[0]):
        class_name = label_map_inverse[i]
        st.write(f"- {class_name}: {prob.item():.2f}")


st.sidebar.header("About")
st.sidebar.info("This app uses a trained Vision Transformer (ViT) model to classify images as either a cow or a buffalo.")
st.sidebar.write("Created based on a model trained in Google Colab.")

# Footer
st.markdown("---")
