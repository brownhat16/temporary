import streamlit as st
import requests
from PIL import Image
import os
import tempfile
import time

# API base URL (adjust if running remotely)
API_URL = "https://temporary-tuvr.onrender.com"

st.set_page_config(
    page_title="Financial Image Analyzer",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Financial Marketing Image Analyzer")
st.markdown("Upload an image to analyze its visual elements using AI models.")

# ------------------ SIDEBAR ------------------
st.sidebar.header("Navigation")
selection = st.sidebar.radio("Go to", ["Analyze Image", "Visualize Detections", "Compare Images"])

# ------------------ HELPERS ------------------
def upload_file(file):
    files = {"file": file.getvalue()}
    return requests.post(f"{API_URL}/analyze-image/", files=files)

def upload_focused_file(file, categories):
    files = {"file": file.getvalue()}
    data = {"categories": ",".join(categories)}
    return requests.post(f"{API_URL}/analyze-image/focused/", files=files, data=data)

def visualize_detections(file):
    files = {"file": file.getvalue()}
    return requests.post(f"{API_URL}/visualize-detections/", files=files)

def compare_images(files):
    upload_files = [("files", f.getvalue()) for f in files]
    return requests.post(f"{API_URL}/compare-images/", files=upload_files)

def display_analysis_results(results):
    for category, data in results.items():
        with st.expander(category, expanded=True):
            for key, value in data.items():
                st.write(f"**{key.replace('_', ' ').title()}**: {value}")

def display_comparison(comparison_data):
    analyses = comparison_data.get("individual_analyses", {})
    for filename, result in analyses.items():
        st.subheader(filename)
        for category, data in result.items():
            if category == "error":
                st.error(data)
                continue
            with st.expander(category, expanded=False):
                for key, value in data.items():
                    st.write(f"**{key}**: {value}")
    st.subheader("AI Summary of Differences")
    comparison_summary = comparison_data.get("comparison", {}).get("raw_content", "No summary available.")
    st.markdown(comparison_summary)

# ------------------ ANALYZE IMAGE ------------------
if selection == "Analyze Image":
    st.header("🔍 Analyze a Single Image")

    # Category selection
    all_categories = [
        "Color Palette", "Layout & Composition", "Image Type", "Elements",
        "Presence of Text", "Theme", "CTA", "Object Detection", "Character",
        "Character Role", "Context", "Offer"
    ]
    selected_categories = st.multiselect("Select Analysis Categories", options=all_categories, default=[
        "Color Palette", "Layout & Composition", "Object Detection"
    ])

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Analyze"):
            with st.spinner("Analyzing... This may take 10-20 seconds."):
                response = upload_focused_file(uploaded_file, selected_categories)
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Analysis complete!")
                    st.json(data["results"])
                    display_analysis_results(data["results"])
                else:
                    st.error("❌ Error analyzing image.")
                    st.write(response.text)

# ------------------ VISUALIZE DETECTIONS ------------------
elif selection == "Visualize Detections":
    st.header("🔎 Visualize Object Detections")

    uploaded_file = st.file_uploader("Upload an image to visualize detections", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Visualize Detections"):
            with st.spinner("Detecting objects and drawing boxes..."):
                response = visualize_detections(uploaded_file)
                if response.status_code == 200:
                    data = response.json()
                    visualization_url = f"{API_URL}{data['visualization_path']}"
                    st.image(visualization_url, caption="Detected Objects", use_column_width=True)
                else:
                    st.error("❌ Failed to visualize detections.")
                    st.write(response.text)

# ------------------ COMPARE IMAGES ------------------
elif selection == "Compare Images":
    st.header("🔄 Compare Multiple Images")

    uploaded_files = st.file_uploader("Upload at least two images to compare", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if len(uploaded_files) < 2:
        st.warning("Please upload at least two images.")
    else:
        st.write(f"Uploaded {len(uploaded_files)} images.")
        if st.button("Compare Images"):
            with st.spinner("Comparing images and generating insights..."):
                response = compare_images(uploaded_files)
                if response.status_code == 200:
                    comparison_data = response.json()
                    st.success("✅ Comparison complete!")
                    display_comparison(comparison_data)
                else:
                    st.error("❌ Error comparing images.")
                    st.write(response.text)

# ------------------ FOOTER ------------------
st.sidebar.markdown("---")
st.sidebar.info("Built with FastAPI backend and Streamlit frontend.")
