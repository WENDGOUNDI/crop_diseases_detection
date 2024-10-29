# Libraries importation
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path

# Create the "data_collection_model_evaluation" if it doesn't exist
Path("./data_collection_model_evaluation").mkdir(parents=True, exist_ok=True)

# Streamlit app title
st.title("AI-Powered Early Detection of Crop Diseases in Kenyan Smallholder Farms")

# Setting sidebar 
# Sidebar with model selection
st.sidebar.header("Project Introduction")
project_info = st.sidebar.info("The goal of this porject is to implement a AI powered\
                               system achieving 95% accuracy in identifying common\
                                diseases for maize, beans, and cassava.The system should reach 90% accuracy\
                                in distinguishing between different stages of disease progression")
st.sidebar.header("Model Selection")
use_maize_model = st.sidebar.checkbox("Maize Model", value=True)  # Maize model selected by default
use_beans_model = st.sidebar.checkbox("Beans Model")
use_cassava_model = st.sidebar.checkbox("Cassava Model")

# Function for loading the trained models
@st.cache_resource
def loadModel(model_path):
    # Loading the mdoels
    loaded_model = YOLO(model_path)
    return loaded_model
    
# Function for running prediction
#@st.cache_resource
def imgPredCNN(predModel, predImg):
    # Predict with the model
    results = predModel.predict(predImg, device="cpu", save=False,  verbose=False)
    labels_names = results[0].names
    pred_cls_idx = [result.probs.top1 for result in results]
    pred_cls_idx = pred_cls_idx[0]
    predicted_label = labels_names[pred_cls_idx]
    #st.write("Using Yolov11 model")
    return predicted_label

# Function for storing image for retraining
def storeImage(saving_image, new_label, image_title):
    #cv2.imwrite(f"./data_collection_model_evaluation/{new_label}/{image_title}.png", saving_image)
    saving_image.save(f"./data_collection_model_evaluation/{new_label}/{image_title}.png")

# Load Maize Crop Trained Model
model_maize = loadModel("./prediction_models/maize_best_model.pt")
# Load Beans Crop Trained Model
model_beans = loadModel("./prediction_models/beans_best_model.pt")
# Load Cassava Crop Trained Model
model_cassava = loadModel("./prediction_models/cassava_best_model.pt")


# Image upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    uploaded_image_file_name = uploaded_image.name
    uploaded_image_file_name = uploaded_image_file_name.split(".")[0]
    # Display the uploaded image
    image = Image.open(uploaded_image)
    resize_image = image.resize((224, 224))
    st.image(resize_image, caption="Uploaded Image", use_column_width=False)

    # Inspect button
    if st.button("Inspect"):
        # Select the model based on checkbox values
        if use_maize_model and not use_beans_model and not use_cassava_model:
            prediction = imgPredCNN(model_maize, image)
            #st.write(prediction)
            st.title(f":green[Crimping Evaluation: {prediction}]")
        elif use_beans_model and not use_cassava_model and not use_maize_model:
            prediction = imgPredCNN(model_beans, image)
            #st.write(prediction)
            st.title(f":green[Crimping Evaluation: {prediction}]")
        elif use_cassava_model and not use_maize_model and not use_beans_model:
            prediction = imgPredCNN(model_cassava, image)
            #st.write(prediction)
            st.title(f":green[Crimping Evaluation: {prediction}]")
        else:
            #st.error("Please select only one model.")
            st.title(f":red[Please select only one model.]")

with st.sidebar:
    st.header("Important Links")
    "[Project Home Page](https://www.omdena.com/chapter-challenges/ai-powered-early-detection-of-crop-diseases-in-kenyan-smallholder-farms)"
    "[Omdena Collaborator](https://collaborator.omdena.com/)"
    st.header("Datasets Links")
    "[Maize Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset/)"
    "[Beans Dataset](https://github.com/AI-Lab-Makerere/ibean/)"
    "[Cassava Dataset](https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification/data/)"
    # Add an input text field
    st.header("Saving False Detection For Model Optimization")
    input_field = st.text_input(" ", placeholder="write here")
    
    if st.button("Save Image", type="primary"):
        if input_field.upper() == "OK":
            storeImage(image, input_field.upper(), uploaded_image_file_name)
            st.success("Done!")
        elif input_field.upper() == "NG":
            storeImage(image, input_field.upper(), uploaded_image_file_name)
            st.success("Done!")


