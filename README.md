# AI-Powered Early Detection of Crop Diseases in Kenyan Smallholder Farms

This project demonstrates the feasibility of using AI for early and accurate detection of crop diseases in Kenyan smallholder farms, focusing on maize, beans, and cassava crops.

**Project Goal:** To develop an AI-powered system with high accuracy (95%) in identifying common diseases for maize, beans, and cassava, and 90% accuracy in distinguishing between different stages of disease progression.

**Proof of Concept (PoC):** This repository contains the code for a PoC demonstrating the feasibility of the project. The PoC includes trained models for maize, beans, and cassava, capable of identifying common diseases affecting these crops.

## Key Features

* **Trained Models:**
    * **Maize Model:** Detects 3 common maize diseases.
    * **Beans Model:** Detects 2 common bean diseases.
    * **Cassava Model:** Detects 4 common cassava diseases.
* **Lightweight Models:** Each model is around 3MB, making them suitable for mobile applications.
* **High Accuracy:** Achieves satisfying accuracy in disease detection, with potential for further improvement.
* **User-Friendly Interface:**  A Streamlit webapp and a Kivy mobile application provide an easy-to-use interface for interacting with the models.

## Main Libraries Used

* **Ultralytics YOLO:**  Accessing Yolov1n pretrained classification pretrained weights used for training.
* **Streamlit:**  For building the interactive web application.
* **Kivy:**  For building mobile application.
* **OpenCV:** For image processing.
* **PIL:** For image manipulation.

## Dataset Information

* **Maize Dataset:** [https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset/] - 4 classes (blight, common rust, gray leaf spot, healthy)
* **Beans Dataset:** [https://github.com/AI-Lab-Makerere/ibean/] - 3 classes (angular keaf sport, bean rust, healthy)
* **Cassava Dataset:** [https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification/data] - 5 classes (bateria blight, brown streak disease, green mottle, moisaic disease, healthy)

**Note:** Some datasets exhibit class imbalance, which can be addressed in future improvements.

## How to Run the PoC

1. **Clone the repository:** `git clone https://github.com/WENDGOUNDI/crop_diseases_detection.git`
2. **Install dependencies:** `pip install -r requirements.txt`.
3. **Run the Streamlit app:** `app_crop_deploy.py`
4. **Run the mobile app:** `get_started_kivy_app.py`

**Note:** You may need to adjust some directories paths. In addition, to minimize potential library version conflicts, create a new virtual environment and then run pip install -r requirements.txt within that environment.

## Future Improvements

* **Address class imbalance:** Implement techniques like data augmentation or weighted loss functions.
* **Improve accuracy:** Fine-tune hyperparameters, explore different YOLO architectures, CNN models...
* **Expand disease coverage:**  Train models to detect a wider range of diseases.
* **Develop a mobile application:** Implement a cross platform mobile app with offline capabilities to ease access / usage by farmers.

## Found this helpful?

If you found this project helpful or interesting, please consider giving it a star! ⭐ 
