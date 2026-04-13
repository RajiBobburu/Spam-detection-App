# Spam Detection ML App
A machine learning-based web application that classifies text messages as **Spam** or **Not Spam (Ham)** using TF-IDF and LinearSVC.
---
## Overview
This project implements an end-to-end **Spam Detection System** that:
- Processes text data using TF-IDF vectorization  
- Trains a Linear Support Vector Machine (LinearSVC)  
- Deploys the model using a Streamlit web interface  
The application allows users to input messages and instantly check whether they are spam or not.
---
##  Key Features
- TF-IDF based text vectorization  
- Machine Learning model (LinearSVC)  
- Real-time prediction using Streamlit  
- High accuracy (~97–98%)  
- Session-based prediction history  
- Modular code structure (separation of concerns)  
- Model persistence using joblib  
---
## Architecture
- User Input → Text Preprocessing → TF-IDF Vectorization → LinearSVC Model → Prediction (Spam / Ham)
---
## Tech Stack
- **Python**
- **Scikit-learn**
- **Streamlit**
- **Pandas**
- **Joblib**
---
## How to Run
### 1️. Clone Repository
### 2️. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️. Train Model
```bash
python train_model.py
```
### 4️.Run Application
```bash
streamlit run app.py
```
---
## Sample Test Messages
### Spam Examples
- "Congratulations! You won a free gift card!"
- "URGENT! Click here to claim your prize"
- "Win cash now!!! Limited offer"
### Normal Messages
- "Hey, are we meeting today?"
- "Please send the report by evening"
- "Call me when you're free"
---
## How It Works
- Text data is cleaned and processed
- TF-IDF converts text into numerical features
- Model predicts spam or ham based on patterns
- Result is displayed instantly in UI
---
## Use Cases
- SMS spam detection
- Email filtering
- Fraud detection
- Content moderation
---
## Demo : 
<img width="1820" height="906" alt="image" src="https://github.com/user-attachments/assets/ad585f7e-4623-4f70-b002-f589b7142b9a" />



Author : Raji Bobburu








