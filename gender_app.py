import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import urllib.request
import os
from collections import deque

# 1. Streamlit Page Config
st.set_page_config(page_title="Gender Classification", page_icon="🧑‍🤝‍🧑")
st.title("Live Gender Classification with Deep Learning")
st.write("Real-time face detection and gender recognition system using Deep Learning technique. Built with PyTorch, and OpenCV. Powered by Streamlit for an interactive experience. Please ensure you have a webcam connected and the required model file in the same directory. The model OpenCV Face Detection Model are from OpenCV's GitHub repository (DNN-based face detector model files), and the gender classification model is a MobileNetV2 trained on a dataset of men and women. The dataset is from saadpd Kaggle's menwomen-classification.")

# 2. Load Models with Caching
@st.cache_resource
def load_models():
    # Load OpenCV DNN model for face detection
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    if not os.path.exists("deploy.prototxt"):
        urllib.request.urlretrieve(prototxt_url, "deploy.prototxt")
    if not os.path.exists("res10_300x300_ssd_iter_140000.caffemodel"):
        urllib.request.urlretrieve(caffemodel_url, "res10_300x300_ssd_iter_140000.caffemodel")
        
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

    # Load model for gender classification
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.mobilenet_v2(weights=None) 
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    
    try:
        model.load_state_dict(torch.load('men_women_mobilenet.pth', map_location=device))
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ 'men_women_mobilenet.pth' กรุณาตรวจสอบโฟลเดอร์")
        
    model.to(device)
    model.eval()
    
    return net, model, device

net, gender_model, device = load_models()

# 3. Data Transform
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
classes = ['Men', 'Women']
history_length = 7

# 4. UI Control
run_camera = st.checkbox("Open Webcam")
frame_window = st.image([]) # showing video feed in Streamlit

# 5. Webcam
if run_camera:
    # Seperate columns for video and info
    col1, col2 = st.columns([3, 1])

    with col1:
        frame_window = st.image([]) # Video

    with col2:
        st.markdown("### Prediction Info")
        info_placeholder = st.empty() # Prediction Info

    cap = cv2.VideoCapture(0)
    pred_history = deque(maxlen=history_length)
    
    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Can't access webcam. Please check your camera permissions.")
            break
            
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # OpenCV DNN detect faces
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        face_detected = False

        # Loop to process confidence more than 60%
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                face_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # box padding
                pad_w = int((endX - startX) * 0.1)
                pad_h = int((endY - startY) * 0.2)
                startX, startY = max(0, startX - pad_w), max(0, startY - pad_h)
                endX, endY = min(w, endX + pad_w), min(h, endY + pad_h)

                face_roi = frame[startY:endY, startX:endX]
                if face_roi.size == 0: continue

                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(face_rgb)
                input_tensor = data_transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = gender_model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    men_prob = probs[0][0].item() 
                    pred_history.append(men_prob)

                # Average
                avg_men_prob = sum(pred_history) / len(pred_history)

                if avg_men_prob > 0.5:
                    label, conf_val, color = 'Men', avg_men_prob * 100, (0, 0, 255) # BGR สำหรับ OpenCV
                else:
                    label, conf_val, color = 'Women', (1 - avg_men_prob) * 100, (255, 20, 147)
            
                if conf_val < 60.0:
                    label_text, color = f"Unknown ({conf_val:.0f}%)", (100, 100, 100)
                    current_label = "Unknown"
                else:
                    label_text = f"{label} {conf_val:.1f}%"
                    current_label = label
                current_conf = conf_val

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, label_text, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if not face_detected:
            pred_history.clear()
            current_label = "-"
            current_conf = 0.0

        # change color space from BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)
        
        if face_detected:
            # HTML/Markdown to format the prediction info
            info_placeholder.markdown(f"""
            #### Gender: {current_label}  
            #### Confidence: {current_conf:.2f}%
            """)
        else:
            info_placeholder.markdown(f"""
            #### Gender: No face detected  
            #### Confidence: 0.00%
            """)
            
    cap.release()
else:
    st.info("Please check the 'Open Webcam' checkbox to start the application. You need to move your face closer to the webcam for better detection. If you have any issues, please ensure your webcam is properly connected and permissions are granted.")