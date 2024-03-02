import streamlit as st
import numpy as np
from PIL import Image
import joblib

image_path="/content/drive/MyDrive/Group_project/Normal/008.JPG"
def extract_features(folder_path):
  import torchvision.models as models
  import os
  import shutil
  import random
  import torch
  import torchvision.transforms as transforms
  import torchvision.models as models
  from PIL import Image
  model = models.resnet50(pretrained=True)
  model.eval()
  features=[]
  input_image = Image.open(folder_path)
  transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to fit ResNet input size
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])
 
  input_tensor = transform(input_image)
  input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
  with torch.no_grad():

    features_tensor = model(input_batch)
  feature_vector = features_tensor.squeeze().numpy()
  features.append(feature_vector)
       
  return features

def model_loading(model_path):
  import joblib
  model=joblib.load(model_path)
  return model
def making_prediction(model):
  predict=model.predict(feature)
  return predict
def checking_pred(predict):
  if predict[0]==1:
    print("Anomaly")
  elif predict[0]==0:
    print("Normal")


def main():
    st.title('Image Classifier')
    st.write('Upload an image and I will classify it!')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        features=extract_features(image)
        m=model_loading("/content/drive/MyDrive/Group_project/model.pkl")
        x=making_prediction(m)
        y=checking_pred(x)

        #prediction = predict(image)
        st.write("Prediction:", y)

if __name__ == '__main__':
    main()

