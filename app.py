import streamlit as st
import pandas as pd
import numpy as np
from model.predict import makeInference


sideBar=st.sidebar # creating a side bar component
inputImage=sideBar.file_uploader(label="Upload a any image of Cat or Dog", type=["jpg", "png"])

if inputImage is not None:
    # Open the image using PIL
    inputImage = inputImage
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(inputImage, caption="Uploaded Image", use_column_width=True)
        modelOutput = makeInference(inputImage=inputImage)
    
    with col2:
        st.write("Model Output:")
        st.write(modelOutput)
else:
    st.write("Please choose a file")