import vertexai
import streamlit as st
import os
from vertexai.preview.generative_models import (GenerativeModel,
 GenerationConfig
)
from dotenv import load_dotenv

load_dotenv()
# "Authentication" to Google Cloud
PROJECT_ID = "polar-cargo-472321-f5"
vertexai.init(project=PROJECT_ID, location="us-central1")

#Load the model
model = GenerativeModel("gemini-2.0-flash-001")

def user_interfaces():
    st.title("Vertex AI experiement")
    st.header("Gemini 1.0 Pro with vertexai")
    
    user_input = st.text_input("Ask anything ...")
    if user_input:
        response = model.generate_content(user_input, stream=True)
        for result in response:
            st.write(result.text, end="")
            
if __name__ == "__main__":
    user_interfaces()

