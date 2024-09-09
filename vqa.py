import os
from google.oauth2 import service_account
import streamlit as st
from google.cloud import storage
from datetime import timedelta
import vertexai
from vertexai.generative_models import (
    Part,
    GenerativeModel,
    SafetySetting
)
from google.cloud.exceptions import NotFound

def local_css(file_name):
    """
    Load custom CSS for styling
    """
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

def get_google_credentials():
    google_credentials = st.secrets["google_credentials"]
    return service_account.Credentials.from_service_account_info(google_credentials)

#Google Cloud Client with the credentials from Secrets
credentials = get_google_credentials()
vertexai.init(project="genai-project-434704", location="us-central1", credentials=credentials)

def list_videos(bucket_name):
    """
    Function to list videos from Google Cloud Storage
    """
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()
    return [blob.name for blob in blobs if blob.name.endswith('.mp4')]

def generate_signed_url(bucket_name, blob_name):
    """
    Generate a signed URL for accessing the video file
    """
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(expiration=timedelta(minutes=30))  
    return url

def upload_video_to_gcs(bucket_name, video_file):
    """
    Upload a video file to Google Cloud Storage
    """
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(video_file.name)
    blob.upload_from_file(video_file)
    
    return blob.name

def analyze_video(video_uri, user_prompt, model_version):
    """
    Analyze video using Vertex AI and user prompt
    """
    video1 = Part.from_uri(mime_type="video/mp4", uri=video_uri)
    
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0,
        "top_p": 0.95,
    }

    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
    ]
    
    model = GenerativeModel(model_version)
    
    responses = model.generate_content(
        [video1, user_prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True
    )

    output = ""
    for response in responses:
        output += response.text

    return output

def main():
    st.title("Visual Question Answering System")


    if st.button("How to Guide", key="guide_button", help="Click to view instructions"):
        with st.expander("How to use this app", expanded=True):
            st.markdown("""
            ### Step-by-Step Guide:
            1. **Upload a Video** (Optional):  
                - You can upload a `.mp4` video using the "Upload Video" section below.
            2. **Select a Video**:  
                - If you've uploaded a video or want to analyze an existing one, select the video from the dropdown.
            3. **Select Model Version**:  
                - Choose between the **Light** (fast and lower-cost) or **Pro** (more powerful) version of the model for analysis.
            4. **Enter Your Analysis Prompt**:  
                - In the text area, provide a prompt for analyzing the video. 
                - For example, ask the system to identify any refueling attempts in the video.
            5. **Run Analysis**:  
                - Click the "Run Analysis" button to start the analysis.
                - The app will generate a detailed report based on the prompt.
            6. **View Output**:  
                - Once the analysis is complete, the results will appear in the text area below the video.
            """)

    bucket_name = "air-refueling-video-analysis-bucket" 
    
    st.header("Upload a Video for Analysis")
    uploaded_video = st.file_uploader("Upload a .mp4 video", type=["mp4"])
    
    if uploaded_video:
        with st.spinner("Uploading video..."):
            try:
                video_blob_name = upload_video_to_gcs(bucket_name, uploaded_video)
                st.success(f"Video '{uploaded_video.name}' uploaded successfully!")
            except NotFound:
                st.error("Error: The specified bucket was not found.")
    
    video_files = list_videos(bucket_name)

    if not video_files:
        st.warning("No videos found in the bucket.")
        return
    
    selected_video = st.selectbox("Select a video to analyze", video_files)
    
    model_version = st.selectbox("Select model version", ["Light", "Pro"])

    model_version_mapping = {
        "Light": "gemini-1.5-flash-001",
        "Pro": "gemini-1.5-pro-001"}
    selected_model_version = model_version_mapping[model_version]
    st.write(f"Selected model version: {model_version}")
    
    user_prompt = st.text_area("Enter your analysis prompt", 
                               value="Give time steps of any aircraft tries an attempt to refuel, do not leave out any attempts due to any reason? During this time layout time for each attempt whether successful or unsuccessful.")

    if selected_video:
        video_url = generate_signed_url(bucket_name, selected_video)
        st.video(video_url)

        if st.button("Run Analysis"):
            with st.spinner("Analyzing video..."):
                video_uri = f"gs://{bucket_name}/{selected_video}"
                analysis_result = analyze_video(video_uri, user_prompt, selected_model_version)
                if analysis_result:
                    st.success("Analysis complete!")
                    st.text_area("Analysis Output", analysis_result, height=300)

if __name__ == "__main__":
    main()
