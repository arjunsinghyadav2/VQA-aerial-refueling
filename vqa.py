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

# Load custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply the CSS
local_css("style.css")

def get_google_credentials():
    google_credentials = st.secrets["google_credentials"]
    return service_account.Credentials.from_service_account_info(google_credentials)

# Google Cloud Client with the credentials from Secrets
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
    
    # Use the file's name as the blob name in GCS
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
    
    model = GenerativeModel(model_version)  # Use the selected model version
    
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
    # Title Section
    st.title("Visual Question Answering System")
    st.markdown("Use AI to analyze aerial refueling videos and extract meaningful insights.")

    # Layout: Split the screen into two columns for file upload and video selection
    st.header("Step 1: Upload or Select a Video")

    # Use two columns for video upload and selection
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload a Video")
        uploaded_video = st.file_uploader("Upload a .mp4 video", type=["mp4"])

        if uploaded_video:
            with st.spinner("Uploading video..."):
                try:
                    video_blob_name = upload_video_to_gcs("air-refueling-video-analysis-bucket", uploaded_video)
                    st.success(f"Video '{uploaded_video.name}' uploaded successfully!")
                except NotFound:
                    st.error("Error: The specified bucket was not found.")
    
    with col2:
        st.subheader("Select an Existing Video")
        video_files = list_videos("air-refueling-video-analysis-bucket")

        if not video_files:
            st.warning("No videos found in the bucket.")
            return

        selected_video = st.selectbox("Select a video to analyze", video_files)
    
    # Model and Prompt Section
    st.header("Step 2: Choose Model Version and Enter Prompt")
    
    # Two columns for model selection and user prompt
    col1, col2 = st.columns([1, 2])

    with col1:
        model_version = st.selectbox("Select Model Version", ["Light", "Pro"])
        model_version_mapping = {
            "Light": "gemini-1.5-flash-001",
            "Pro": "gemini-1.5-pro-001"
        }
        selected_model_version = model_version_mapping[model_version]

    with col2:
        user_prompt = st.text_area("Enter your analysis prompt", 
                                   value="Give time steps of any aircraft tries an attempt to refuel, do not leave out any attempts due to any reason? During this time layout time for each attempt whether successful or unsuccessful.")

    # Display the selected video
    if selected_video:
        st.header("Step 3: Preview Video and Run Analysis")
        video_url = generate_signed_url("air-refueling-video-analysis-bucket", selected_video)
        st.video(video_url)

        if st.button("Run Analysis"):
            with st.spinner("Analyzing video..."):
                video_uri = f"gs://air-refueling-video-analysis-bucket/{selected_video}"
                analysis_result = analyze_video(video_uri, user_prompt, selected_model_version)
                if analysis_result:
                    st.success("Analysis complete!")
                    st.text_area("Analysis Output", analysis_result, height=300)

    # Add a helpful How to Guide section at the bottom
    with st.expander("How to use this app", expanded=False):
        st.markdown("""
        ### Step-by-Step Guide:
        1. **Upload or Select a Video**: Upload a video or choose from existing videos.
        2. **Select a Model Version**: Choose between the light or pro version depending on your needs.
        3. **Enter an Analysis Prompt**: Provide a custom prompt for the AI to analyze.
        4. **Run Analysis**: Click the button to run the analysis and review the output.
        """)

if __name__ == "__main__":
    main()
