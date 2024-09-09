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
import time

# Load custom CSS for styling
def local_css():
    st.markdown("""
        <style>
        /* Set background color to dark navy blue */
        body {
            background-color: #001f3f;
        }

        /* Set text color to make it readable on a dark background */
        .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader {
            color: #ffffff;
        }

        /* Customize button appearance */
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 5px;
        }

        /* Customize text input and text area */
        textarea, input {
            background-color: #001f3f;
            color: white;
            border: 1px solid #007bff;
        }

        /* Customizing select boxes */
        select {
            background-color: #001f3f;
            color: white;
        }
        
        /* Customizing video frame */
        .stVideo {
            border: 2px solid #007bff;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

# Apply the CSS
local_css()

def get_google_credentials():
    google_credentials = st.secrets["google_credentials"]
    return service_account.Credentials.from_service_account_info(google_credentials)

# Google Cloud Client with the credentials from Secrets
credentials = get_google_credentials()
storage_client = storage.Client(credentials=credentials)
vertexai.init(project="genai-project-434704", location="us-central1", credentials=credentials)

def list_videos(bucket_name):
    """
    Function to list videos from Google Cloud Storage
    """
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs()
    return [blob.name for blob in blobs if blob.name.endswith('.mp4')]

def generate_signed_url(bucket_name, blob_name):
    """
    Generate a signed URL for accessing the video file
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(expiration=timedelta(minutes=30)) 
    return url

def upload_video_to_gcs(bucket_name, video_file):
    """
    Upload a video file to Google Cloud Storage
    """
    bucket = storage_client.bucket(bucket_name)
    
    try:
        # Create a new blob with a unique name to avoid overwriting
        blob_name = f"{int(time.time())}_{video_file.name}" 
        blob = bucket.blob(blob_name)
        blob.upload_from_file(video_file)

        st.success(f"Video '{video_file.name}' uploaded successfully as '{blob_name}'!")
        return blob_name
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None

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
    st.markdown("<h1 style='text-align: center; font-size: 36px;'>Visual Question Answering System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Use AI to analyze aerial refueling videos and extract meaningful insights.</p>", unsafe_allow_html=True)

    bucket_name = "air-refueling-video-analysis-bucket"  # Make sure this is the correct bucket name

    # Step 1: Upload or Select a Video
    st.markdown("<h2 style='font-size: 24px;'>Step 1: Upload or Select a Video</h2>", unsafe_allow_html=True)

    # Upload and Video Preview - Video preview under upload button
    selected_video = st.selectbox("Select a video to analyze", list_videos(bucket_name))

    # Add a button to allow video upload
    uploaded_video = st.file_uploader("Upload a .mp4 video", type=["mp4"])

    if uploaded_video is not None:
        if st.button("Upload Selected Video"):
            with st.spinner("Uploading video..."):
                uploaded_blob_name = upload_video_to_gcs(bucket_name, uploaded_video)
                if uploaded_blob_name:
                    # Update the selectbox options after upload
                    all_videos = list_videos(bucket_name)
                    st.session_state.video_options = all_videos
                    st.success(f"Video uploaded successfully and saved as '{uploaded_blob_name}'")
                else:
                    st.error("Upload Failed")
    
    # Display the selected video underneath the upload option
    if selected_video:
        video_url = generate_signed_url(bucket_name, selected_video)
        st.video(video_url)

    # Step 2: Model and Prompt
    st.markdown("<h2 style='font-size: 24px;'>Step 2: Choose Model Version and Enter Prompt</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

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

    # "Run Analysis" button right under the text area
    if st.button("Run Analysis"):
        with st.spinner("Analyzing video..."):
            video_uri = f"gs://{bucket_name}/{selected_video}"
            analysis_result = analyze_video(video_uri, user_prompt, selected_model_version)
            if analysis_result:
                st.success("Analysis complete!")
                st.text_area("Analysis Output", analysis_result, height=300)

    # Expander for "How to use this app"
    with st.expander("How to use this app", expanded=False):
        st.markdown("""
        ### Step-by-Step Guide:
        1. **Select a Video**: Choose from existing videos or upload a new one.
        2. **Select a Model Version**: Choose between the light or pro version depending on your needs.
        3. **Enter an Analysis Prompt**: Provide a custom prompt for the AI to analyze.
        4. **Run Analysis**: Click the button to run the analysis and review the output.
        """)

if __name__ == "__main__":
    main()
