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

def analyze_video(video_uri, user_prompt):
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
    model = GenerativeModel("gemini-1.5-pro-001")
    
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

    bucket_name = "air-refueling-video-analysis-bucket"  # Your bucket name
    video_files = list_videos(bucket_name)

    if not video_files:
        st.warning("No videos found in the bucket.")
        return
    selected_video = st.selectbox("Select a video to analyze", video_files)
    user_prompt = st.text_area("Enter your analysis prompt", 
                               value="Give time steps of any aircraft tries an attempt to refuel, do not leave out any attempts due to any reason? During this time layout time for each attempt whether successful or unsuccessful.")

    if selected_video:
        video_url = generate_signed_url(bucket_name, selected_video)
        st.video(video_url)

        if st.button("Run Analysis"):
            with st.spinner("Analyzing video..."):
                video_uri = f"gs://{bucket_name}/{selected_video}"
                analysis_result = analyze_video(video_uri, user_prompt)
                if analysis_result:
                    st.success("Analysis complete!")
                    # Set the height of the text area to 300 pixels or any desired value
                    st.text_area("Analysis Output", analysis_result, height=300)

if __name__ == "__main__":
    main()
