import google.generativeai as genai
import os
import sys
import argparse
import time
from moviepy import *
from dotenv import load_dotenv


# --- Configuration ---
# Model name: 'gemini-1.5-flash' is faster and cheaper,
# 'gemini-1.5-pro' might provide higher quality results.
GEMINI_MODEL_NAME = "gemini-1.5-pro"
# Temporary audio file format
AUDIO_FORMAT = ".mp3"
# --- End Configuration ---
load_dotenv(override=True)

def configure_gemini():
    """Configures the Gemini API with the API key from environment variables."""
    api_key = os.environ['GOOGLE_API_KEY']
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please get an API key from https://aistudio.google.com/app/apikey")
        print("and set the environment variable.")
        sys.exit(1) # Exit if no API key
    try:
        genai.configure(api_key=api_key)
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        sys.exit(1)

def extract_audio(video_path, audio_path):
    """
    Extracts audio from a video file using moviepy.

    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path to save the extracted audio file.

    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    print(f"Extracting audio from '{os.path.basename(video_path)}'...")
    try:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at '{video_path}'")
            return False
        
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        if audio_clip is None:
             print(f"Error: Could not extract audio stream from '{video_path}'.")
             video_clip.close()
             return False
        audio_clip.write_audiofile(audio_path, codec='mp3') # Specify codec for mp3
        audio_clip.close()
        video_clip.close()
        print(f"Audio extracted successfully to '{os.path.basename(audio_path)}'.")
        return True
    except ImportError:
        print("Error: MoviePy library not found. Please install it: pip install moviepy")
        return False
    except Exception as e:
        # Catch potential ffmpeg errors or other issues
        print(f"Error during audio extraction: {e}")
        # Attempt to close clips if they were opened
        try:
            if 'audio_clip' in locals() and audio_clip:
                audio_clip.close()
            if 'video_clip' in locals() and video_clip:
                video_clip.close()
        except Exception:
            pass # Ignore errors during cleanup closing
        return False

def upload_audio_file(audio_path):
    """
    Uploads the audio file to Google AI Studio for processing.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        File: The uploaded file object from the Gemini API, or None on error.
    """
    print(f"Uploading '{os.path.basename(audio_path)}' to Gemini...")
    try:
        # Gemini API usually takes some time to process the upload
        audio_file = genai.upload_file(path=audio_path)
        print(f"Upload successful. File URI: {audio_file.uri}")
        # Simple polling to check if the file is ready (ACTIVE state)
        while audio_file.state.name == "PROCESSING":
            print("Waiting for file processing...")
            time.sleep(5) # Wait for 5 seconds
            audio_file = genai.get_file(audio_file.name) # Re-fetch file state
            if audio_file.state.name == "FAILED":
                 print(f"Error: File processing failed for {audio_file.name}")
                 return None
        
        if audio_file.state.name != "ACTIVE":
            print(f"Error: File {audio_file.name} is not active. Current state: {audio_file.state.name}")
            return None

        print("File processed and ready.")
        return audio_file
    except Exception as e:
        print(f"Error uploading or processing file: {e}")
        return None

def summarize_audio(audio_file_obj):
    """
    Sends the uploaded audio to Gemini for summarization.

    Args:
        audio_file_obj (File): The file object returned by genai.upload_file.

    Returns:
        str: The Markdown summary from Gemini, or None on error.
    """
    print(f"Sending audio ('{audio_file_obj.display_name}') to Gemini for summarization...")
    try:
        model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)

        # Craft the prompt
        prompt = f"""
Please analyze the following audio file, which was extracted from a video.

Your tasks are:
1.  Provide a concise summary of the main content discussed in the audio.
2.  Identify and list the key topics covered.

Present the output in Markdown format, using headings for the summary and key topics.
"""

        # Generate content using the prompt and the uploaded file
        response = model.generate_content([prompt, audio_file_obj])

        print("Summary generation complete.")
        return response.text
    except Exception as e:
        print(f"Error during Gemini summarization: {e}")
        # Check for specific API errors if possible (e.g., response.prompt_feedback)
        try:
            if response and response.prompt_feedback:
                 print(f"Prompt Feedback: {response.prompt_feedback}")
        except Exception:
            pass # Ignore errors checking feedback if response object is weird
        return None

def delete_file(file_path):
    """Deletes a file if it exists."""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Temporary file '{os.path.basename(file_path)}' deleted.")
        except OSError as e:
            print(f"Warning: Could not delete temporary file '{file_path}'. Error: {e}")
    else:
        print(f"Info: Temporary file '{os.path.basename(file_path)}' not found for deletion (might have failed creation).")


def main():
    """Main function to orchestrate the video summarization process."""
   
    global GEMINI_MODEL_NAME # Allow modification of the global model name

   
    parser = argparse.ArgumentParser(description="Summarize a video using Gemini API.")
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument(
        "-o", "--output-audio",
        help="Optional: Path to save the intermediate audio file. If not provided, a temporary file is used and deleted.",
        default=None
    )
    
    # --- Optional: Add argument to override model ---
    parser.add_argument(
        "--model",
        help=f"Specify the Gemini model to use (default: {GEMINI_MODEL_NAME})",
        default=GEMINI_MODEL_NAME,
        choices=['gemini-1.5-flash', 'gemini-1.5-pro'] # Add other compatible models if needed
    )


    args = parser.parse_args()
    
    GEMINI_MODEL_NAME = args.model
    print(f"Using Gemini model: {GEMINI_MODEL_NAME}")


    # --- Setup ---
    configure_gemini()
    video_path = args.video_path
    
    # Determine audio file path
    if args.output_audio:
        audio_path = args.output_audio
        # Ensure it has the correct extension
        if not audio_path.lower().endswith(AUDIO_FORMAT):
             audio_path += AUDIO_FORMAT
        delete_intermediate_audio = False # Don't delete if user specified path
        print(f"Will save intermediate audio to: {audio_path}")
    else:
        # Create temporary audio file path
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = f"{base_name}_temp_audio{AUDIO_FORMAT}"
        delete_intermediate_audio = True # Delete temporary file later

    audio_file_obj = None
    summary = None

    try:
        # --- Step 1: Extract Audio ---
        if not extract_audio(video_path, audio_path):
            sys.exit(1) # Exit if audio extraction fails

        # --- Step 2: Upload Audio to Gemini ---
        if not os.path.exists(audio_path):
             print(f"Error: Extracted audio file '{audio_path}' not found after extraction step.")
             sys.exit(1)
             
        audio_file_obj = upload_audio_file(audio_path)
        if not audio_file_obj:
            # Error message already printed in upload_audio_file
            sys.exit(1)

        # --- Step 3: Summarize Audio with Gemini ---
        summary = summarize_audio(audio_file_obj)
        if not summary:
            # Error message already printed in summarize_audio
            sys.exit(1)

        # --- Step 4: Print Summary ---
        print("\n" + "="*20 + " Summary " + "="*20)
        print(summary)
        print("="*50 + "\n")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    
    finally:
        # --- Step 5: Cleanup ---
        # Delete the uploaded file from Google AI storage (optional but good practice)
        if audio_file_obj:
            try:
                 print(f"Deleting uploaded file '{audio_file_obj.name}' from Gemini storage...")
                 genai.delete_file(audio_file_obj.name)
                 print("Uploaded file deleted successfully.")
            except Exception as e:
                 print(f"Warning: Could not delete uploaded file '{audio_file_obj.name}' from Gemini storage. Error: {e}")
                 
        # Delete the local temporary audio file if needed
        if delete_intermediate_audio:
            delete_file(audio_path)
        else:
             print(f"Intermediate audio file kept at: {audio_path}")


if __name__ == "__main__":
    main()