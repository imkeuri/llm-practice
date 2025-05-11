

# --- Configuration ---
# Model name: Use Pro for potentially better aggregation and folder naming
GEMINI_MODEL_NAME = "gemini-1.5-pro"
# Temporary audio file format
AUDIO_FORMAT = ".mp3"
# Supported video extensions (lowercase)
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.mpeg', '.mpg')
# --- End Configuration ---

# Load environment variables from .env file
load_dotenv(override=True)

# --- Helper Functions ---

def sanitize_foldername(name):
    """Removes invalid characters and replaces spaces for a valid folder name."""
    # Remove characters that are not alphanumeric, underscore, or hyphen
    name = re.sub(r'[^\w\-]+', '_', name)
    # Replace multiple consecutive underscores with a single one
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Limit length (optional, e.g., to 50 chars)
    name = name[:50]
    if not name: # Handle empty name after sanitization
        name = "video_summaries"
    return name

# --- Core Functions (Mostly unchanged, but check GEMINI_MODEL_NAME usage) ---

def configure_gemini():
    """Configures the Gemini API with the API key from environment variables."""
    api_key = os.getenv("GOOGLE_API_KEY") # Use getenv for safer access
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please get an API key from https://aistudio.google.com/app/apikey")
        print("and set the environment variable in a .env file or system environment.")
        sys.exit(1)
    try:
        genai.configure(api_key=api_key)
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        sys.exit(1)

def extract_audio(video_path, audio_path):
    """Extracts audio from a video file using moviepy."""
    print(f"\n---\nExtracting audio from '{os.path.basename(video_path)}'...")
    try:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at '{video_path}'")
            return False

        # Explicitly create objects and use context manager if possible (though moviepy isn't great with it)
        video_clip = None
        audio_clip = None
        try:
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            if audio_clip is None:
                 print(f"Error: Could not extract audio stream from '{video_path}'.")
                 return False
            # Use a logger or suppress verbose output from write_audiofile if desired
            audio_clip.write_audiofile(audio_path, codec='mp3', logger=None) # Specify codec for mp3
            print(f"Audio extracted successfully to '{os.path.basename(audio_path)}'.")
            return True
        finally:
            # Ensure resources are released
             if audio_clip:
                 audio_clip.close()
             if video_clip:
                 video_clip.close()

    except ImportError:
        print("Error: MoviePy library not found or import failed. Please install it: pip install moviepy")
        return False
    except Exception as e:
        print(f"Error during audio extraction for '{os.path.basename(video_path)}': {e}")
        return False


def upload_audio_file(audio_path):
    """Uploads the audio file to Google AI Studio for processing."""
    print(f"Uploading '{os.path.basename(audio_path)}' to Gemini...")
    try:
        audio_file = genai.upload_file(path=audio_path)
        print(f"Upload successful. File URI: {audio_file.uri}. Waiting for processing...")

        # Polling to check if the file is ready (ACTIVE state)
        polling_interval = 5 # seconds
        time_limit = 300 # 5 minutes limit for processing
        start_time = time.time()
        while time.time() - start_time < time_limit:
            audio_file = genai.get_file(audio_file.name) # Re-fetch file state
            if audio_file.state.name == "ACTIVE":
                print("File processed and ready.")
                return audio_file
            elif audio_file.state.name == "FAILED":
                 print(f"Error: File processing failed for {audio_file.name}. Reason: {getattr(audio_file, 'error', 'Unknown')}")
                 # Attempt to delete the failed file
                 try:
                     genai.delete_file(audio_file.name)
                     print(f"Deleted failed uploaded file '{audio_file.name}'.")
                 except Exception as del_e:
                     print(f"Warning: Could not delete failed uploaded file '{audio_file.name}': {del_e}")
                 return None
            elif audio_file.state.name == "PROCESSING":
                print(f"Still processing... (State: {audio_file.state.name})")
                time.sleep(polling_interval)
            else: # Should not happen ideally, but good to handle (e.g., DELETED)
                print(f"Warning: File {audio_file.name} entered unexpected state: {audio_file.state.name}")
                return None # Treat unexpected states as failure

        # If loop finishes without returning, it timed out
        print(f"Error: File processing timed out after {time_limit} seconds for {audio_file.name}.")
        # Attempt to delete the timed-out file
        try:
            genai.delete_file(audio_file.name)
            print(f"Deleted timed-out uploaded file '{audio_file.name}'.")
        except Exception as del_e:
            print(f"Warning: Could not delete timed-out uploaded file '{audio_file.name}': {del_e}")
        return None

    except Exception as e:
        print(f"Error uploading or checking file '{os.path.basename(audio_path)}': {e}")
        return None

def summarize_audio(audio_file_obj, video_filename):
    """Sends the uploaded audio to Gemini for summarization."""
    global GEMINI_MODEL_NAME # Use the potentially overridden model name
    print(f"Sending '{video_filename}' audio ('{audio_file_obj.display_name}') to Gemini ({GEMINI_MODEL_NAME}) for summarization...")
    model = None # Define model outside try block for potential feedback access in except
    response = None
    try:
        model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
        prompt = f"""
Analyze the audio from the video file named '{video_filename}'.

Your tasks:
1. Provide a concise summary of the main content.
2. Identify and list the key topics or points discussed.

Present the output in Markdown format. Start with a level 3 heading (###) containing the summary, followed by a level 3 heading containing the key topics/points.
Example:
### Summary
[Your summary here]

### Key Topics
* Topic 1
* Topic 2
* ...
"""
        response = model.generate_content([prompt, audio_file_obj])
        print(f"Summary generation complete for '{video_filename}'.")
        # Basic check for empty or problematic response text
        if not response.text or len(response.text.strip()) < 10:
             print(f"Warning: Received potentially empty or invalid summary for '{video_filename}'.")
             # Provide default text to avoid issues later
             return f"### Summary\nFailed to generate summary for {video_filename}.\n\n### Key Topics\nN/A"
        return response.text

    except Exception as e:
        print(f"Error during Gemini summarization for '{video_filename}': {e}")
        try:
            # Access feedback if available on the response object even in case of error
            if response and response.prompt_feedback:
                 print(f"Prompt Feedback: {response.prompt_feedback}")
            # Check safety ratings if available
            if response and response.candidates and response.candidates[0].safety_ratings:
                print(f"Safety Ratings: {response.candidates[0].safety_ratings}")
        except Exception as fe:
            print(f"Could not retrieve feedback/safety info on error: {fe}")
        # Return placeholder text on error
        return f"### Summary\nError during summarization for {video_filename}.\n\n### Key Topics\nN/A"


def generate_aggregate_summary_and_foldername(all_summaries):
    """
    Uses Gemini to synthesize summaries and suggest a folder name.

    Args:
        all_summaries (dict): Dictionary where keys are filenames and values are Markdown summaries.

    Returns:
        tuple: (aggregate_markdown_content, suggested_folder_name) or (None, None) on error.
    """
    global GEMINI_MODEL_NAME
    if not all_summaries:
        print("No summaries to aggregate.")
        return None, None

    print("\n---\nGenerating aggregate summary and suggesting folder name...")

    # Combine individual summaries into one large text block for context
    full_context = ""
    for filename, summary in all_summaries.items():
        full_context += f"## Summary for: {filename}\n\n{summary}\n\n---\n\n"

    try:
        model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
        prompt = f"""
You have been provided with summaries and key points extracted from multiple videos. Your tasks are:

1.  **Synthesize:** Analyze all the provided summaries and key points. Create a concise, overarching summary that captures the main themes or topics discussed across *all* the videos. If there's no clear common theme, briefly state that and list the main topic of each video.
2.  **Compile Key Points:** Extract and list the most important key points from *all* the videos combined. Group similar points if appropriate.
3.  **Suggest Folder Name:** Based *only* on the common themes or the most prominent topics identified, suggest a short, descriptive, filesystem-friendly folder name (using underscores instead of spaces, alphanumeric characters preferred). The name should reflect the core subject matter.

**Output Format:**

Provide your response strictly in the following format:

FOLDER_NAME:[Your suggested folder name here]
CONTENT:
[Your overarching summary here]

### Combined Key Points
* Combined point 1
* Combined point 2
* ...

**Input Summaries:**

{full_context}
"""

        # Increase timeout for potentially longer aggregation task
        request_options = {"timeout": 600} # 10 minutes timeout for generation
        response = model.generate_content(prompt, request_options=request_options)

        # --- Parse the response ---
        response_text = response.text
        folder_name_match = re.search(r"FOLDER_NAME:(.*)", response_text)
        content_match = re.search(r"CONTENT:(.*)", response_text, re.DOTALL) # DOTALL to match across newlines

        suggested_folder_name = "aggregated_summaries" # Default
        aggregate_content = "Error: Could not parse Gemini response for aggregation." # Default

        if folder_name_match:
            suggested_folder_name = folder_name_match.group(1).strip()
            print(f"Gemini suggested folder name: '{suggested_folder_name}'")
        else:
             print("Warning: Could not parse suggested folder name from Gemini response.")

        if content_match:
            aggregate_content = content_match.group(1).strip()
            print("Aggregate summary generated.")
        else:
            print("Warning: Could not parse aggregate content from Gemini response.")
            # Fallback: Structure the individual summaries if aggregation fails
            aggregate_content = "# Video Summaries\n\n"
            for filename, summary in all_summaries.items():
                 aggregate_content += f"## {filename}\n\n{summary}\n\n---\n\n"


        return aggregate_content, suggested_folder_name

    except Exception as e:
        print(f"Error during Gemini aggregation/folder name generation: {e}")
        try:
             if response and response.prompt_feedback:
                  print(f"Prompt Feedback: {response.prompt_feedback}")
        except Exception:
             pass
        # Fallback content if the API call itself fails
        aggregate_content = "# Video Summaries (Aggregation Failed)\n\n"
        for filename, summary in all_summaries.items():
            aggregate_content += f"## {filename}\n\n{summary}\n\n---\n\n"
        return aggregate_content, "summarization_error"


def delete_file(file_path):
    """Deletes a file if it exists, with error handling."""
    if file_path and os.path.exists(file_path): # Check if path is not None or empty
        try:
            os.remove(file_path)
            print(f"Temporary file '{os.path.basename(file_path)}' deleted.")
        except OSError as e:
            print(f"Warning: Could not delete temporary file '{file_path}'. Error: {e}")
    # else: No need to print if it doesn't exist, might be intentional

def delete_uploaded_file(file_obj):
     """Deletes the file from Gemini storage if the object exists."""
     if file_obj and hasattr(file_obj, 'name'):
         try:
             print(f"Deleting uploaded file '{file_obj.name}' ({file_obj.display_name}) from Gemini storage...")
             genai.delete_file(file_obj.name)
             print("Uploaded file deleted successfully.")
         except Exception as e:
             print(f"Warning: Could not delete uploaded file '{file_obj.name}' from Gemini storage. Error: {e}")


# --- Main Execution Logic ---

def main():
    """Main function to orchestrate the video summarization process for a directory."""
    global GEMINI_MODEL_NAME # Allow modification of the global model name

    parser = argparse.ArgumentParser(description="Summarize all videos in a directory using Gemini API.")
    parser.add_argument("input_dir", help="Path to the directory containing video files.")
    # Removed --output-audio as temporary files are now managed internally per video
    parser.add_argument(
        "--model",
        help=f"Specify the Gemini model to use (default: {GEMINI_MODEL_NAME})",
        default=GEMINI_MODEL_NAME,
        choices=['gemini-1.5-flash', 'gemini-1.5-pro']
    )
    parser.add_argument(
        "--output-parent-dir",
        help="Optional: Parent directory where the new summary folder will be created. Defaults to the input directory.",
        default=None
    )


    args = parser.parse_args()

    GEMINI_MODEL_NAME = args.model # Set the global model name based on argument
    print(f"Using Gemini model: {GEMINI_MODEL_NAME}")

    input_dir = args.input_dir
    output_parent_dir = args.output_parent_dir if args.output_parent_dir else input_dir

    # --- Input Validation ---
    if not os.path.isdir(input_dir):
        print(f"Error: Input path '{input_dir}' is not a valid directory.")
        sys.exit(1)
    if not os.path.isdir(output_parent_dir):
         print(f"Error: Output parent directory '{output_parent_dir}' is not valid or accessible.")
         sys.exit(1)


    # --- Setup ---
    configure_gemini()
    all_summaries = {} # Dictionary to store {filename: summary_markdown}
    processed_files = 0
    failed_files = 0

    print(f"\nScanning directory '{input_dir}' for videos...")

    # --- Iterate and Process Videos ---
    for item in os.scandir(input_dir):
        if item.is_file() and item.name.lower().endswith(VIDEO_EXTENSIONS):
            video_path = item.path
            video_filename = item.name
            base_name = os.path.splitext(video_filename)[0]
            # Create a unique temporary audio file path in the same directory
            # Note: Consider using tempfile module for more robust temp file handling if needed
            audio_path = os.path.join(input_dir, f"temp_audio_{base_name}_{int(time.time())}{AUDIO_FORMAT}")
            audio_file_obj = None # Reset for each video

            try:
                print(f"\nProcessing video: {video_filename}")

                # 1. Extract Audio
                if not extract_audio(video_path, audio_path):
                    raise ValueError(f"Audio extraction failed for {video_filename}") # Raise error to trigger finally

                # 2. Upload Audio
                audio_file_obj = upload_audio_file(audio_path)
                if not audio_file_obj:
                    raise ValueError(f"Audio upload/processing failed for {video_filename}")

                # 3. Summarize Audio
                summary = summarize_audio(audio_file_obj, video_filename)
                if summary: # Even if summary indicates an error, store it
                    all_summaries[video_filename] = summary
                    print(f"Summary stored for {video_filename}")
                    processed_files += 1
                else:
                     # Should not happen with current summarize_audio error handling, but as fallback:
                     all_summaries[video_filename] = f"### Summary\nUnknown error generating summary for {video_filename}.\n\n### Key Topics\nN/A"
                     print(f"Stored placeholder summary due to unknown error for {video_filename}")
                     failed_files += 1 # Count as failed if summary is None/empty despite handling


            except Exception as e:
                print(f"--- Failed to process video: {video_filename}. Reason: {e} ---")
                failed_files += 1
                # Ensure placeholder exists if error happened before summary generation
                if video_filename not in all_summaries:
                    all_summaries[video_filename] = f"### Summary\nFailed to process video {video_filename} due to error: {e}\n\n### Key Topics\nN/A"

            finally:
                # 4. Cleanup (for the current video)
                delete_file(audio_path) # Delete local temp audio
                delete_uploaded_file(audio_file_obj) # Delete file from Gemini storage

    print(f"\n---\nProcessed {processed_files} video(s), failed to process {failed_files} video(s).")

    if not all_summaries:
        print("No video summaries were generated. Exiting.")
        sys.exit(0)

    # --- Aggregation and Output ---
    aggregate_content, suggested_folder_name = generate_aggregate_summary_and_foldername(all_summaries)

    if aggregate_content and suggested_folder_name:
        # Sanitize and create output folder
        output_folder_name = sanitize_foldername(suggested_folder_name)
        output_dir_path = os.path.join(output_parent_dir, output_folder_name)
        try:
            os.makedirs(output_dir_path, exist_ok=True) # exist_ok=True prevents error if folder exists
            print(f"\nOutput folder created/ensured at: '{output_dir_path}'")

            # Write the aggregated content to a Markdown file
            output_md_filename = f"{output_folder_name}_summary.md"
            output_md_path = os.path.join(output_dir_path, output_md_filename)

            with open(output_md_path, 'w', encoding='utf-8') as f:
                f.write(aggregate_content)
            print(f"Aggregated summary saved to: '{output_md_path}'")

        except OSError as e:
            print(f"Error creating output directory or writing file: {e}")
            # As a fallback, print the aggregate content to console
            print("\n--- Aggregated Summary (Could not write to file) ---")
            print(aggregate_content)
            print("--- End of Aggregated Summary ---")
    else:
        print("Could not generate aggregate summary or folder name. No output file created.")
        # Optionally print the individual summaries collected if aggregation failed severely
        # print("\n--- Individual Summaries ---")
        # for fname, summ in all_summaries.items():
        #     print(f"## {fname}\n{summ}\n---")


if __name__ == "__main__":
    main()