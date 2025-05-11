import ollama # New import for Ollama
from faster_whisper import WhisperModel # New import for STT
import os
import sys
import argparse
import time
import re
import torch # Needed by faster-whisper

# from moviepy.editor import VideoFileClip # Use specific import below
from moviepy.video.io.VideoFileClip import VideoFileClip
from dotenv import load_dotenv

# --- Configuration ---
# --- Ollama Configuration ---
OLLAMA_MODEL_NAME = "llama3.3" # The model name you pulled in Ollama (e.g., 'llama3', 'llama3:70b')
OLLAMA_HOST = "http://localhost:11434" # Default Ollama API endpoint

# --- Whisper Configuration ---
# Options: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"
# Smaller models are faster but less accurate.
WHISPER_MODEL_SIZE = "tiny"
# Use "cuda" if you have a compatible NVIDIA GPU and CUDA installed, otherwise "cpu"
# Check GPU memory requirements for larger models
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Compute type, e.g., "float16" for GPU, "int8" for CPU (check faster-whisper docs)
WHISPER_COMPUTE_TYPE = "float16" if WHISPER_DEVICE == "cuda" else "int8"

# --- General Configuration ---
AUDIO_FORMAT = ".mp3"
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.mpeg', '.mpg')
# --- End Configuration ---

# Load environment variables from .env file (e.g., for optional API keys if needed later)
load_dotenv(override=True)

# --- Global variable for Whisper model (load once) ---
whisper_model = None

def load_whisper_model():
    """Loads the Faster Whisper model into memory."""
    global whisper_model
    if whisper_model is None:
        print(f"Loading Whisper model '{WHISPER_MODEL_SIZE}' on device '{WHISPER_DEVICE}' with compute type '{WHISPER_COMPUTE_TYPE}'...")
        try:
            whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
            print("Whisper model loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("Please ensure faster-whisper and its dependencies (like torch) are installed correctly.")
            print("Also check model name, device, and compute type compatibility.")
            sys.exit(1)
    return whisper_model

# --- Helper Functions (sanitize_foldername is unchanged) ---
def sanitize_foldername(name):
    """Removes invalid characters and replaces spaces for a valid folder name."""
    name = re.sub(r'[^\w\-]+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    name = name[:50]
    if not name:
        name = "video_summaries"
    return name

# --- Core Processing Functions ---

def extract_audio(video_path, audio_path):
    """Extracts audio from a video file using moviepy."""
    print(f"\n---\nExtracting audio from '{os.path.basename(video_path)}'...")
    try:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at '{video_path}'")
            return False
        video_clip = None
        audio_clip = None
        try:
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            if audio_clip is None:
                 print(f"Error: Could not extract audio stream from '{video_path}'.")
                 return False
            audio_clip.write_audiofile(audio_path, codec='mp3', logger=None)
            print(f"Audio extracted successfully to '{os.path.basename(audio_path)}'.")
            return True
        finally:
             if audio_clip: audio_clip.close()
             if video_clip: video_clip.close()
    except ImportError:
        print("Error: MoviePy library not found or import failed. Please install it: pip install moviepy")
        return False
    except Exception as e:
        print(f"Error during audio extraction for '{os.path.basename(video_path)}': {e}")
        return False

def transcribe_audio(audio_path):
    """Transcribes audio to text using Faster Whisper."""
    global whisper_model
    if whisper_model is None:
        load_whisper_model() # Load if not already loaded
        if whisper_model is None: return None # Exit if loading failed

    print(f"Transcribing '{os.path.basename(audio_path)}' using Whisper '{WHISPER_MODEL_SIZE}'...")
    try:
        # beam_size=5 is a standard parameter for Whisper transcription
        segments, info = whisper_model.transcribe(audio_path, beam_size=5)
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

        full_transcript = ""
        # Process segments generator
        for segment in segments:
            # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)) # Optional: print segments
            full_transcript += segment.text + " "

        transcript_length = len(full_transcript.split())
        print(f"Transcription complete ({transcript_length} words).")
        if transcript_length == 0:
             print("Warning: Transcription resulted in empty text.")
             return "" # Return empty string, not None

        return full_transcript.strip()

    except Exception as e:
        print(f"Error during audio transcription: {e}")
        return None


def summarize_text_with_llama3(text, video_filename):
    """Sends text to Ollama Llama 3 for summarization."""
    if not text:
        print(f"Skipping summarization for '{video_filename}' due to empty transcript.")
        return f"### Summary\nTranscription resulted in empty text for {video_filename}.\n\n### Key Topics\nN/A"

    print(f"Sending transcript of '{video_filename}' to Ollama ({OLLAMA_MODEL_NAME}) for summarization...")

    prompt = f"""
You are an expert summarization assistant. Analyze the following text, which is a transcript from the video file named '{video_filename}'.

Your tasks:
1. Provide a concise summary of the main content.
2. Identify and list the key topics or points discussed
3. Divve into the details of the content, including any notable quotes or statistics.
4. Search for any specific themes or patterns in the text.
5. Is there any extra information that stands out or is particularly relevant to the video's subject matter?

Present the output in Markdown format. Start with a level 3 heading (###) containing the summary, followed by a level 3 heading containing the key topics/points. Respond ONLY with the Markdown summary and key topics as requested.

Transcript:
---
{text}
---
"""
    try:
        # Use the Ollama chat endpoint for better instruction following
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.5} # Adjust temperature for creativity vs determinism
        )
        summary_content = response['message']['content']
        print(f"Summarization complete for '{video_filename}'.")
         # Basic check for empty or problematic response text
        if not summary_content or len(summary_content.strip()) < 10:
             print(f"Warning: Received potentially empty or invalid summary for '{video_filename}'.")
             return f"### Summary\nFailed to generate summary for {video_filename} (possibly empty response from Ollama).\n\n### Key Topics\nN/A"
        return summary_content.strip()

    except Exception as e:
        print(f"Error communicating with Ollama or summarizing '{video_filename}': {e}")
        print("Please ensure the Ollama server is running and the model is available.")
        return f"### Summary\nError during summarization for {video_filename} (Ollama communication failed: {e}).\n\n### Key Topics\nN/A"


def generate_aggregate_summary_and_foldername(all_summaries):
    """Uses Ollama Llama 3 to synthesize summaries and suggest a folder name."""
    if not all_summaries:
        print("No summaries to aggregate.")
        return None, None

    print("\n---\nGenerating aggregate summary and suggesting folder name using Ollama...")

    full_context = ""
    for filename, summary in all_summaries.items():
        # Only include if summary seems valid (basic check)
        if summary and "failed" not in summary.lower() and "error" not in summary.lower():
             full_context += f"## Summary for: {filename}\n\n{summary}\n\n---\n\n"
        else:
             full_context += f"## Processing Failed for: {filename}\n\n{summary}\n\n---\n\n"

    if not full_context:
        print("No valid individual summaries found to aggregate.")
        return "# Video Summaries\n\nNo valid summaries were generated.", "aggregation_failed"


    prompt = f"""
You are an expert analysis assistant. You have been provided with summaries (and potentially error messages) extracted from multiple videos. Your tasks are:

1.  **Synthesize:** Analyze all the provided summaries. Create a concise, overarching summary that captures the main themes or topics discussed across *all* the videos with successful summaries. If there's no clear common theme, briefly state that and list the main topic of each successfully summarized video. Mention any videos that failed processing if applicable.
2.  **Compile Key Points:** Extract and list the most important key points from *all* the successfully generated summaries combined. Group similar points if appropriate. Do not forget to include any notable quotes or statistics that were highlighted in the summaries and are relevant to the overall context.
3.  **Suggest Folder Name:** Based *only* on the common themes or the most prominent topics identified in the successful summaries, suggest a short, descriptive, filesystem-friendly folder name (using underscores, alphanumeric characters). The name should reflect the core subject matter.

**Output Format:**

Provide your response strictly in the following format, with NO other text before or after:

FOLDER_NAME:[Your suggested folder name here]
CONTENT:
[Your overarching summary here]

### Combined Key Points
* Combined point 1
* Combined point 2
* ...

**Input Summaries/Logs:**

{full_context}
"""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.5}
        )
        response_text = response['message']['content']

        # --- Parse the response ---
        folder_name_match = re.search(r"FOLDER_NAME:(.*)", response_text)
        content_match = re.search(r"CONTENT:(.*)", response_text, re.DOTALL)

        suggested_folder_name = "aggregated_summaries"
        aggregate_content = "Error: Could not parse Ollama response for aggregation."

        if folder_name_match:
            suggested_folder_name = folder_name_match.group(1).strip()
            print(f"Ollama suggested folder name: '{suggested_folder_name}'")
        else:
            print("Warning: Could not parse suggested folder name from Ollama response.")

        if content_match:
            aggregate_content = content_match.group(1).strip()
            print("Aggregate summary generated.")
        else:
            print("Warning: Could not parse aggregate content from Ollama response.")
            # Fallback: Structure the individual summaries if aggregation fails
            aggregate_content = "# Video Summaries (Aggregation Parsing Failed)\n\n"
            for filename, summary in all_summaries.items():
                 aggregate_content += f"## {filename}\n\n{summary}\n\n---\n\n"

        return aggregate_content, suggested_folder_name

    except Exception as e:
        print(f"Error during Ollama aggregation/folder name generation: {e}")
        # Fallback content if the API call itself fails
        aggregate_content = "# Video Summaries (Aggregation Failed)\n\n"
        for filename, summary in all_summaries.items():
            aggregate_content += f"## {filename}\n\n{summary}\n\n---\n\n"
        return aggregate_content, "summarization_error"


def delete_file(file_path):
    """Deletes a file if it exists, with error handling."""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Temporary file '{os.path.basename(file_path)}' deleted.")
        except OSError as e:
            print(f"Warning: Could not delete temporary file '{file_path}'. Error: {e}")

# --- Main Execution Logic ---

def main():
    """Main function to orchestrate the video summarization process using local STT and LLM."""
    # Use global configurations set at the top
    global OLLAMA_MODEL_NAME, WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

    parser = argparse.ArgumentParser(description="Summarize videos in a directory using local Whisper STT and Ollama Llama 3.")
    parser.add_argument("input_dir", help="Path to the directory containing video files.")
    parser.add_argument(
        "--ollama-model",
        help=f"Specify the Ollama model name to use (default: {OLLAMA_MODEL_NAME})",
        default=OLLAMA_MODEL_NAME,
    )
    parser.add_argument(
        "--whisper-model",
        help=f"Specify the Whisper model size (default: {WHISPER_MODEL_SIZE})",
        default=WHISPER_MODEL_SIZE,
        choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
    )
    # Add device and compute type selection if desired, otherwise use auto-detection/defaults
    parser.add_argument(
        "--output-parent-dir",
        help="Optional: Parent directory for the new summary folder. Defaults to the input directory.",
        default=None
    )

    args = parser.parse_args()

    # Override defaults if provided via command line
    OLLAMA_MODEL_NAME = args.ollama_model
    WHISPER_MODEL_SIZE = args.whisper_model
    # Note: WHISPER_DEVICE and COMPUTE_TYPE are still determined automatically based on torch.cuda

    print(f"Using Ollama model: {OLLAMA_MODEL_NAME}")
    print(f"Using Whisper model: {WHISPER_MODEL_SIZE} (Device: {WHISPER_DEVICE}, Compute: {WHISPER_COMPUTE_TYPE})")

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
    load_whisper_model() # Load STT model once at the start
    all_summaries = {}
    processed_files = 0
    failed_files = 0

    print(f"\nScanning directory '{input_dir}' for videos...")

    # --- Iterate and Process Videos ---
    for item in os.scandir(input_dir):
        if item.is_file() and item.name.lower().endswith(VIDEO_EXTENSIONS):
            video_path = item.path
            video_filename = item.name
            base_name = os.path.splitext(video_filename)[0]
            audio_path = os.path.join(input_dir, f"temp_audio_{base_name}_{int(time.time())}{AUDIO_FORMAT}")
            transcript = None
            summary = None

            try:
                print(f"\nProcessing video: {video_filename}")

                # 1. Extract Audio
                if not extract_audio(video_path, audio_path):
                    raise ValueError(f"Audio extraction failed") # Raise error to trigger finally

                # 2. Transcribe Audio
                transcript = transcribe_audio(audio_path)
                if transcript is None: # Check for transcription failure
                    raise ValueError(f"Audio transcription failed")

                # 3. Summarize Text with Llama 3
                summary = summarize_text_with_llama3(transcript, video_filename)
                if summary: # Store summary (even if it contains an error message from the function)
                    all_summaries[video_filename] = summary
                    print(f"Summary stored for {video_filename}")
                    # Consider a summary indicating error as processed but failed
                    if "error" in summary.lower() or "failed" in summary.lower():
                        failed_files += 1
                    else:
                        processed_files += 1
                else:
                     # Should not happen with current summarize_text_with_llama3 error handling
                     raise ValueError("Unknown error during summarization (returned None)")


            except Exception as e:
                print(f"--- Failed to process video: {video_filename}. Reason: {e} ---")
                failed_files += 1
                # Ensure placeholder exists if error happened before summary generation
                if video_filename not in all_summaries:
                    all_summaries[video_filename] = f"### Summary\nFailed to process video {video_filename} due to error: {e}\n\n### Key Topics\nN/A"

            finally:
                # 4. Cleanup (Delete local temp audio)
                delete_file(audio_path)

    print(f"\n---\nSuccessfully processed {processed_files} video(s), failed to process {failed_files} video(s).")

    if not all_summaries:
        print("No video summaries were generated or processing failed for all videos. Exiting.")
        sys.exit(0)

    # --- Aggregation and Output ---
    aggregate_content, suggested_folder_name = generate_aggregate_summary_and_foldername(all_summaries)

    if aggregate_content and suggested_folder_name:
        output_folder_name = sanitize_foldername(suggested_folder_name)
        output_dir_path = os.path.join(output_parent_dir, output_folder_name)
        try:
            os.makedirs(output_dir_path, exist_ok=True)
            print(f"\nOutput folder created/ensured at: '{output_dir_path}'")

            output_md_filename = f"{output_folder_name}_summary.md"
            output_md_path = os.path.join(output_dir_path, output_md_filename)

            with open(output_md_path, 'w', encoding='utf-8') as f:
                f.write(aggregate_content)
            print(f"Aggregated summary saved to: '{output_md_path}'")

        except OSError as e:
            print(f"Error creating output directory or writing file: {e}")
            print("\n--- Aggregated Summary (Could not write to file) ---")
            print(aggregate_content)
            print("--- End of Aggregated Summary ---")
    else:
        print("Could not generate aggregate summary or folder name. No output file created.")


if __name__ == "__main__":
    main()