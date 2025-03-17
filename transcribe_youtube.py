import imageio_ffmpeg
import gradio as gr
import subprocess
from faster_whisper import WhisperModel
import uuid
import os
import tqdm
tqdm.tqdm.disable = True

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

print("FFmpeg:", imageio_ffmpeg.get_ffmpeg_exe())

model_size = "small"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

def download_audio(youtube_url, file_name) -> None:
    """
    Download audio from a YouTube video using yt-dlp and convert it to mp3.

    Args:
        youtube_url (str): The URL of the YouTube video.
        file_name (str): The name of the output audio file.

    Raises:
        subprocess.CalledProcessError: If the command fails.

    Returns:
        None
    """
    ffmpeg_location = r"your/local/path/to/ffmpeg.exe"
    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "-x",
        "--audio-format", "mp3",
        "-o", file_name,
        "--ffmpeg-location", ffmpeg_location,
        youtube_url
    ]
    subprocess.run(cmd, check=True)

def transcribe_youtube(youtube_url, progress=gr.Progress(track_tqdm=False)):
    """
    Transcribe the audio of a YouTube video.

    Args:
        youtube_url (str): The URL of the YouTube video.

    Returns:
        str: The transcription of the audio.
    """

    file_name = f"video_audio_converted_{uuid.uuid4().hex}.mp3"
    
    # Step 1: Download and Transcription
    progress((0, 100), desc="Starting download...")
    try:
        download_audio(youtube_url, file_name)
    except Exception as e:
        progress((100, 100), desc="Download error")
        return f"Error downloading audio: {e}"
    
    progress((10, 100), desc="Download complete. Starting transcription...")
    
    segments, info = model.transcribe(file_name, beam_size=5)
    segments = list(segments)
    total_segments = len(segments)
    
    if total_segments == 0:
        progress((100, 100), desc="No segments detected")
        return "No audio to transcribe."
    
    progress((20, 100), desc=f"Detected language: {info.language} (Prob: {info.language_probability:.2f})")
    
    transcription = ""
    start_percent = 20
    end_percent = 90
    for i, segment in enumerate(segments, start=1):
        current_percent = start_percent + (end_percent - start_percent) * (i / total_segments)
        progress((int(current_percent), 100), desc=f"Processing segment {i}/{total_segments}")
        transcription += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
    
    progress((100, 100), desc="Transcription complete")
    
    try:
        os.remove(file_name)
    except Exception as e:
        print(f"Could not remove temporary file {file_name}: {e}")
    
    return transcription

def refine_transcription(transcription, progress=gr.Progress(track_tqdm=False)) -> str:
    """
    Refine the transcription using ChatOllama.

    Args:
        transcription (str): The transcription to refine.

    Returns:
        str: The refined transcription.
    """
    # This function refines the transcription with its own progress bar
    progress((0, 100), desc="Starting refinement...")
    
    llm = ChatOllama(temperature=0, model="gemma3:12b")
    template = """You are an expert assistant in refining raw video transcriptions. The text provided contains timestamps, occasional disfluencies, and formatting artifacts that make it hard to read. Your task is to reformat the transcription so that it is clear and well-organized, while preserving all the original content and details. Do not summarize or omit any information; just remove unnecessary timestamps and artifacts, and adjust the text for improved readability.

    Raw Transcription:
    {transcription}

    Refined Transcription (in the language of the transcription):
    """
    prompt = ChatPromptTemplate.from_template(template)
    messages = prompt.invoke({"transcription": transcription})

    progress((20, 100), desc="Refinement in progress...")

    response = llm.invoke(messages)
    
    progress((100, 100), desc="Refinement complete")
    return response.content

with gr.Blocks() as demo:
    gr.Markdown("# Automatic YouTube Video Transcription and Refinement")
    
    with gr.Column():
        gr.Markdown("## Transcription")
        youtube_url = gr.Textbox(label="YouTube Link", placeholder="Paste the YouTube video link here")
        transcribe_btn = gr.Button("Transcribe")
        transcription_box = gr.Textbox(label="Complete Transcription", lines=15)
    
    with gr.Column():
        gr.Markdown("## Refinement")
        refined_box = gr.Textbox(label="Refined Transcription", lines=15)
    
    # When the button is clicked, the video is transcribed and the transcription is shown.
    transcribe_btn.click(
        fn=transcribe_youtube,
        inputs=youtube_url,
        outputs=transcription_box
    )
    
    # When the transcription box is updated, automatically call the refinement function.
    transcription_box.change(
        fn=refine_transcription,
        inputs=transcription_box,
        outputs=refined_box
    )

demo.launch()
