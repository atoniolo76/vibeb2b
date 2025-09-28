import os
import json
import speech_recognition as sr
import subprocess
import logging
from typing import Dict, List, Any
from datetime import datetime
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FFmpegVideoProcessor:
    def __init__(self):
        """Initialize the FFmpeg-based video processor."""
        self.recognizer = sr.Recognizer()
        logger.info("FFmpeg video processor initialized")
    
    def extract_audio_from_video(self, video_path: str, audio_path: str = None) -> str:
        """
        Extract audio from MP4 video file using FFmpeg.
        
        Args:
            video_path: Path to the input MP4 file
            audio_path: Path for the output audio file (optional)
            
        Returns:
            Path to the extracted audio file
        """
        if audio_path is None:
            audio_path = video_path.replace('.mp4', '.wav')
        
        try:
            logger.info(f"Extracting audio from {video_path}")
            
            # Use FFmpeg to extract audio
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # Audio codec
                '-ar', '16000',  # Sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")
            
            logger.info(f"Audio extracted to {audio_path}")
            return audio_path
            
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install FFmpeg and add it to your PATH.")
            raise Exception("FFmpeg is required but not found. Please install FFmpeg.")
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            raise
    


    # OLD - uses Google Speech Recognition
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using Google Speech Recognition with timestamps.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing transcription with timestamps
        """
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Load audio file
            audio = AudioSegment.from_wav(audio_path)
            
            # Split audio into chunks based on silence
            chunks = split_on_silence(
                audio,
                min_silence_len=500,  # 500ms of silence
                silence_thresh=audio.dBFS-14,
                keep_silence=500
            )
            
            sentences = []
            current_time = 0
            
            for i, chunk in enumerate(chunks):
                # Export chunk to temporary file
                chunk_filename = f"temp_chunk_{i}.wav"
                chunk.export(chunk_filename, format="wav")
                
                try:
                    # Transcribe chunk
                    with sr.AudioFile(chunk_filename) as source:
                        audio_data = self.recognizer.record(source)
                        text = self.recognizer.recognize_google(audio_data)
                    
                    # Calculate timestamps
                    chunk_duration = len(chunk) / 1000.0  # Convert to seconds
                    end_time = current_time + chunk_duration
                    
                    sentence_data = {
                        "start_time": round(current_time, 2),
                        "end_time": round(end_time, 2),
                        "duration": round(chunk_duration, 2),
                        "text": text.strip(),
                        "chunk_index": i
                    }
                    
                    sentences.append(sentence_data)
                    current_time = end_time
                    
                except sr.UnknownValueError:
                    logger.warning(f"Could not understand audio in chunk {i}")
                except sr.RequestError as e:
                    logger.error(f"Error with speech recognition service: {e}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(chunk_filename):
                        os.remove(chunk_filename)
            
            # Combine all text
            full_text = " ".join([sentence["text"] for sentence in sentences])
            
            return {
                "transcription": {
                    "full_text": full_text,
                    "language": "en",  # Assuming English
                    "total_duration": round(current_time, 2),
                    "sentences": sentences,
                    "total_sentences": len(sentences)
                },
                "metadata": {
                    "model_used": "google_speech_recognition",
                    "processed_at": datetime.now().isoformat(),
                    "total_chunks": len(chunks)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            raise

    def new_transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper (OpenAI) for improved accuracy and timestamps.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary containing transcription with timestamps
        """
        try:
            logger.info(f"Transcribing audio with Whisper: {audio_path}")

            # Load Whisper model (using 'base' for speed, can change to 'small', 'medium', etc.)
            model = whisper.load_model("base")

            # Transcribe the audio
            result = model.transcribe(audio_path, language='en')  # Specify language if known

            # Extract sentences from segments
            sentences = []
            for segment in result.get('segments', []):
                sentences.append({
                    "start": segment['start'],
                    "end": segment['end'],
                    "text": segment['text'].strip()
                })

            total_duration = sentences[-1]['end'] if sentences else 0.0

            return {
                "transcription": {
                    "full_text": result.get('text', '').strip(),
                    "language": result.get('language', 'en'),
                    "total_duration": round(total_duration, 2),
                    "sentences": sentences,
                    "total_sentences": len(sentences)
                },
                "metadata": {
                    "model_used": "openai_whisper",
                    "processed_at": datetime.now().isoformat(),
                    "model_size": "base"
                }
            }

        except Exception as e:
            logger.error(f"Failed to transcribe audio with Whisper: {e}")
            raise

    def process_video(self, video_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Complete video processing pipeline: extract audio and transcribe.
        
        Args:
            video_path: Path to the input MP4 file
            output_path: Path for the output JSON file (optional)
            
        Returns:
            Transcription data as dictionary
        """
        try:
            # Extract audio
            audio_path = self.extract_audio_from_video(video_path)
            
            # Transcribe audio
            # transcription_data = self.transcribe_audio(audio_path)
            transcription_data = self.new_transcribe_audio(audio_path)
            
            # Save to JSON file if output path is provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(transcription_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Transcription saved to {output_path}")
            
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info("Temporary audio file cleaned up")
            
            return transcription_data
            
        except Exception as e:
            logger.error(f"Failed to process video: {e}")
            raise

def process_upload_folder_ffmpeg(upload_folder: str = "Upload", output_folder: str = "Output"):
    """
    Process all MP4 files in the upload folder using FFmpeg and speech recognition.
    
    Args:
        upload_folder: Path to the folder containing MP4 files
        output_folder: Path to save the output JSON files
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize processor
    processor = FFmpegVideoProcessor()
    
    # Find all MP4 files in upload folder
    mp4_files = [f for f in os.listdir(upload_folder) if f.lower().endswith('.mp4')]
    
    if not mp4_files:
        logger.warning(f"No MP4 files found in {upload_folder}")
        return
    
    logger.info(f"Found {len(mp4_files)} MP4 file(s) to process")
    
    for mp4_file in mp4_files:
        try:
            video_path = os.path.join(upload_folder, mp4_file)
            output_filename = mp4_file.replace('.mp4', '_transcription_ffmpeg.json')
            output_path = os.path.join(output_folder, output_filename)
            
            logger.info(f"Processing: {mp4_file}")
            transcription_data = processor.process_video(video_path, output_path)
            
            # Print summary
            total_sentences = transcription_data['transcription']['total_sentences']
            total_duration = transcription_data['transcription']['total_duration']
            logger.info(f"Completed: {mp4_file} - {total_sentences} sentences, {total_duration}s duration")
            
        except Exception as e:
            logger.error(f"Failed to process {mp4_file}: {e}")
            continue

if __name__ == "__main__":
    # Process all MP4 files in the Upload folder
    process_upload_folder_ffmpeg()
