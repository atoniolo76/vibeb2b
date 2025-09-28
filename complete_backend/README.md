# Video Analysis Pipeline

A complete AI-powered video analysis system that processes MP4 meeting recordings to provide transcription, emotion analysis, and actionable feedback.

## 🎯 Features

- **Video Transcription**: Extracts audio and transcribes speech with timestamps
- **Emotion Analysis**: Analyzes facial emotions every 5th frame using FER model
- **Data Synchronization**: Combines transcription and emotion data by time ranges
- **AI Feedback**: Uses Gemini AI to provide actionable meeting performance insights

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Place MP4 files** in the `Upload/` folder

3. **Run the complete pipeline**:
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
ProcessVidApp/
├── main.py                           # Main pipeline orchestrator
├── video_processor_ffmpeg.py         # Video transcription (Step 1)
├── main_emotion_parser.py            # Emotion analysis (Step 2)
├── json_parser.py                    # Data synchronization (Step 3)
├── gemini_analyzer.py                # AI feedback (Step 4)
├── FER_static_ResNet50_AffectNet.pt  # Emotion detection model
├── requirements.txt                  # Python dependencies
├── Upload/                           # Place MP4 files here
├── Output/                           # Analysis results
└── Output/gemini_output.txt          # AI feedback report
```

## 🔄 Pipeline Flow

1. **Transcription**: Extracts audio from MP4 → Transcribes speech with timestamps
2. **Emotion Analysis**: Processes every 5th frame → Detects emotions and calculates engagement/frustration
3. **Synchronization**: Matches transcription timestamps with emotion data → Creates unified analysis
4. **AI Analysis**: Sends data to Gemini → Generates actionable feedback report

## 📊 Output Files

- **`Output/gemini_output.txt`**: AI-powered meeting performance feedback
- **`Output/synchronized_analysis.txt`**: Timestamped transcription with emotion scores
- **`Output/analysis_summary.txt`**: Statistical summary of emotions
- **`Output/*_transcription_ffmpeg.json`**: Raw transcription data
- **`Output/*_emotion_analysis.json`**: Raw emotion analysis data

## 🎛️ Emotion Scoring

- **Engagement (0-100)**: Based on happiness, reduced by negative emotions
- **Frustration/Confusion (0-100)**: Based on fear, surprise, disgust, and brow furrow analysis

## 📋 Requirements

- Python 3.8+
- FFmpeg (for audio extraction)
- Internet connection (for Gemini API and Google Speech Recognition)
- Sufficient disk space for temporary files

## 🔧 Individual Components

You can also run individual components:

```bash
# Just transcription
python video_processor_ffmpeg.py

# Just emotion analysis  
python main_emotion_parser.py

# Just data synchronization
python json_parser.py

# Just AI analysis (requires synchronized data)
python gemini_analyzer.py
```

## 📝 Example Output

```
[00:12 - 00:15] - I'm so confused - Frustration/Confusion: 1.6 - Engagement: 74.6
[00:19 - 00:24] - what what is happening I'm so confused - Frustration/Confusion: 4.7 - Engagement: 0.0
```

## 🤖 AI Feedback

The Gemini AI analyzes your meeting performance and provides:
- Top 3-5 highest engagement moments
- Top 3-5 highest confusion/frustration moments  
- Specific, actionable feedback for each key point
- Improvement suggestions based on emotional patterns

## ⚠️ Notes

- First run will download the FER model (one-time setup)
- Processing time depends on video length and model complexity
- Temporary files are automatically cleaned up
- All analysis is saved locally for privacy