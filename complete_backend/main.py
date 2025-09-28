#!/usr/bin/env python3
"""
MAIN PIPELINE - Complete Video Analysis System
Orchestrates the entire video analysis pipeline:
1. Video transcription (FFmpeg + Speech Recognition)
2. Emotion analysis (FER model + MediaPipe)
3. Data synchronization (JSON parser)
4. AI-powered feedback (Gemini API)

This is the main entry point for the complete video analysis system.
"""

import os
import sys
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all required files and dependencies are available."""
    logger.info("🔍 Checking system requirements...")
    
    # Check for required files
    required_files = [
        "video_processor_ffmpeg.py",
        "main_emotion_parser.py", 
        "json_parser.py",
        "gemini_analyzer.py",
        "FER_static_ResNet50_AffectNet.pt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        return False
    
    # Check for Upload folder
    if not os.path.exists("Upload"):
        logger.error("❌ Upload folder not found")
        return False
    
    # Check for MP4 files
    mp4_files = [f for f in os.listdir("Upload") if f.lower().endswith('.mp4')]
    if not mp4_files:
        logger.error("❌ No MP4 files found in Upload folder")
        return False
    
    logger.info(f"✅ Found {len(mp4_files)} MP4 file(s) to process")
    logger.info("✅ All requirements met")
    return True

def run_transcription():
    """Step 1: Run video transcription using FFmpeg."""
    logger.info("🎤 STEP 1: Starting video transcription...")
    
    try:
        # Import and run the FFmpeg processor
        from video_processor_ffmpeg import process_upload_folder_ffmpeg
        
        process_upload_folder_ffmpeg()
        logger.info("✅ Transcription completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Transcription failed: {e}")
        return False

def run_emotion_analysis():
    """Step 2: Run emotion analysis using FER model."""
    logger.info("😊 STEP 2: Starting emotion analysis...")
    
    try:
        # Import and run the emotion parser
        from main_emotion_parser import process_upload_folder_emotions
        
        process_upload_folder_emotions()
        logger.info("✅ Emotion analysis completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Emotion analysis failed: {e}")
        return False

def run_data_synchronization():
    """Step 3: Synchronize transcription and emotion data."""
    logger.info("🔄 STEP 3: Starting data synchronization...")
    
    try:
        # Import and run the JSON parser
        from json_parser import JSONParser
        
        parser = JSONParser()
        results = parser.process_all_files("Output/synchronized_analysis.txt")
        
        if results:
            # Generate summary report
            summary = parser.generate_summary_report(results, "Output/analysis_summary.txt")
            logger.info("✅ Data synchronization completed successfully")
            return True
        else:
            logger.error("❌ No data to synchronize")
            return False
            
    except Exception as e:
        logger.error(f"❌ Data synchronization failed: {e}")
        return False

def run_ai_analysis():
    """Step 4: Run AI-powered performance analysis using Gemini."""
    logger.info("🤖 STEP 4: Starting AI performance analysis...")
    
    try:
        # Check if synchronized analysis exists
        if not os.path.exists("Output/synchronized_analysis.txt"):
            logger.error("❌ Synchronized analysis file not found")
            return False
        
        # Import and run the Gemini analyzer
        from gemini_analyzer import GeminiAnalyzer
        
        API_KEY = "AIzaSyAsJlOxDK9nKLpnKoiAv0DU32Fw5QkVFNQ"
        MODEL = "gemini-2.0-flash-exp"
        
        analyzer = GeminiAnalyzer(API_KEY, MODEL)
        
        # Read synchronized data
        synchronized_data = analyzer.read_synchronized_analysis("Output/synchronized_analysis.txt")
        
        # Analyze with Gemini
        analysis = analyzer.analyze_meeting_performance(synchronized_data)
        
        # Save analysis
        analyzer.save_analysis(analysis, "Output/gemini_output.txt")
        
        logger.info("✅ AI analysis completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ AI analysis failed: {e}")
        return False

def print_pipeline_summary():
    """Print a summary of the pipeline results."""
    logger.info("📊 PIPELINE SUMMARY")
    logger.info("=" * 50)
    
    # Check output files
    output_files = [
        ("Transcription", "Output/*_transcription_ffmpeg.json"),
        ("Emotion Analysis", "Output/*_emotion_analysis.json"), 
        ("Synchronized Data", "Output/synchronized_analysis.txt"),
        ("Analysis Summary", "Output/analysis_summary.txt"),
        ("AI Feedback", "Output/gemini_output.txt")
    ]
    
    for name, pattern in output_files:
        if "*" in pattern:
            # Check for any files matching pattern
            import glob
            files = glob.glob(pattern)
            status = "✅" if files else "❌"
            count = len(files) if files else 0
            logger.info(f"{status} {name}: {count} file(s)")
        else:
            status = "✅" if os.path.exists(pattern) else "❌"
            logger.info(f"{status} {name}: {pattern}")
    
    logger.info("=" * 50)

def main():
    """Main pipeline orchestrator."""
    start_time = datetime.now()
    
    print("🚀 VIDEO ANALYSIS PIPELINE")
    print("=" * 50)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        logger.error("❌ Requirements check failed. Exiting.")
        return False
    
    # Run pipeline steps
    steps = [
        ("Transcription", run_transcription),
        ("Emotion Analysis", run_emotion_analysis), 
        ("Data Synchronization", run_data_synchronization),
        ("AI Analysis", run_ai_analysis)
    ]
    
    success_count = 0
    for step_name, step_function in steps:
        logger.info(f"\n🔄 Running {step_name}...")
        if step_function():
            success_count += 1
            logger.info(f"✅ {step_name} completed successfully")
        else:
            logger.error(f"❌ {step_name} failed")
            # Continue with next step even if one fails
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 50)
    print("🏁 PIPELINE COMPLETED")
    print("=" * 50)
    print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"Steps completed: {success_count}/{len(steps)}")
    print("=" * 50)
    
    print_pipeline_summary()
    
    if success_count == len(steps):
        logger.info("🎉 All pipeline steps completed successfully!")
        print("\n🎉 SUCCESS! Check the following files for results:")
        print("   📄 Output/gemini_output.txt - AI-powered meeting feedback")
        print("   📄 Output/synchronized_analysis.txt - Timestamped analysis")
        print("   📄 Output/analysis_summary.txt - Statistical summary")
        return True
    else:
        logger.warning(f"⚠️ Pipeline completed with {len(steps) - success_count} failures")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n👋 Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)
