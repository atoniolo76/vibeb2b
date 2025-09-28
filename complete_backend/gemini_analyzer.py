#!/usr/bin/env python3
"""
GEMINI ANALYZER
Uses Google Gemini API to analyze meeting performance based on synchronized
transcription and emotion data. Provides actionable feedback on engagement
and confusion/frustration patterns.
"""

import os
import json
import requests
import logging
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiAnalyzer:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        """Initialize the Gemini analyzer."""
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        logger.info(f"Gemini Analyzer initialized with model: {model}")
    
    def read_synchronized_analysis(self, file_path: str) -> str:
        """Read the synchronized analysis text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Read synchronized analysis from {file_path}")
            return content
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise
    
    def call_gemini_api(self, prompt: str) -> str:
        """Call the Gemini API with the given prompt."""
        try:
            url = f"{self.base_url}/{self.model}:generateContent"
            headers = {
                "Content-Type": "application/json",
            }
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048,
                }
            }
            
            params = {"key": self.api_key}
            
            logger.info("Calling Gemini API...")
            response = requests.post(url, headers=headers, json=payload, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    logger.info("✅ Gemini API call successful")
                    return content
                else:
                    raise Exception("No content in API response")
            else:
                raise Exception(f"API call failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to call Gemini API: {e}")
            raise
    
    def analyze_meeting_performance(self, synchronized_data: str) -> str:
        """Analyze meeting performance using Gemini API."""
        prompt = f"""This is a timestamp of my meeting. Each line is formatted as 

Timestamp - what I said in the meeting - frustration/confusion of the audience (out of 100) - engagement of the audience (out of 100)

I want you to give me advice on my performance this meeting. Identify the top 3-5 spots of highest engagement, and top 3-5 spots of highest confusion/frustration. Write specific, actionable feedback for each key point. Include what I said, when I said it, and what I can do to improve if relevant. 

Your reply should be ONLY that feedback list, nothing else responding to this message like opening comments or additional end comments. Just the feedback itself. 

Meeting Data:
{synchronized_data}"""

        return self.call_gemini_api(prompt)
    
    def save_analysis(self, analysis: str, output_file: str = "gemini_output.txt"):
        """Save the Gemini analysis to a file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(analysis)
            logger.info(f"Gemini analysis saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            raise

def main():
    """Main function to run the Gemini analyzer."""
    # Configuration
    API_KEY = "AIzaSyChau4MVYnQTDahLYnlmT_vt4mN4L_TI2k"
    MODEL = "gemini-2.0-flash-exp"  # Using 2.0 flash as requested
    SYNCHRONIZED_FILE = "Output/synchronized_analysis.txt"
    OUTPUT_FILE = "Output/gemini_output.txt"
    
    # Check if synchronized analysis file exists
    if not os.path.exists(SYNCHRONIZED_FILE):
        logger.error(f"Synchronized analysis file not found: {SYNCHRONIZED_FILE}")
        logger.info("Please run json_parser.py first to generate the synchronized analysis")
        return
    
    try:
        # Initialize analyzer
        analyzer = GeminiAnalyzer(API_KEY, MODEL)
        
        # Read synchronized analysis
        logger.info("Reading synchronized analysis data...")
        synchronized_data = analyzer.read_synchronized_analysis(SYNCHRONIZED_FILE)
        
        if not synchronized_data.strip():
            logger.error("Synchronized analysis file is empty")
            return
        
        # Analyze meeting performance
        logger.info("Analyzing meeting performance with Gemini...")
        analysis = analyzer.analyze_meeting_performance(synchronized_data)
        
        # Save analysis
        analyzer.save_analysis(analysis, OUTPUT_FILE)
        
        # Print summary
        print("\n" + "=" * 60)
        print("GEMINI MEETING ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Analysis saved to: {OUTPUT_FILE}")
        print(f"Analysis length: {len(analysis)} characters")
        print("=" * 60)
        
        # Show preview of analysis
        preview = analysis[:500] + "..." if len(analysis) > 500 else analysis
        print("\nPreview of analysis:")
        print("-" * 40)
        print(preview)
        print("-" * 40)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()
