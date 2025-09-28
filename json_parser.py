#!/usr/bin/env python3
"""
JSON PARSER - Synchronizes Transcription and Emotion Data
Parses both transcription and emotion analysis JSON files from the Output folder.
For each time range in the transcription, finds the median engagement and frustration
scores from the emotion analysis within the same time range.

Output format:
"[time range start - time range end] - text found in transcript - Frustration/Confusion: [median score] - Engagement: [median score]"
"""

import os
import json
import logging
from typing import Dict, List, Any, Tuple
import statistics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JSONParser:
    def __init__(self, output_folder="Output"):
        """Initialize the JSON parser."""
        self.output_folder = output_folder
        logger.info("JSON Parser initialized")
    
    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Load and parse a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded JSON file: {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            raise
    
    def find_matching_files(self) -> List[Tuple[str, str]]:
        """
        Find matching transcription and emotion analysis files.
        
        Returns:
            List of tuples (transcription_file, emotion_file)
        """
        files = os.listdir(self.output_folder)
        
        # Find transcription files
        transcription_files = [f for f in files if f.endswith('_transcription_ffmpeg.json')]
        # Find emotion analysis files
        emotion_files = [f for f in files if f.endswith('_emotion_analysis.json')]
        
        matching_pairs = []
        
        for trans_file in transcription_files:
            # Extract base name (remove _transcription_ffmpeg.json)
            base_name = trans_file.replace('_transcription_ffmpeg.json', '')
            
            # Find corresponding emotion file
            emotion_file = f"{base_name}_emotion_analysis.json"
            
            if emotion_file in emotion_files:
                matching_pairs.append((trans_file, emotion_file))
                logger.info(f"Found matching pair: {trans_file} <-> {emotion_file}")
            else:
                logger.warning(f"No emotion analysis file found for {trans_file}")
        
        return matching_pairs
    
    def get_emotion_scores_in_range(self, emotion_data: Dict[str, Any], start_time: float, end_time: float) -> Tuple[float, float]:
        """
        Get median frustration and engagement scores within a time range.
        
        Args:
            emotion_data: Emotion analysis JSON data
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Tuple of (median_frustration, median_engagement)
        """
        frame_analysis = emotion_data.get('frame_analysis', [])
        
        # Filter frames within the time range
        scores_in_range = []
        for frame in frame_analysis:
            frame_timestamp = frame.get('timestamp', 0)
            if start_time <= frame_timestamp <= end_time:
                emotions = frame.get('emotions', {})
                frustration = emotions.get('frustration', 0)
                engagement = emotions.get('engagement', 0)
                scores_in_range.append((frustration, engagement))
        
        if not scores_in_range:
            logger.warning(f"No emotion data found in range {start_time:.2f}s - {end_time:.2f}s")
            return 0.0, 0.0
        
        # Calculate median scores
        frustrations = [score[0] for score in scores_in_range]
        engagements = [score[1] for score in scores_in_range]
        
        median_frustration = statistics.median(frustrations) if frustrations else 0.0
        median_engagement = statistics.median(engagements) if engagements else 0.0
        
        logger.debug(f"Range {start_time:.2f}s - {end_time:.2f}s: {len(scores_in_range)} frames, "
                    f"median frustration: {median_frustration:.2f}, median engagement: {median_engagement:.2f}")
        
        return median_frustration, median_engagement
    
    def format_time_range(self, start_time: float, end_time: float) -> str:
        """Format time range as MM:SS - MM:SS."""
        start_min = int(start_time // 60)
        start_sec = int(start_time % 60)
        end_min = int(end_time // 60)
        end_sec = int(end_time % 60)
        
        return f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
    
    def parse_and_synchronize(self, transcription_file: str, emotion_file: str) -> List[str]:
        """
        Parse and synchronize transcription and emotion data.
        
        Args:
            transcription_file: Path to transcription JSON file
            emotion_file: Path to emotion analysis JSON file
            
        Returns:
            List of formatted output lines
        """
        # Load JSON files
        trans_data = self.load_json_file(os.path.join(self.output_folder, transcription_file))
        emotion_data = self.load_json_file(os.path.join(self.output_folder, emotion_file))
        
        # Extract transcription sentences
        transcription = trans_data.get('transcription', {})
        sentences = transcription.get('sentences', [])
        
        output_lines = []
        
        logger.info(f"Processing {len(sentences)} sentences from transcription")
        
        for sentence in sentences:
            start_time = sentence.get('start_time', 0)
            end_time = sentence.get('end_time', 0)
            text = sentence.get('text', '').strip()
            
            if not text:  # Skip empty sentences
                continue
            
            # Get emotion scores for this time range
            median_frustration, median_engagement = self.get_emotion_scores_in_range(
                emotion_data, start_time, end_time
            )
            
            # Format the output line
            time_range = self.format_time_range(start_time, end_time)
            output_line = (f"[{time_range}] - {text} - "
                          f"Frustration/Confusion: {median_frustration:.1f} - "
                          f"Engagement: {median_engagement:.1f}")
            
            output_lines.append(output_line)
            logger.info(f"Processed: {time_range} - {text[:50]}{'...' if len(text) > 50 else ''}")
        
        return output_lines
    
    def process_all_files(self, output_text_file: str = None) -> Dict[str, List[str]]:
        """
        Process all matching transcription and emotion files.
        
        Args:
            output_text_file: Path to save the combined output (optional)
            
        Returns:
            Dictionary mapping file pairs to their output lines
        """
        matching_pairs = self.find_matching_files()
        
        if not matching_pairs:
            logger.warning("No matching transcription and emotion files found")
            return {}
        
        all_results = {}
        all_output_lines = []
        
        for trans_file, emotion_file in matching_pairs:
            logger.info(f"Processing pair: {trans_file} <-> {emotion_file}")
            
            try:
                output_lines = self.parse_and_synchronize(trans_file, emotion_file)
                all_results[f"{trans_file} + {emotion_file}"] = output_lines
                all_output_lines.extend(output_lines)
                
                # Add separator between different video files
                if len(matching_pairs) > 1:
                    all_output_lines.append(f"\n--- End of {trans_file} ---\n")
                
            except Exception as e:
                logger.error(f"Failed to process {trans_file} + {emotion_file}: {e}")
                continue
        
        # Save to text file if specified
        if output_text_file:
            with open(output_text_file, 'w', encoding='utf-8') as f:
                for line in all_output_lines:
                    f.write(line + '\n')
            logger.info(f"Combined output saved to {output_text_file}")
        
        return all_results
    
    def generate_summary_report(self, results: Dict[str, List[str]], summary_file: str = None) -> Dict[str, Any]:
        """
        Generate a summary report of the analysis.
        
        Args:
            results: Results from process_all_files
            summary_file: Path to save summary report (optional)
            
        Returns:
            Summary statistics
        """
        total_sentences = sum(len(lines) for lines in results.values())
        
        # Extract all frustration and engagement scores
        all_frustrations = []
        all_engagements = []
        
        for lines in results.values():
            for line in lines:
                # Parse scores from the formatted line
                if "Frustration/Confusion:" in line and "Engagement:" in line:
                    try:
                        parts = line.split("Frustration/Confusion: ")[1].split(" - Engagement: ")
                        frustration = float(parts[0])
                        engagement = float(parts[1])
                        all_frustrations.append(frustration)
                        all_engagements.append(engagement)
                    except (ValueError, IndexError):
                        continue
        
        summary = {
            'total_video_files': len(results),
            'total_sentences_analyzed': total_sentences,
            'average_frustration': statistics.mean(all_frustrations) if all_frustrations else 0,
            'average_engagement': statistics.mean(all_engagements) if all_engagements else 0,
            'median_frustration': statistics.median(all_frustrations) if all_frustrations else 0,
            'median_engagement': statistics.median(all_engagements) if all_engagements else 0,
            'max_frustration': max(all_frustrations) if all_frustrations else 0,
            'max_engagement': max(all_engagements) if all_engagements else 0,
            'min_frustration': min(all_frustrations) if all_frustrations else 0,
            'min_engagement': min(all_engagements) if all_engagements else 0
        }
        
        if summary_file:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("EMOTION ANALYSIS SUMMARY REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total video files processed: {summary['total_video_files']}\n")
                f.write(f"Total sentences analyzed: {summary['total_sentences_analyzed']}\n\n")
                f.write("FRUSTRATION/CONFUSION STATISTICS:\n")
                f.write(f"  Average: {summary['average_frustration']:.2f}\n")
                f.write(f"  Median: {summary['median_frustration']:.2f}\n")
                f.write(f"  Range: {summary['min_frustration']:.2f} - {summary['max_frustration']:.2f}\n\n")
                f.write("ENGAGEMENT STATISTICS:\n")
                f.write(f"  Average: {summary['average_engagement']:.2f}\n")
                f.write(f"  Median: {summary['median_engagement']:.2f}\n")
                f.write(f"  Range: {summary['min_engagement']:.2f} - {summary['max_engagement']:.2f}\n")
            
            logger.info(f"Summary report saved to {summary_file}")
        
        return summary

def main():
    """Main function to run the JSON parser."""
    parser = JSONParser()
    
    # Process all files and generate output
    results = parser.process_all_files("Output/synchronized_analysis.txt")
    
    if results:
        # Generate summary report
        summary = parser.generate_summary_report(results, "Output/analysis_summary.txt")
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("EMOTION ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total video files processed: {summary['total_video_files']}")
        print(f"Total sentences analyzed: {summary['total_sentences_analyzed']}")
        print(f"\nAverage Frustration/Confusion: {summary['average_frustration']:.2f}")
        print(f"Average Engagement: {summary['average_engagement']:.2f}")
        print(f"\nMedian Frustration/Confusion: {summary['median_frustration']:.2f}")
        print(f"Median Engagement: {summary['median_engagement']:.2f}")
        print("=" * 60)
        
        logger.info("JSON parsing and synchronization completed successfully!")
    else:
        logger.warning("No files were processed")

if __name__ == "__main__":
    main()
