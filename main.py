#!/usr/bin/env python3
"""
Conversation Detection System

This system analyzes call recordings to identify meaningful conversations between
callers and receivers using advanced audio analysis and machine learning.
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import ConversationDetectionConfig
from vad import AudioProcessor
from analyzer import ConversationAnalyzer
from metrics import MetricsCalculator
from models import ConversationAnalysis, MetricsResult
import utils


class ConversationDetector:
    """Conversation detection system for analyzing call recordings."""

    def __init__(self, config: ConversationDetectionConfig = None):
        """Initialize the conversation detector.

        Args:
            config: Configuration object, uses default if None
        """
        self.config = config or ConversationDetectionConfig()
        self.logger = utils.setup_logging(self.config)
        utils.setup_directories(self.config)

        # Initialize components
        self.audio_processor = AudioProcessor(self.config)
        self.conversation_analyzer = ConversationAnalyzer(self.config)
        self.metrics_calculator = MetricsCalculator(self.config)

    def analyze_single_recording(
        self, recording_data: Tuple[str, str]
    ) -> Optional[ConversationAnalysis]:
        """Analyze a single recording for conversation detection.

        Args:
            recording_data: Tuple of (recording_id, file_path)

        Returns:
            ConversationAnalysis object or None if analysis fails
        """
        recording_id, file_path = recording_data
        self.logger.debug(f"Analyzing recording: {recording_id}")

        try:
            # Load and prepare audio
            audio_data, sample_rate, is_mono = (
                self.audio_processor.load_and_prepare_audio(file_path)
            )
            duration = len(audio_data[0]) / sample_rate

            # Analyze each channel
            caller_analysis = self.audio_processor.analyze_channel(
                audio_data[self.config.audio.caller_channel],
                sample_rate,
                self.config.audio.caller_channel,
                "caller",
            )

            receiver_analysis = self.audio_processor.analyze_channel(
                audio_data[self.config.audio.receiver_channel],
                sample_rate,
                self.config.audio.receiver_channel,
                "receiver",
            )

            # Calculate audio quality metrics
            audio_quality_metrics = (
                self.audio_processor.calculate_audio_quality_metrics(
                    audio_data, sample_rate
                )
            )

            # Create conversation analysis
            analysis = self.conversation_analyzer.analyze_conversation(
                recording_id=recording_id,
                duration=duration,
                caller_analysis=caller_analysis,
                receiver_analysis=receiver_analysis,
                audio_quality_dict=audio_quality_metrics,
                is_mono=is_mono,
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing recording {recording_id}: {str(e)}")
            self.logger.debug(f"Error details: {e}", exc_info=True)
            return None

    def process_recordings_batch(
        self, recordings: List[Tuple[str, str]], batch_size: int = None
    ) -> List[ConversationAnalysis]:
        """Process multiple recordings in batches.

        Args:
            recordings: List of (recording_id, file_path) tuples
            batch_size: Number of recordings to process in each batch

        Returns:
            List of ConversationAnalysis results
        """
        if batch_size is None:
            batch_size = self.config.processing.batch_size

        self.logger.info(f"Processing batch of {len(recordings)} recordings")

        # Use the utility function to process the batch
        results, _ = utils.process_batch(
            recordings, self.analyze_single_recording, batch_size=batch_size
        )

        return results

    def run_analysis(
        self, recordings_dir: str = None, max_recordings: int = None
    ) -> Tuple[List[ConversationAnalysis], Optional[MetricsResult]]:
        """Run the complete conversation detection analysis pipeline.

        Args:
            recordings_dir: Directory containing audio recordings
            max_recordings: Maximum number of recordings to process

        Returns:
            Tuple of (list of analysis results, optional metrics)
        """
        start_time = datetime.now()

        if recordings_dir:
            self.config.paths.input_recordings_dir = recordings_dir

        # Discover recordings
        recordings = utils.discover_recordings(self.config)
        if max_recordings:
            recordings = recordings[:max_recordings]

        # Process in batches
        batch_size = self.config.processing.batch_size
        all_analyses = []

        for i in range(0, len(recordings), batch_size):
            batch = recordings[i : i + batch_size]
            results = self.process_recordings_batch(batch, batch_size)
            all_analyses.extend(results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        utils.save_results(all_analyses, self.config, timestamp)

        # Calculate metrics if ground truth is available
        metrics = None
        try:
            ground_truth = self.metrics_calculator.generate_ground_truth_from_transcription_report()
            if ground_truth:
                metrics = self.metrics_calculator.calculate_metrics(
                    all_analyses, ground_truth
                )

                # Generate error analysis
                false_positives, false_negatives = (
                    self.metrics_calculator.analyze_errors(all_analyses, ground_truth)
                )

                # Save metrics and generate report
                self.metrics_calculator.save_metrics(metrics)
                self.metrics_calculator.generate_metrics_report(
                    metrics, false_positives, false_negatives
                )
        except Exception as e:
            self.logger.warning(f"Could not calculate metrics: {str(e)}")

        end_time = datetime.now()
        self.logger.info(f"Analysis completed in {end_time - start_time}")

        return all_analyses, metrics


def main():
    """Main entry point for conversation detection from command line."""
    parser = argparse.ArgumentParser(description="Conversation Detection System")
    parser.add_argument(
        "--recordings",
        help="Path to directory containing audio recordings",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face token for accessing models",
    )
    parser.add_argument(
        "--config",
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--output",
        help="Path to output directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of recordings to process in each batch (default: 100)",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=None,
        help="Number of batches to process",
    )
    args = parser.parse_args()

    try:
        # Load configuration
        if args.config:
            config = ConversationDetectionConfig.from_json(args.config)
        else:
            config = ConversationDetectionConfig()

        # Override configuration with command line arguments
        if args.token:
            config.huggingface_token = args.token

        if args.recordings:
            config.paths.input_recordings_dir = args.recordings

        if args.output:
            config.paths.output_dir = args.output

        if args.batch_size:
            config.processing.batch_size = args.batch_size

        # Initialize detector
        detector = ConversationDetector(config)

        # Get recordings
        recordings = utils.discover_recordings(config)

        # Calculate total recordings to process
        total_recordings = len(recordings)
        if args.num_batches:
            total_recordings = min(total_recordings, args.batch_size * args.num_batches)
            recordings = recordings[:total_recordings]

        print(f"\nProcessing {total_recordings} recordings")
        if args.num_batches:
            print(f"Processing in {args.num_batches} batches of {args.batch_size}")

        # Process recordings in batches
        all_analyses = []

        if args.num_batches:
            # Process specified number of batches
            for batch_num in range(args.num_batches):
                start_idx = batch_num * args.batch_size
                end_idx = min(start_idx + args.batch_size, total_recordings)
                batch_recordings = recordings[start_idx:end_idx]

                if not batch_recordings:
                    break

                print(f"\nProcessing batch {batch_num + 1}/{args.num_batches}")
                print(f"Recordings {start_idx + 1} to {end_idx}")

                try:
                    # Process the batch
                    batch_analyses = detector.process_recordings_batch(
                        batch_recordings, batch_size=args.batch_size
                    )
                    all_analyses.extend(batch_analyses)

                    # Save intermediate results after each batch
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    utils.save_results(batch_analyses, config, timestamp)

                except KeyboardInterrupt:
                    print(
                        "\n\nProcessing interrupted by user. Saving partial results..."
                    )
                    break
                except Exception as e:
                    print(f"\nError processing batch {batch_num + 1}: {str(e)}")
                    continue
        else:
            # Process all recordings
            all_analyses, _ = detector.run_analysis()

        # Save final results if we have any analyses
        if all_analyses:
            # Calculate metrics if ground truth is available
            try:
                # Try to calculate metrics if ground truth is available
                ground_truth = detector.metrics_calculator.generate_ground_truth_from_transcription_report()
                if ground_truth:
                    metrics = detector.metrics_calculator.calculate_metrics(
                        all_analyses, ground_truth
                    )
                    detector.metrics_calculator.save_metrics(metrics)

                    # Generate error analysis
                    false_positives, false_negatives = (
                        detector.metrics_calculator.analyze_errors(
                            all_analyses, ground_truth
                        )
                    )

                    # Generate report
                    detector.metrics_calculator.generate_metrics_report(
                        metrics, false_positives, false_negatives
                    )
            except Exception as e:
                print(f"\nError calculating metrics: {str(e)}")

            print("\n✅ Analysis completed successfully!")
            print(f"Processed {len(all_analyses)} recordings")
            print(f"Results saved in: {config.paths.output_dir}")
        else:
            print("\n❌ No recordings were successfully processed")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
