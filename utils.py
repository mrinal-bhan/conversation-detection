"""Utility functions for conversation detection."""

import os
import logging
import json
import csv
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path
import traceback
from tqdm import tqdm

from config import ConversationDetectionConfig
from models import ConversationAnalysis


def setup_logging(config: ConversationDetectionConfig) -> logging.Logger:
    """Set up logging configuration with rotating file handler.

    Args:
        config: Configuration object

    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(config.paths.logs_dir, exist_ok=True)

    # Create log file with timestamp
    log_file = os.path.join(
        config.paths.logs_dir,
        f"conversation_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    # Create logger
    logger = logging.getLogger("conversation_detection")
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def setup_directories(config: ConversationDetectionConfig) -> None:
    """Create necessary directories for output and temporary files.

    Args:
        config: Configuration object
    """
    directories = [
        config.paths.output_dir,
        config.paths.logs_dir,
        config.paths.temp_dir,
        os.path.join(config.paths.output_dir, "segments"),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.debug(f"Created directory: {directory}")


def discover_recordings(config: ConversationDetectionConfig) -> List[Tuple[str, str]]:
    """Discover audio recordings in the specified directory.

    Args:
        config: Configuration object

    Returns:
        List of (recording_id, file_path) tuples

    Raises:
        FileNotFoundError: If recordings directory doesn't exist
    """
    recordings_path = Path(config.paths.input_recordings_dir)

    if not recordings_path.exists():
        raise FileNotFoundError(
            f"Recordings directory not found: {config.paths.input_recordings_dir}"
        )

    # Find all .mp3 files
    audio_files = list(recordings_path.glob("*.mp3"))

    # Create list of (recording_id, file_path) tuples
    recordings = []
    for audio_file in audio_files:
        recording_id = audio_file.stem
        recordings.append((recording_id, str(audio_file)))

    logging.info(
        f"Discovered {len(recordings)} recordings in {config.paths.input_recordings_dir}"
    )
    return recordings


def save_results(
    analyses: List[ConversationAnalysis],
    config: ConversationDetectionConfig,
    timestamp: Optional[str] = None,
) -> Tuple[str, str]:
    """Save analysis results to JSON and CSV files.

    Args:
        analyses: List of conversation analysis results
        config: Configuration object
        timestamp: Optional timestamp string for filenames

    Returns:
        Tuple of (json_path, csv_path)
    """
    # Create timestamp if not provided
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results directory if it doesn't exist
    os.makedirs(config.paths.output_dir, exist_ok=True)

    # Save JSON results
    json_path = os.path.join(
        config.paths.output_dir, f"conversation_analysis_{timestamp}.json"
    )

    json_data = [analysis.to_dict() for analysis in analyses if analysis is not None]

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Save CSV results
    csv_path = os.path.join(
        config.paths.output_dir, f"conversation_analysis_{timestamp}.csv"
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "recording_id",
                "duration",
                "conversation_detected",
                "conversation_confidence",
                "caller_has_speech",
                "receiver_has_speech",
                "caller_speech_duration",
                "receiver_speech_duration",
                "turn_switches",
                "snr_db",
                "is_mono",
            ]
        )

        for analysis in analyses:
            if analysis is not None:
                writer.writerow(
                    [
                        analysis.recording_id,
                        analysis.duration,
                        analysis.conversation_detected,
                        analysis.conversation_confidence,
                        analysis.caller_analysis.has_speech,
                        analysis.receiver_analysis.has_speech,
                        analysis.caller_analysis.total_speech_duration,
                        analysis.receiver_analysis.total_speech_duration,
                        analysis.turn_taking_analysis.turn_switches,
                        analysis.audio_quality_metrics.snr_db,
                        analysis.is_mono,
                    ]
                )

    logging.info(f"Results saved to: {json_path} and {csv_path}")
    return json_path, csv_path


def process_batch(
    recordings: List[Tuple[str, str]],
    process_func,
    batch_size: int = 100,
    total_batches: Optional[int] = None,
    skip_errors: bool = True,
) -> Tuple[List[Any], Dict[str, int]]:
    """Process recordings in batches.

    Args:
        recordings: List of (recording_id, file_path) tuples
        process_func: Function to process each recording
        batch_size: Number of recordings to process in each batch
        total_batches: Maximum number of batches to process
        skip_errors: Whether to skip errors and continue processing

    Returns:
        Tuple of (results, error_counts)
    """
    results = []
    skipped_recordings = []
    error_counts = {"too_short": 0, "load_error": 0, "processing_error": 0}

    # Calculate total number of recordings to process
    total_recordings = len(recordings)
    if total_batches is not None:
        total_recordings = min(total_recordings, batch_size * total_batches)

    # Process recordings
    with tqdm(total=total_recordings, desc="Processing recordings") as pbar:
        for i, recording_data in enumerate(recordings[:total_recordings]):
            try:
                # Add a small delay between recordings to prevent resource exhaustion
                # time.sleep(0.1)  # Uncomment if needed

                result = process_func(recording_data)
                if result:
                    results.append(result)
                    pbar.set_postfix(
                        successful=len(results), skipped=len(skipped_recordings)
                    )
                else:
                    skipped_recordings.append(recording_data[0])
                    error_counts["processing_error"] += 1
            except ValueError as e:
                if "too short" in str(e):
                    logging.warning(
                        f"Skipping {recording_data[0]}: Audio file too short"
                    )
                    error_counts["too_short"] += 1
                else:
                    logging.error(
                        f"Error processing recording {recording_data[0]}: {e}"
                    )
                    error_counts["processing_error"] += 1
                skipped_recordings.append(recording_data[0])

                if not skip_errors:
                    raise
            except Exception as e:
                logging.error(f"Error processing recording {recording_data[0]}: {e}")
                logging.debug(traceback.format_exc())
                error_counts["load_error"] += 1
                skipped_recordings.append(recording_data[0])

                if not skip_errors:
                    raise
            finally:
                pbar.update(1)

                # Check if batch limit reached
                if total_batches is not None and (i + 1) % batch_size == 0:
                    batch_num = (i + 1) // batch_size
                    if batch_num >= total_batches:
                        break

    # Log batch summary
    logging.info(
        f"\nBatch processing summary:"
        f"\n- Total recordings: {total_recordings}"
        f"\n- Successfully processed: {len(results)}"
        f"\n- Failed/Skipped: {len(skipped_recordings)}"
        f"\n  * Too short: {error_counts['too_short']}"
        f"\n  * Load errors: {error_counts['load_error']}"
        f"\n  * Processing errors: {error_counts['processing_error']}"
        f"\n- Success rate: {(len(results) / total_recordings * 100):.1f}%"
    )

    if skipped_recordings:
        logging.warning(
            "\nSkipped recordings:"
            "\n"
            + "\n".join(skipped_recordings[:10])
            + (
                f"\n... and {len(skipped_recordings) - 10} more"
                if len(skipped_recordings) > 10
                else ""
            )
        )

    return results, error_counts
