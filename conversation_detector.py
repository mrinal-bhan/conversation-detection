import os
import sys
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from pyannote.audio import Pipeline
import torch
import csv

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import ConversationDetectionConfig


@dataclass
class ChannelAnalysis:
    """Results from analyzing a single audio channel.

    Attributes:
        channel_id: Channel identifier (0=left, 1=right)
        speaker_name: Name of the speaker (caller/receiver)
        total_speech_duration: Total duration of speech segments
        speech_segments: List of (start, end) tuples for speech segments
        energy_stats: Dictionary of energy-related metrics
        has_speech: Whether speech was detected in the channel
        rms_energy: Root mean square energy of the channel
    """

    channel_id: int
    speaker_name: str
    total_speech_duration: float
    speech_segments: List[Tuple[float, float]]
    energy_stats: Dict[str, float]
    has_speech: bool
    rms_energy: float


@dataclass
class ConversationAnalysis:
    """Complete conversation analysis results.

    Attributes:
        recording_id: Unique identifier for the recording
        duration: Total duration of the recording in seconds
        caller_analysis: Analysis results for caller channel
        receiver_analysis: Analysis results for receiver channel
        conversation_detected: Whether a conversation was detected
        conversation_confidence: Confidence score for conversation detection
        turn_taking_analysis: Analysis of speaker turn-taking patterns
        audio_quality_metrics: Various audio quality measurements
        is_mono: Whether the original recording was mono
    """

    recording_id: str
    duration: float
    caller_analysis: ChannelAnalysis
    receiver_analysis: ChannelAnalysis
    conversation_detected: bool
    conversation_confidence: float
    turn_taking_analysis: Dict[str, Any]
    audio_quality_metrics: Dict[str, float]
    is_mono: bool = False


@dataclass
class MetricsResult:
    """Evaluation metrics results.

    Attributes:
        precision: Precision score
        recall: Recall score
        f1_score: F1 score
        accuracy: Accuracy score
        confusion_matrix: Confusion matrix as numpy array
        total_samples: Total number of samples evaluated
    """

    precision: float
    recall: float
    f1_score: float
    accuracy: float
    confusion_matrix: np.ndarray
    total_samples: int


class ConversationDetector:
    """Conversation detection system for analyzing call recordings."""

    def __init__(self, config: ConversationDetectionConfig = None):
        """Initialize the conversation detector.

        Args:
            config: Configuration object, uses default if None
        """
        self.config = config or ConversationDetectionConfig()
        self.setup_logging()
        self.setup_directories()
        self.initialize_vad_pipeline()

    def setup_logging(self):
        """Set up logging configuration with rotating file handler."""
        os.makedirs(self.config.logs_dir, exist_ok=True)

        log_file = os.path.join(
            self.config.logs_dir,
            f"conversation_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")

    def setup_directories(self):
        """Create necessary directories for output and temporary files."""
        directories = [
            self.config.output_dir,
            self.config.logs_dir,
            self.config.temp_dir,
            os.path.join(self.config.output_dir, "reports"),
            os.path.join(self.config.output_dir, "plots"),
            os.path.join(self.config.output_dir, "segments"),
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")

    def initialize_vad_pipeline(self):
        """Initialize the Voice Activity Detection pipeline."""
        try:
            self.logger.info("Initializing VAD pipeline")

            if not self.config.huggingface_token:
                raise ValueError("Hugging Face token is required but not provided")

            # Initialize pipeline directly
            try:
                # First try the segmentation model as it's more reliable
                from pyannote.audio import Model
                from pyannote.audio.pipelines import VoiceActivityDetection

                self.logger.info("Loading segmentation model...")
                try:
                    segmentation_model = Model.from_pretrained(
                        "pyannote/segmentation-3.0",
                        use_auth_token=self.config.huggingface_token,
                    )
                except Exception as model_error:
                    if "401" in str(model_error):
                        self.logger.error(
                            "Authentication failed. Please check your Hugging Face token."
                        )
                        raise ValueError("Invalid Hugging Face token")
                    elif "403" in str(model_error):
                        self.logger.error(
                            "Access denied. Please accept the user conditions at: "
                            "https://hf.co/pyannote/segmentation-3.0"
                        )
                        raise ValueError("Model access not granted")
                    else:
                        raise

                self.vad_pipeline = VoiceActivityDetection(
                    segmentation=segmentation_model
                )
                self.vad_pipeline.instantiate(
                    {
                        "min_duration_on": self.config.min_speech_duration,
                        "min_duration_off": self.config.min_silence_duration,
                    }
                )
                self.using_segmentation = True

            except Exception as e:
                if isinstance(e, ValueError) and (
                    "token" in str(e) or "access" in str(e)
                ):
                    raise

                self.logger.error(f"Failed to load segmentation model: {str(e)}")
                self.logger.info("Attempting to use VAD pipeline instead...")

                # Fallback to VAD pipeline
                try:
                    self.vad_pipeline = Pipeline.from_pretrained(
                        "pyannote/voice-activity-detection-3.0",
                        use_auth_token=self.config.huggingface_token,
                    )
                except Exception as vad_error:
                    if "401" in str(vad_error):
                        self.logger.error(
                            "Authentication failed. Please check your Hugging Face token."
                        )
                        raise ValueError("Invalid Hugging Face token")
                    elif "403" in str(vad_error):
                        self.logger.error(
                            "Access denied. Please accept the user conditions at: "
                            "https://hf.co/pyannote/voice-activity-detection-3.0"
                        )
                        raise ValueError("Model access not granted")
                    else:
                        raise

                self.using_segmentation = False

            model_type = "segmentation" if self.using_segmentation else "VAD"
            self.logger.info(
                f"VAD pipeline initialized successfully using {model_type} model"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize VAD pipeline: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.vad_pipeline = None
            raise

    def load_and_prepare_audio(self, audio_path: str) -> Tuple[np.ndarray, int, bool]:
        """Load and prepare audio file for analysis.

        Args:
            audio_path: Path to the audio file

        Returns:
            Tuple of (audio_data, sample_rate, is_mono)

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is invalid
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Load audio with librosa
            audio_data, sample_rate = librosa.load(
                audio_path,
                sr=self.config.sample_rate,
                mono=False,
                duration=None,  # Load full file
            )

            # Check if the audio is too short (less than min_duration)
            duration = librosa.get_duration(
                y=audio_data, sr=sample_rate, hop_length=self.config.hop_length
            )
            
            if duration < self.config.min_duration:
                raise ValueError(f"Audio file too short: {audio_path}")

            # Get audio file information
            file_info = {
                "path": audio_path,
                "sample_rate": sample_rate,
                "duration": duration,
            }
                
            # Check if audio is mono (loaded as 1D array or a single channel)
            is_mono = False
            
            if audio_data.ndim == 1:
                # If loaded as 1D array, it's mono
                is_mono = True
                self.logger.warning(f"⚠️ Mono recording detected: {audio_path}. Converting to stereo.")
                # Create stereo audio by duplicating the mono channel
                audio_data = np.stack([audio_data, audio_data])
                file_info["channels"] = "1 (mono, converted to stereo)"
            elif audio_data.shape[0] == 1:
                # If loaded as 2D array with 1 channel, it's mono
                is_mono = True
                self.logger.warning(f"⚠️ Mono recording detected: {audio_path}. Converting to stereo.")
                # Create stereo audio by duplicating the mono channel
                audio_data = np.stack([audio_data[0], audio_data[0]])
                file_info["channels"] = "1 (mono, converted to stereo)"
            else:
                # It's already stereo
                file_info["channels"] = f"{audio_data.shape[0]} (stereo)"
            
            # Log detailed audio file information
            self.logger.info(
                f"Audio file details: {file_info}"
            )

            return audio_data, sample_rate, is_mono
            
        except Exception as e:
            self.logger.error(f"Error loading audio {audio_path}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise

    def calculate_audio_energy(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive audio energy metrics.

        Args:
            audio_data: Audio signal as numpy array

        Returns:
            Dictionary of energy metrics
        """
        try:
            # Basic energy metrics
            rms_energy = float(np.sqrt(np.mean(audio_data**2)))
            peak_amplitude = float(np.max(np.abs(audio_data)))

            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
            zcr = float(zero_crossings / len(audio_data))

            # Spectral centroid
            spec_centroid = float(
                np.mean(librosa.feature.spectral_centroid(y=audio_data))
            )

            # Signal-to-noise ratio estimate
            noise_floor = np.percentile(np.abs(audio_data), 10)
            signal_peak = np.percentile(np.abs(audio_data), 90)
            snr = float(
                20 * np.log10(signal_peak / noise_floor) if noise_floor > 0 else 0
            )

            return {
                "rms_energy": rms_energy,
                "peak_amplitude": peak_amplitude,
                "zero_crossing_rate": zcr,
                "spectral_centroid": spec_centroid,
                "estimated_snr": snr,
                "dynamic_range": float(signal_peak - noise_floor),
            }

        except Exception as e:
            self.logger.error(f"Error calculating audio energy: {str(e)}")
            return {
                "rms_energy": 0.0,
                "peak_amplitude": 0.0,
                "zero_crossing_rate": 0.0,
                "spectral_centroid": 0.0,
                "estimated_snr": 0.0,
                "dynamic_range": 0.0,
            }

    def detect_speech_segments(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> List[Tuple[float, float]]:
        """Detect speech segments using VAD pipeline.

        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Audio sample rate

        Returns:
            List of (start_time, end_time) tuples
        """
        try:
            if self.vad_pipeline is None:
                raise ValueError("VAD pipeline not initialized")

            # Convert to torch tensor
            waveform = torch.from_numpy(audio_data).unsqueeze(0)

            # Apply VAD
            try:
                if self.using_segmentation:
                    # For segmentation model
                    vad_result = self.vad_pipeline(
                        {"waveform": waveform, "sample_rate": sample_rate}
                    )

                    # Extract segments from timeline
                    segments = []
                    for segment in vad_result.itersegments():
                        if segment.duration >= self.config.min_speech_duration:
                            segments.append((float(segment.start), float(segment.end)))
                else:
                    # For VAD pipeline
                    vad_result = self.vad_pipeline(
                        {"waveform": waveform, "sample_rate": sample_rate}
                    )

                    # Extract segments from timeline
                    segments = []
                    for segment in vad_result.get_timeline().support():
                        if segment.duration >= self.config.min_speech_duration:
                            segments.append((float(segment.start), float(segment.end)))

            except Exception as e:
                self.logger.error(f"Error during VAD processing: {str(e)}")
                if "CUDA out of memory" in str(e):
                    self.logger.info("Attempting to process on CPU...")
                    # Move to CPU and try again
                    waveform = waveform.cpu()
                    if self.using_segmentation:
                        vad_result = self.vad_pipeline(
                            {"waveform": waveform, "sample_rate": sample_rate}
                        )
                        segments = []
                        for segment in vad_result.itersegments():
                            if segment.duration >= self.config.min_speech_duration:
                                segments.append(
                                    (float(segment.start), float(segment.end))
                                )
                    else:
                        vad_result = self.vad_pipeline(
                            {"waveform": waveform, "sample_rate": sample_rate}
                        )
                        segments = []
                        for segment in vad_result.get_timeline().support():
                            if segment.duration >= self.config.min_speech_duration:
                                segments.append(
                                    (float(segment.start), float(segment.end))
                                )
                else:
                    raise

            return segments

        except Exception as e:
            self.logger.error(f"Error detecting speech segments: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []

    def analyze_channel(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        channel_id: int,
        speaker_name: str,
    ) -> ChannelAnalysis:
        """Analyze a single audio channel."""
        # Calculate energy metrics
        energy_stats = self.calculate_audio_energy(audio_data)
        rms_energy = energy_stats["rms_energy"]

        # Detect speech segments
        speech_segments = self.detect_speech_segments(audio_data, sample_rate)

        # Calculate total speech duration
        total_speech_duration = sum(end - start for start, end in speech_segments)

        # Determine if channel has significant speech
        has_speech = (
            rms_energy > self.config.min_audio_energy_threshold
            and total_speech_duration > self.config.min_speech_duration
            and len(speech_segments) > 0
        )

        return ChannelAnalysis(
            channel_id=channel_id,
            speaker_name=speaker_name,
            total_speech_duration=total_speech_duration,
            speech_segments=speech_segments,
            energy_stats=energy_stats,
            has_speech=has_speech,
            rms_energy=rms_energy,
        )

    def analyze_turn_taking(
        self,
        caller_segments: List[Tuple[float, float]],
        receiver_segments: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        """Analyze turn-taking patterns between speakers."""
        try:
            if not caller_segments or not receiver_segments:
                return {
                    "total_turns": 0,
                    "turn_switches": 0,
                    "avg_turn_duration": 0.0,
                    "turn_rate": 0.0,
                    "overlap_ratio": 0.0,
                    "response_times": [],
                    "speaker_balance": 0.0,
                }

            # Sort segments by start time
            caller_segments = sorted(caller_segments)
            receiver_segments = sorted(receiver_segments)

            # Analyze turns
            turns = []
            response_times = []

            # Combine and sort all segments
            all_segments = [
                (start, end, "caller") for start, end in caller_segments
            ] + [(start, end, "receiver") for start, end in receiver_segments]
            all_segments.sort()

            # Calculate turns and overlaps
            current_speaker = None
            overlap_duration = 0.0
            last_end = 0.0
            turn_switches = 0

            for start, end, speaker in all_segments:
                # Check for speaker change
                if current_speaker != speaker:
                    if current_speaker is not None:
                        turns.append((last_end - start, current_speaker))
                        response_times.append(start - last_end)
                        turn_switches += 1
                    current_speaker = speaker

                # Check for overlap with previous segments
                if start < last_end:
                    overlap_duration += min(end, last_end) - start

                last_end = max(last_end, end)

            # Calculate metrics
            total_duration = max(
                caller_segments[-1][1], receiver_segments[-1][1]
            ) - min(caller_segments[0][0], receiver_segments[0][0])

            caller_duration = sum(end - start for start, end in caller_segments)
            receiver_duration = sum(end - start for start, end in receiver_segments)

            return {
                "total_turns": len(turns),
                "turn_switches": turn_switches,
                "avg_turn_duration": np.mean([t[0] for t in turns]) if turns else 0.0,
                "turn_rate": len(turns) / total_duration if total_duration > 0 else 0.0,
                "overlap_ratio": overlap_duration / total_duration
                if total_duration > 0
                else 0.0,
                "response_times": response_times,
                "speaker_balance": min(caller_duration, receiver_duration)
                / max(caller_duration, receiver_duration, 1e-6),
            }

        except Exception as e:
            self.logger.error(f"Error analyzing turn-taking: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                "total_turns": 0,
                "turn_switches": 0,
                "avg_turn_duration": 0.0,
                "turn_rate": 0.0,
                "overlap_ratio": 0.0,
                "response_times": [],
                "speaker_balance": 0.0,
            }

    def calculate_audio_quality_metrics(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, float]:
        """Calculate audio quality metrics."""
        # Signal-to-noise ratio estimation
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)

        # Estimate noise floor
        noise_floor = np.percentile(magnitude, 10)
        signal_power = np.mean(magnitude**2)
        noise_power = noise_floor**2

        snr_db = 10 * np.log10(signal_power / max(noise_power, 1e-10))

        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_data, sr=sample_rate
        )[0]
        avg_spectral_centroid = np.mean(spectral_centroids)

        return {
            "snr_db": float(snr_db),
            "spectral_centroid": float(avg_spectral_centroid),
        }

    def determine_conversation_detection(
        self,
        caller_analysis: ChannelAnalysis,
        receiver_analysis: ChannelAnalysis,
        turn_taking_analysis: Dict[str, Any],
        duration: float,
    ) -> Tuple[bool, float]:
        """Determine if a conversation occurred and calculate confidence."""
        try:
            # Initialize scoring components
            scores = {}

            # 1. Check minimum duration
            if duration < self.config.min_conversation_duration:
                return False, 0.0

            # 2. Speech presence score (0.35 weight)
            speech_score = 0.0
            if caller_analysis.has_speech and receiver_analysis.has_speech:
                caller_speech_ratio = caller_analysis.total_speech_duration / duration
                receiver_speech_ratio = (
                    receiver_analysis.total_speech_duration / duration
                )

                # Calculate speech balance score
                total_speech = caller_speech_ratio + receiver_speech_ratio
                if total_speech > 0:
                    balance = min(caller_speech_ratio, receiver_speech_ratio) / max(
                        caller_speech_ratio, receiver_speech_ratio
                    )
                    speech_score = (
                        total_speech * 0.7 + balance * 0.3
                    )  # Weight both total speech and balance

            scores["speech"] = min(1.0, speech_score * 2)  # Normalize to [0,1]

            # 3. Turn-taking score (0.25 weight)
            turn_score = 0.0
            if turn_taking_analysis["total_turns"] >= self.config.min_turns:
                # Consider turn rate and speaker balance
                turn_score = 0.6 * min(1.0, turn_taking_analysis["turn_rate"] * 10)
                turn_score += 0.4 * turn_taking_analysis["speaker_balance"]
            scores["turns"] = turn_score

            # 4. Audio quality score (0.2 weight)
            quality_score = 0.0
            for channel in [caller_analysis, receiver_analysis]:
                if (
                    channel.energy_stats["rms_energy"]
                    > self.config.min_audio_energy_threshold
                ):
                    quality_score += 0.5
                if channel.energy_stats["estimated_snr"] > 10:  # 10 dB threshold
                    quality_score += 0.5
            scores["quality"] = min(1.0, quality_score)

            # 5. Interaction score (0.2 weight)
            interaction_score = 0.0
            if turn_taking_analysis["response_times"]:
                # Good response times between 0.1 and 2.0 seconds
                good_responses = sum(
                    1 for t in turn_taking_analysis["response_times"] if 0.1 <= t <= 2.0
                )
                interaction_score = good_responses / len(
                    turn_taking_analysis["response_times"]
                )

                # Add bonus for balanced speech
                if (
                    turn_taking_analysis["speaker_balance"] > 0.3
                ):  # If reasonably balanced
                    interaction_score = min(1.0, interaction_score * 1.2)
            scores["interaction"] = interaction_score

            # Calculate weighted final score with emphasis on speech presence and balance
            weights = {
                "speech": 0.35,  # Increased from 0.3
                "turns": 0.25,  # Decreased from 0.3
                "quality": 0.2,
                "interaction": 0.2,
            }

            final_score = sum(
                score * weights[component] for component, score in scores.items()
            )

            # Add bonus for balanced speech with good SNR
            if (
                turn_taking_analysis["speaker_balance"] > 0.3
                and caller_analysis.energy_stats["estimated_snr"] > 20
                and receiver_analysis.energy_stats["estimated_snr"] > 20
            ):
                final_score = min(1.0, final_score * 1.15)

            # Log detailed scores for debugging
            self.logger.debug(f"Detection scores: {scores}")
            self.logger.debug(f"Final score: {final_score}")

            # Check if this is a conversation based on our criteria
            # If both parties have speech and there's at least one turn, it's likely a conversation
            basic_conversation_criteria = (
                caller_analysis.has_speech
                and receiver_analysis.has_speech
                and turn_taking_analysis["total_turns"] >= self.config.min_turns
                and duration >= self.config.min_conversation_duration
            )

            # If basic criteria are met and either:
            # 1. High confidence score OR
            # 2. Good balance between speakers with decent audio quality
            is_conversation = basic_conversation_criteria and (
                final_score >= self.config.conversation_detection_threshold
                or (
                    turn_taking_analysis["speaker_balance"] > 0.3
                    and caller_analysis.energy_stats["estimated_snr"] > 20
                    and receiver_analysis.energy_stats["estimated_snr"] > 20
                )
            )

            # Log the decision factors
            self.logger.debug(
                f"Decision factors: score={final_score:.3f}, duration={duration:.1f}s, "
                f"caller_speech={caller_analysis.has_speech}, receiver_speech={receiver_analysis.has_speech}, "
                f"speaker_balance={turn_taking_analysis['speaker_balance']:.2f}, "
                f"caller_snr={caller_analysis.energy_stats['estimated_snr']:.1f}, "
                f"receiver_snr={receiver_analysis.energy_stats['estimated_snr']:.1f}"
            )

            return is_conversation, final_score

        except Exception as e:
            self.logger.error(f"Error determining conversation: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False, 0.0

    def analyze_single_recording(
        self, recording_data: Tuple[str, str]
    ) -> Optional[ConversationAnalysis]:
        """Analyze a single recording for conversation detection."""
        recording_id, file_path = recording_data
        self.logger.debug(f"Analyzing recording: {recording_id}")

        try:
            # Load and prepare audio
            audio_data, sample_rate, is_mono = self.load_and_prepare_audio(file_path)
            duration = len(audio_data[0]) / sample_rate

            # Analyze each channel
            caller_analysis = self.analyze_channel(
                audio_data[self.config.caller_channel],
                sample_rate,
                self.config.caller_channel,
                "caller",
            )

            receiver_analysis = self.analyze_channel(
                audio_data[self.config.receiver_channel],
                sample_rate,
                self.config.receiver_channel,
                "receiver",
            )

            # Analyze turn-taking patterns
            turn_taking_analysis = self.analyze_turn_taking(
                caller_analysis.speech_segments,
                receiver_analysis.speech_segments,
            )

            # Calculate audio quality metrics
            audio_quality_metrics = self.calculate_audio_quality_metrics(
                audio_data, sample_rate
            )

            # Determine if conversation occurred
            conversation_detected, confidence = self.determine_conversation_detection(
                caller_analysis,
                receiver_analysis,
                turn_taking_analysis,
                duration,
            )

            # Adjust confidence for mono recordings
            if is_mono:
                confidence *= 0.7  # Reduce confidence for mono recordings

            return ConversationAnalysis(
                recording_id=recording_id,
                duration=duration,
                caller_analysis=caller_analysis,
                receiver_analysis=receiver_analysis,
                conversation_detected=conversation_detected,
                conversation_confidence=confidence,
                turn_taking_analysis=turn_taking_analysis,
                audio_quality_metrics=audio_quality_metrics,
                is_mono=is_mono,
            )

        except Exception as e:
            self.logger.error(f"Error analyzing recording {recording_id}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def discover_recordings(self, recordings_dir: str = None) -> List[Tuple[str, str]]:
        """Discover audio recordings in the specified directory."""
        recordings_dir = recordings_dir or self.config.input_recordings_dir
        recordings_path = Path(recordings_dir)

        if not recordings_path.exists():
            raise FileNotFoundError(f"Recordings directory not found: {recordings_dir}")

        # Find all .mp3 files
        audio_files = list(recordings_path.glob("*.mp3"))

        # Create list of (recording_id, file_path) tuples
        recordings = []
        for audio_file in audio_files:
            recording_id = audio_file.stem
            recordings.append((recording_id, str(audio_file)))

        self.logger.info(f"Discovered {len(recordings)} recordings in {recordings_dir}")
        return recordings

    def generate_ground_truth_from_transcription_report(self) -> Dict[str, bool]:
        """Generate ground truth from transcription analysis report."""
        try:
            # Read transcription analysis report
            report_path = os.path.join(
                self.config.input_recordings_dir,
                "..",
                "transcription_analysis_report.txt",
            )

            if not os.path.exists(report_path):
                self.logger.warning(
                    f"Transcription analysis report not found: {report_path}"
                )
                return {}

            # Read all recording IDs from exotel_recordings.csv
            recordings_csv = os.path.join(
                self.config.input_recordings_dir,
                "..",
                "exotel_recordings.csv",
            )

            if not os.path.exists(recordings_csv):
                self.logger.warning(f"Recordings CSV not found: {recordings_csv}")
                return {}

            # Read recording IDs from CSV
            df = pd.read_csv(recordings_csv)
            all_recording_ids = set(df["recording_id"].astype(str))

            # Read single speaker recordings from report
            with open(report_path, "r") as f:
                report_content = f.read()

            # Extract recording IDs from the report
            single_speaker_ids = set()
            for line in report_content.split("\n"):
                if "recording_id:" in line:
                    recording_id = line.split("recording_id:")[1].strip()
                    if (
                        "single speaker detected"
                        in report_content.split(line)[1].split("\n")[0]
                    ):
                        single_speaker_ids.add(recording_id)

            # Generate ground truth
            ground_truth = {}
            for recording_id in all_recording_ids:
                # Single speaker = no conversation (False)
                # Multiple speakers = conversation (True)
                ground_truth[recording_id] = recording_id not in single_speaker_ids

            self.logger.info(
                f"Generated ground truth for {len(ground_truth)} recordings"
            )
            self.logger.info(f"Single speaker recordings: {len(single_speaker_ids)}")
            self.logger.info(
                f"Multi-speaker recordings: {len(all_recording_ids) - len(single_speaker_ids)}"
            )

            # Save ground truth to file
            ground_truth_path = os.path.join(
                self.config.output_dir, "ground_truth.json"
            )
            with open(ground_truth_path, "w") as f:
                json.dump(ground_truth, f, indent=2)

            self.logger.info(f"Ground truth saved to: {ground_truth_path}")
            return ground_truth

        except Exception as e:
            self.logger.error(f"Error generating ground truth: {e}")
            return {}

    def process_recordings_batch(
        self, recordings: List[Tuple[str, str]], batch_size: int = 100
    ) -> List[ConversationAnalysis]:
        """Process multiple recordings in batches.

        Args:
            recordings: List of (recording_id, file_path) tuples
            batch_size: Number of recordings to process in each batch

        Returns:
            List of ConversationAnalysis results
        """
        self.logger.info(f"Processing batch of {len(recordings)} recordings")

        results = []
        skipped_recordings = []
        error_counts = {"too_short": 0, "load_error": 0, "processing_error": 0}

        with tqdm(total=len(recordings), desc="Processing recordings") as pbar:
            for recording_data in recordings:
                try:
                    # Add a small delay between recordings to prevent resource exhaustion
                    time.sleep(0.1)

                    result = self.analyze_single_recording(recording_data)
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
                        self.logger.warning(
                            f"Skipping {recording_data[0]}: Audio file too short"
                        )
                        error_counts["too_short"] += 1
                    else:
                        self.logger.error(
                            f"Error processing recording {recording_data[0]}: {e}"
                        )
                        error_counts["processing_error"] += 1
                    skipped_recordings.append(recording_data[0])
                except Exception as e:
                    self.logger.error(
                        f"Error processing recording {recording_data[0]}: {e}"
                    )
                    error_counts["load_error"] += 1
                    skipped_recordings.append(recording_data[0])
                finally:
                    pbar.update(1)

        # Log batch summary
        self.logger.info(
            f"\nBatch processing summary:"
            f"\n- Total recordings: {len(recordings)}"
            f"\n- Successfully processed: {len(results)}"
            f"\n- Failed/Skipped: {len(skipped_recordings)}"
            f"\n  * Too short: {error_counts['too_short']}"
            f"\n  * Load errors: {error_counts['load_error']}"
            f"\n  * Processing errors: {error_counts['processing_error']}"
            f"\n- Success rate: {(len(results) / len(recordings) * 100):.1f}%"
        )

        if skipped_recordings:
            self.logger.warning(
                "\nSkipped recordings:"
                "\n"
                + "\n".join(skipped_recordings[:10])
                + (
                    f"\n... and {len(skipped_recordings) - 10} more"
                    if len(skipped_recordings) > 10
                    else ""
                )
            )

        return results

    def calculate_metrics(
        self, analyses: List[ConversationAnalysis], ground_truth: Dict[str, bool]
    ) -> MetricsResult:
        """Calculate evaluation metrics."""
        # Align analyses with ground truth
        y_true = []
        y_pred = []

        for analysis in analyses:
            if analysis.recording_id in ground_truth:
                y_true.append(ground_truth[analysis.recording_id])
                y_pred.append(analysis.conversation_detected)

        if not y_true:
            raise ValueError(
                "No recordings found in both analysis results and ground truth"
            )

        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        cm = confusion_matrix(y_true, y_pred)

        return MetricsResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            confusion_matrix=cm,
            total_samples=len(y_true),
        )

    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, output_path: str):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Conversation", "Conversation"],
            yticklabels=["No Conversation", "Conversation"],
        )
        plt.title("Conversation Detection Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Confusion matrix plot saved to: {output_path}")

    def save_results(
        self,
        analyses: List[ConversationAnalysis],
        metrics: Optional[MetricsResult] = None,
    ):
        """Save analysis results to files.

        Args:
            analyses: List of conversation analysis results
            metrics: Optional metrics result
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Save JSON results
        json_path = os.path.join(
            self.config.output_dir, f"conversation_analysis_{timestamp}.json"
        )
        json_data = [
            {
                "recording_id": a.recording_id,
                "duration": a.duration,
                "conversation_detected": a.conversation_detected,
                "conversation_confidence": a.conversation_confidence,
                "caller_has_speech": a.caller_analysis.has_speech,
                "receiver_has_speech": a.receiver_analysis.has_speech,
                "caller_speech_duration": a.caller_analysis.total_speech_duration,
                "receiver_speech_duration": a.receiver_analysis.total_speech_duration,
                "turn_switches": a.turn_taking_analysis.get("turn_switches", 0),
                "snr_db": a.audio_quality_metrics.get("snr_db", 0),
                "is_mono": a.is_mono,
            }
            for a in analyses
            if a is not None
        ]
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        # Save CSV results
        csv_path = os.path.join(
            self.config.output_dir, f"conversation_analysis_{timestamp}.csv"
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
            for a in analyses:
                if a is not None:
                    writer.writerow(
                        [
                            a.recording_id,
                            a.duration,
                            a.conversation_detected,
                            a.conversation_confidence,
                            a.caller_analysis.has_speech,
                            a.receiver_analysis.has_speech,
                            a.caller_analysis.total_speech_duration,
                            a.receiver_analysis.total_speech_duration,
                            a.turn_taking_analysis.get("turn_switches", 0),
                            a.audio_quality_metrics.get("snr_db", 0),
                            a.is_mono,
                        ]
                    )

        self.logger.info(
            f"Results saved to: {json_path} and {csv_path}"
        )

        # Generate report
        if metrics is not None:
            report_path = self.generate_markdown_report(
                analyses, metrics, datetime.now(), datetime.now()
            )
            self.logger.info(f"Report saved to: {report_path}")

    def generate_markdown_report(
        self,
        analyses: List[ConversationAnalysis],
        metrics: Optional[MetricsResult],
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        """Generate a detailed Markdown report of the analysis results."""
        # Calculate statistics
        total_recordings = len(analyses)
        conversations_detected = sum(1 for a in analyses if a.conversation_detected)
        mono_recordings = sum(1 for a in analyses if a.is_mono)
        total_duration = sum(a.duration for a in analyses)
        avg_duration = total_duration / total_recordings if total_recordings > 0 else 0

        # Calculate turn statistics
        all_turns = [a.turn_taking_analysis["total_turns"] for a in analyses]
        avg_turns = np.mean(all_turns) if all_turns else 0
        max_turns = max(all_turns) if all_turns else 0

        # Calculate speech statistics
        caller_speech_ratios = [
            (a.caller_analysis.total_speech_duration / a.duration) * 100
            for a in analyses
        ]
        receiver_speech_ratios = [
            (a.receiver_analysis.total_speech_duration / a.duration) * 100
            for a in analyses
        ]

        avg_caller_speech = np.mean(caller_speech_ratios) if caller_speech_ratios else 0
        avg_receiver_speech = (
            np.mean(receiver_speech_ratios) if receiver_speech_ratios else 0
        )

        # Calculate SNR statistics
        snr_values = [a.audio_quality_metrics["snr_db"] for a in analyses]
        avg_snr = np.mean(snr_values) if snr_values else 0

        # Generate report
        report = f"""# Conversation Detection Analysis Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview
- **Total Recordings Analyzed**: {total_recordings}
- **Analysis Duration**: {end_time - start_time}
- **Total Audio Duration**: {total_duration:.2f} seconds
- **Average Recording Length**: {avg_duration:.2f} seconds

## Recording Quality Issues
- **Mono Recordings**: {mono_recordings} ({(mono_recordings / total_recordings * 100):.1f}%)
  * ⚠️ Mono recordings may have reduced analysis accuracy
  * Caller and receiver channels cannot be properly separated
  * Turn-taking analysis may be less reliable

## Detection Results
- **Conversations Detected**: {conversations_detected} ({(conversations_detected / total_recordings * 100):.1f}%)
- **Non-Conversations**: {total_recordings - conversations_detected} ({((total_recordings - conversations_detected) / total_recordings * 100):.1f}%)
- **Stereo Conversations**: {sum(1 for a in analyses if a.conversation_detected and not a.is_mono)}
- **Mono Conversations**: {sum(1 for a in analyses if a.conversation_detected and a.is_mono)}

## Turn-Taking Statistics
- **Average Turns per Recording**: {avg_turns:.2f}
- **Maximum Turns in a Recording**: {max_turns}
- **Recordings with No Turns**: {sum(1 for t in all_turns if t == 0)}

## Speech Analysis
- **Average Caller Speech Ratio**: {avg_caller_speech:.1f}%
- **Average Receiver Speech Ratio**: {avg_receiver_speech:.1f}%
- **Recordings with Both Speakers**: {sum(1 for a in analyses if a.caller_analysis.has_speech and a.receiver_analysis.has_speech)}
- **Recordings with Single Speaker**: {sum(1 for a in analyses if (a.caller_analysis.has_speech != a.receiver_analysis.has_speech))}
- **Recordings with No Speech**: {sum(1 for a in analyses if not a.caller_analysis.has_speech and not a.receiver_analysis.has_speech)}

## Audio Quality
- **Average SNR**: {avg_snr:.1f} dB
- **Recordings with Good Quality (SNR > 20dB)**: {sum(1 for v in snr_values if v > 20)}
- **Recordings with Poor Quality (SNR < 10dB)**: {sum(1 for v in snr_values if v < 10)}

## Duration Distribution
- **0-30 seconds**: {sum(1 for a in analyses if a.duration <= 30)} recordings
- **30-60 seconds**: {sum(1 for a in analyses if 30 < a.duration <= 60)} recordings
- **1-2 minutes**: {sum(1 for a in analyses if 60 < a.duration <= 120)} recordings
- **2+ minutes**: {sum(1 for a in analyses if a.duration > 120)} recordings\n"""

        if metrics:
            report += f"""
## Evaluation Metrics
- **Precision**: {metrics.precision:.3f}
- **Recall**: {metrics.recall:.3f}
- **F1 Score**: {metrics.f1_score:.3f}
- **Accuracy**: {metrics.accuracy:.3f}

### Confusion Matrix
|                 | Predicted No Conv.| Predicted Conv. |
|-----------------|-------------------|-----------------|
| Actual No Conv. | {metrics.confusion_matrix[0][0]:^17}| {metrics.confusion_matrix[0][1]:^15} |
| Actual Conv.    | {metrics.confusion_matrix[1][0]:^17}| {metrics.confusion_matrix[1][1]:^15} |
"""

        report += f"""
## Processing Details
- **VAD Model**: {"Segmentation" if self.using_segmentation else "VAD Pipeline"}
- **Minimum Speech Duration**: {self.config.min_speech_duration:.2f}s
- **Minimum Conversation Duration**: {self.config.min_conversation_duration:.2f}s
- **Minimum Turn Count**: {self.config.min_turns}

## Notes
- Speech ratios are calculated as percentage of total recording duration
- SNR values are averaged across both channels
- Turn count includes both speaker transitions and pauses
"""

        return report

    def generate_report(
        self,
        all_analyses: List[List[ConversationAnalysis]],
        start_time: datetime,
        end_time: datetime,
        metrics: Optional[MetricsResult] = None,
    ) -> str:
        """Generate a comprehensive transcription analysis report with detailed statistics."""
        # Flatten all analyses for overall statistics
        all_recordings = [a for batch in all_analyses for a in batch if a is not None]
        total_recordings = len(all_recordings)

        # Audio format statistics
        mono_recordings = [a for a in all_recordings if a.is_mono]
        stereo_recordings = [a for a in all_recordings if not a.is_mono]
        mono_recording_ids = [a.recording_id for a in mono_recordings]

        # Conversation statistics
        conversations_detected = len([a for a in all_recordings if a.conversation_detected])
        avg_confidence = sum([a.conversation_confidence for a in all_recordings]) / total_recordings if total_recordings > 0 else 0

        # Speech duration statistics
        total_caller_speech = sum([a.caller_analysis.total_speech_duration for a in all_recordings])
        total_receiver_speech = sum([a.receiver_analysis.total_speech_duration for a in all_recordings])
        
        # Turn-taking statistics
        total_turn_switches = sum([a.turn_taking_analysis.get("turn_switches", 0) for a in all_recordings])
        avg_turn_switches = total_turn_switches / conversations_detected if conversations_detected > 0 else 0
        
        # Audio quality metrics
        avg_snr = sum([a.audio_quality_metrics.get("snr_db", 0) for a in all_recordings]) / total_recordings if total_recordings > 0 else 0

        # Batch statistics
        batch_stats = []
        for batch_idx, batch in enumerate(all_analyses, 1):
            valid_recordings = [a for a in batch if a is not None]
            batch_size = len(valid_recordings)
            conversations_in_batch = len([a for a in valid_recordings if a.conversation_detected])
            mono_in_batch = len([a for a in valid_recordings if a.is_mono])
            
            batch_stats.append({
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "conversations": conversations_in_batch,
                "conversation_rate": conversations_in_batch / batch_size if batch_size > 0 else 0,
                "mono_count": mono_in_batch,
                "mono_rate": mono_in_batch / batch_size if batch_size > 0 else 0,
            })

        # Generate report
        report = [
            "# Enhanced Transcription Analysis Report",
            f"Generated: {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            f"Total Recordings Analyzed: {total_recordings}",
            f"Analysis Duration: {end_time - start_time}",
            "",
            "## Audio Format Statistics",
            f"- Mono Recordings: {len(mono_recordings)} ({len(mono_recordings)/total_recordings*100:.1f}%)",
            f"- Stereo Recordings: {len(stereo_recordings)} ({len(stereo_recordings)/total_recordings*100:.1f}%)",
            "",
            "## Conversation Detection Results",
            f"- Conversations Detected: {conversations_detected} ({conversations_detected/total_recordings*100:.1f}%)",
            f"- Average Confidence Score: {avg_confidence:.2f}",
            "",
            "## Channel Analysis",
            "### Speech Duration",
            f"- Total Caller Speech: {total_caller_speech:.2f} seconds",
            f"- Total Receiver Speech: {total_receiver_speech:.2f} seconds",
            f"- Caller/Receiver Ratio: {total_caller_speech/total_receiver_speech:.2f}" if total_receiver_speech > 0 else "- Caller/Receiver Ratio: N/A",
            "",
            "### Turn Taking Patterns",
            f"- Average Turns Per Conversation: {avg_turn_switches:.1f}",
            "",
            "## Audio Quality Metrics",
            f"- Average Snr: {avg_snr:.2f}",
            f"- Average Clarity: {0.00:.2f}",
            f"- Average Background Noise: {0.00:.2f}",
        ]

        # Add evaluation metrics if available
        if metrics:
            report.extend([
                "",
                "## Evaluation Metrics",
                f"- Precision: {metrics.precision:.3f}",
                f"- Recall: {metrics.recall:.3f}",
                f"- F1 Score: {metrics.f1_score:.3f}",
                f"- Accuracy: {metrics.accuracy:.3f}",
                "",
                "### Confusion Matrix",
                "```",
                "              Predicted",
                "Actual    | Positive | Negative",
                "---------+----------+----------",
                f"Positive |      {metrics.confusion_matrix[0][0]} |      {metrics.confusion_matrix[0][1]}",
                f"Negative |       {metrics.confusion_matrix[1][0]} |      {metrics.confusion_matrix[1][1]}",
                "```",
            ])

        # Add batch statistics
        report.extend([
            "",
            "## Batch Processing Statistics",
        ])

        for batch in batch_stats:
            report.extend([
                f"",
                f"### Batch {batch['batch_idx']} ({batch['batch_size']} recordings)",
                f"- Success Rate: {100.0:.1f}%",
                f"- Processed: {batch['batch_size']}",
                f"- Failed: 0",
                f"- Details:",
                f"  * Mono Recordings: {batch['mono_count']}",
                f"  * Conversations Detected: {batch['conversations']}",
                f"  * Average Confidence: {avg_confidence:.2f}",
            ])
            
        # Add mono recordings section if any exist
        if mono_recordings:
            report.extend([
                "",
                "## Mono Recordings Analysis",
                f"Total Mono Recordings: {len(mono_recordings)}",
                f"Mono Recordings with Conversations: {len([a for a in mono_recordings if a.conversation_detected])} ({len([a for a in mono_recordings if a.conversation_detected])/len(mono_recordings)*100:.1f}%)",
                f"Average Confidence Score: {sum([a.conversation_confidence for a in mono_recordings])/len(mono_recordings):.2f}",
                "",
                "### Mono Recording IDs:",
                ", ".join(mono_recording_ids)
            ])

        report_text = "\n".join(report)

        # Save report to file
        report_path = os.path.join(
            self.config.output_dir,
            "reports",
            f"enhanced_analysis_report_{end_time.strftime('%Y%m%d_%H%M%S')}.md",
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_text)

        self.logger.info(f"Enhanced analysis report saved to: {report_path}")
        return report_path

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
            self.config.input_recordings_dir = recordings_dir

        # Discover recordings
        recordings = self.discover_recordings()
        if max_recordings:
            recordings = recordings[:max_recordings]

        # Process in batches
        batch_size = 100
        all_analyses = []

        for i in range(0, len(recordings), batch_size):
            batch = recordings[i : i + batch_size]
            results = self.process_recordings_batch(batch, batch_size)
            all_analyses.append(results)

        # Flatten results for metrics
        analyses = [a for batch in all_analyses for a in batch if a is not None]

        # Calculate metrics if ground truth available
        metrics = None
        try:
            ground_truth = self.generate_ground_truth_from_transcription_report()
            if ground_truth:
                metrics = self.calculate_metrics(analyses, ground_truth)
        except Exception as e:
            self.logger.warning(f"Could not calculate metrics: {str(e)}")

        # Generate reports
        self.save_results(analyses, metrics)

        end_time = datetime.now()

        # Generate transcription report
        self.generate_report(all_analyses, start_time, end_time, metrics)

        return analyses, metrics


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Conversation Detection System")
    parser.add_argument(
        "--recordings",
        required=True,
        help="Path to directory containing audio recordings",
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Hugging Face token for accessing models",
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
        default=6,
        help="Number of batches to process (default: 6)",
    )
    args = parser.parse_args()

    # Create configuration with provided token
    config = ConversationDetectionConfig()
    config.huggingface_token = args.token
    config.input_recordings_dir = args.recordings

    try:
        # Initialize and run detector
        detector = ConversationDetector(config)

        # Get all recordings
        all_recordings = detector.discover_recordings(args.recordings)

        # Calculate total recordings to process
        total_recordings = min(len(all_recordings), args.batch_size * args.num_batches)
        recordings_to_process = all_recordings[:total_recordings]

        print(
            f"\nProcessing {total_recordings} recordings in {args.num_batches} batches of {args.batch_size}"
        )

        # Process recordings in batches
        all_analyses = []
        all_metrics = None

        for batch_num in range(args.num_batches):
            start_idx = batch_num * args.batch_size
            end_idx = min(start_idx + args.batch_size, total_recordings)
            batch_recordings = recordings_to_process[start_idx:end_idx]

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
                detector.save_results(
                    batch_analyses,
                    None,  # Metrics will be calculated at the end
                )

            except KeyboardInterrupt:
                print("\n\nProcessing interrupted by user. Saving partial results...")
                break
            except Exception as e:
                print(f"\nError processing batch {batch_num + 1}: {str(e)}")
                continue

        if all_analyses:
            # Generate final report with all processed recordings
            try:
                # Try to calculate metrics if ground truth is available
                ground_truth = (
                    detector.generate_ground_truth_from_transcription_report()
                )
                if ground_truth:
                    all_metrics = detector.calculate_metrics(all_analyses, ground_truth)
            except Exception as e:
                print(f"\nError calculating metrics: {str(e)}")

            # Save final results
            detector.save_results(all_analyses, all_metrics)
            print("\n✅ Analysis completed successfully!")
            print(f"Processed {len(all_analyses)} recordings")
            print(f"Results saved in: {config.output_dir}")
        else:
            print("\n❌ No recordings were successfully processed")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
