import os
import sys
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from pyannote.audio import Pipeline
import torch

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
    """

    recording_id: str
    duration: float
    caller_analysis: ChannelAnalysis
    receiver_analysis: ChannelAnalysis
    conversation_detected: bool
    conversation_confidence: float
    turn_taking_analysis: Dict[str, Any]
    audio_quality_metrics: Dict[str, float]


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

    def load_and_prepare_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and prepare audio file for analysis.

        Args:
            audio_path: Path to the audio file

        Returns:
            Tuple of (audio_data, sample_rate)

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
            )

            # Validate audio data
            if audio_data is None:
                raise ValueError(f"Failed to load audio data from {audio_path}")

            # Handle different channel configurations
            if audio_data.ndim == 1:
                self.logger.warning(f"Converting mono to stereo: {audio_path}")
                audio_data = np.stack([audio_data, audio_data])
            elif audio_data.shape[0] > 2:
                self.logger.warning(f"Using first 2 channels from {audio_path}")
                audio_data = audio_data[:2]

            # Validate audio length
            if len(audio_data[0]) < sample_rate:  # Less than 1 second
                raise ValueError(f"Audio file too short: {audio_path}")

            return audio_data, sample_rate

        except Exception as e:
            self.logger.error(f"Error loading audio {audio_path}: {str(e)}")
            self.logger.error(traceback.format_exc())
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
            is_conversation = (
                basic_conversation_criteria
                and (
                    final_score >= self.config.conversation_detection_threshold
                    or (
                        turn_taking_analysis["speaker_balance"] > 0.3
                        and caller_analysis.energy_stats["estimated_snr"] > 20
                        and receiver_analysis.energy_stats["estimated_snr"] > 20
                    )
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
        """Analyze a single recording for conversation detection.

        Args:
            recording_data: Tuple of (recording_id, file_path)

        Returns:
            ConversationAnalysis object if successful, None if failed
        """
        recording_id, file_path = recording_data
        self.logger.debug(f"Analyzing recording: {recording_id}")

        try:
            # Load and prepare audio
            audio_data, sample_rate = self.load_and_prepare_audio(file_path)
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

            return ConversationAnalysis(
                recording_id=recording_id,
                duration=duration,
                caller_analysis=caller_analysis,
                receiver_analysis=receiver_analysis,
                conversation_detected=conversation_detected,
                conversation_confidence=confidence,
                turn_taking_analysis=turn_taking_analysis,
                audio_quality_metrics=audio_quality_metrics,
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
        self, recordings: List[Tuple[str, str]]
    ) -> List[ConversationAnalysis]:
        """Process multiple recordings in parallel."""
        self.logger.info(
            f"Processing {len(recordings)} recordings with {self.config.max_workers} workers"
        )

        results = []
        failed_recordings = []

        with tqdm(total=len(recordings), desc="Processing recordings") as pbar:
            for recording_data in recordings:
                try:
                    result = self.analyze_single_recording(recording_data)
                    if result:
                        results.append(result)
                    else:
                        failed_recordings.append(recording_data[0])
                except Exception as e:
                    self.logger.error(
                        f"Error processing recording {recording_data[0]}: {e}"
                    )
                    failed_recordings.append(recording_data[0])
                pbar.update(1)

        if failed_recordings:
            self.logger.warning(
                f"Failed to process {len(failed_recordings)} recordings"
            )

        self.logger.info(
            f"Successfully processed {len(results)} out of {len(recordings)} recordings"
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
        """Save analysis results and metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_data = []
        for analysis in analyses:
            result = {
                "recording_id": analysis.recording_id,
                "duration": analysis.duration,
                "conversation_detected": analysis.conversation_detected,
                "conversation_confidence": analysis.conversation_confidence,
                "caller_has_speech": analysis.caller_analysis.has_speech,
                "receiver_has_speech": analysis.receiver_analysis.has_speech,
                "caller_speech_duration": analysis.caller_analysis.total_speech_duration,
                "receiver_speech_duration": analysis.receiver_analysis.total_speech_duration,
                "turn_switches": analysis.turn_taking_analysis["turn_switches"],
                "snr_db": analysis.audio_quality_metrics["snr_db"],
            }
            results_data.append(result)

        # Save as JSON
        json_path = os.path.join(
            self.config.output_dir, f"conversation_analysis_{timestamp}.json"
        )
        with open(json_path, "w") as f:
            json.dump(results_data, f, indent=2)

        # Save as CSV
        csv_path = os.path.join(
            self.config.output_dir, f"conversation_analysis_{timestamp}.csv"
        )
        df = pd.DataFrame(results_data)
        df.to_csv(csv_path, index=False)

        self.logger.info(f"Results saved to: {json_path} and {csv_path}")

        # Save confusion matrix plot if metrics are available
        if metrics:
            plot_path = os.path.join(
                self.config.output_dir,
                "plots",
                f"confusion_matrix_{timestamp}.png",
            )
            self.plot_confusion_matrix(metrics.confusion_matrix, plot_path)

    def run_analysis(
        self, recordings_dir: str = None, max_recordings: int = None
    ) -> Tuple[List[ConversationAnalysis], Optional[MetricsResult]]:
        """Run the complete conversation detection analysis."""
        self.logger.info("Starting conversation detection analysis")
        start_time = datetime.now()

        try:
            # Discover recordings
            recordings = self.discover_recordings(recordings_dir)

            # Limit recordings if specified
            if max_recordings and len(recordings) > max_recordings:
                recordings = recordings[:max_recordings]
                self.logger.info(
                    f"Limited analysis to first {max_recordings} recordings"
                )

            # Generate ground truth from transcription report
            ground_truth = self.generate_ground_truth_from_transcription_report()

            # Process recordings
            analyses = self.process_recordings_batch(recordings)

            if not analyses:
                raise ValueError("No recordings were successfully processed")

            # Calculate metrics if ground truth is available
            metrics = None
            if ground_truth:
                try:
                    metrics = self.calculate_metrics(analyses, ground_truth)
                    self.logger.info("Evaluation completed successfully")
                except Exception as e:
                    self.logger.error(f"Error during evaluation: {e}")

            # Save results
            self.save_results(analyses, metrics)

            # Log summary
            end_time = datetime.now()
            duration = end_time - start_time
            conversation_count = sum(1 for a in analyses if a.conversation_detected)

            self.logger.info(f"Analysis completed in {duration}")
            self.logger.info(f"Processed {len(analyses)} recordings")
            self.logger.info(
                f"Detected conversations in {conversation_count} recordings ({conversation_count / len(analyses) * 100:.1f}%)"
            )

            if metrics:
                self.logger.info(
                    f"Evaluation metrics - Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}, F1: {metrics.f1_score:.3f}"
                )

            return analyses, metrics

        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            self.logger.debug(traceback.format_exc())
            raise
