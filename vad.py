"""Audio processing for conversation detection."""

import os
import logging
import traceback
from typing import Dict, List, Tuple
import numpy as np
import librosa
import torch
from pyannote.audio import Pipeline
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

from config import ConversationDetectionConfig
from models import ChannelAnalysis


class AudioProcessor:
    """Audio processing for conversation detection."""

    def __init__(self, config: ConversationDetectionConfig):
        """Initialize audio processor.

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.vad_pipeline = None
        self.using_segmentation = False
        self.initialize_vad_pipeline()

    def initialize_vad_pipeline(self) -> None:
        """Initialize the Voice Activity Detection pipeline."""
        try:
            self.logger.info("Initializing VAD pipeline")

            if not self.config.huggingface_token:
                raise ValueError("Hugging Face token is required but not provided")

            # First try the segmentation model as it's more reliable
            try:
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
                        "min_duration_on": self.config.audio.min_speech_duration,
                        "min_duration_off": self.config.audio.min_silence_duration,
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
                sr=self.config.audio.sample_rate,
                mono=False,
                duration=None,  # Load full file
            )

            # Check if the audio is too short (less than min_duration)
            duration = librosa.get_duration(
                y=audio_data, sr=sample_rate, hop_length=self.config.audio.hop_length
            )

            if duration < self.config.audio.min_duration:
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
                self.logger.warning(
                    f"⚠️ Mono recording detected: {audio_path}. Converting to stereo."
                )
                # Create stereo audio by duplicating the mono channel
                audio_data = np.stack([audio_data, audio_data])
                file_info["channels"] = "1 (mono, converted to stereo)"
            elif audio_data.shape[0] == 1:
                # If loaded as 2D array with 1 channel, it's mono
                is_mono = True
                self.logger.warning(
                    f"⚠️ Mono recording detected: {audio_path}. Converting to stereo."
                )
                # Create stereo audio by duplicating the mono channel
                audio_data = np.stack([audio_data[0], audio_data[0]])
                file_info["channels"] = "1 (mono, converted to stereo)"
            else:
                # It's already stereo
                file_info["channels"] = f"{audio_data.shape[0]} (stereo)"

            # Log detailed audio file information
            self.logger.info(f"Audio file details: {file_info}")

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

            # Determine device based on configuration
            device = (
                "cuda"
                if self.config.processing.use_gpu and torch.cuda.is_available()
                else "cpu"
            )

            # Apply VAD
            try:
                # Move tensor to appropriate device
                waveform = waveform.to(device)

                if self.using_segmentation:
                    # For segmentation model
                    vad_result = self.vad_pipeline(
                        {"waveform": waveform, "sample_rate": sample_rate}
                    )

                    # Extract segments from timeline
                    segments = []
                    for segment in vad_result.itersegments():
                        if segment.duration >= self.config.audio.min_speech_duration:
                            segments.append((float(segment.start), float(segment.end)))
                else:
                    # For VAD pipeline
                    vad_result = self.vad_pipeline(
                        {"waveform": waveform, "sample_rate": sample_rate}
                    )

                    # Extract segments from timeline
                    segments = []
                    for segment in vad_result.get_timeline().support():
                        if segment.duration >= self.config.audio.min_speech_duration:
                            segments.append((float(segment.start), float(segment.end)))

            except Exception as e:
                self.logger.error(f"Error during VAD processing: {str(e)}")
                if "CUDA out of memory" in str(e):
                    self.logger.info(
                        "CUDA out of memory. Attempting to process on CPU..."
                    )
                    # Move to CPU and try again
                    waveform = waveform.cpu()
                    if self.using_segmentation:
                        vad_result = self.vad_pipeline(
                            {"waveform": waveform, "sample_rate": sample_rate}
                        )
                        segments = []
                        for segment in vad_result.itersegments():
                            if (
                                segment.duration
                                >= self.config.audio.min_speech_duration
                            ):
                                segments.append(
                                    (float(segment.start), float(segment.end))
                                )
                    else:
                        vad_result = self.vad_pipeline(
                            {"waveform": waveform, "sample_rate": sample_rate}
                        )
                        segments = []
                        for segment in vad_result.get_timeline().support():
                            if (
                                segment.duration
                                >= self.config.audio.min_speech_duration
                            ):
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
        """Analyze a single audio channel.

        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Audio sample rate
            channel_id: Channel identifier (0=left, 1=right)
            speaker_name: Name of the speaker (caller/receiver)

        Returns:
            ChannelAnalysis object with results
        """
        # Calculate energy metrics
        energy_stats = self.calculate_audio_energy(audio_data)
        rms_energy = energy_stats["rms_energy"]

        # Detect speech segments
        speech_segments = self.detect_speech_segments(audio_data, sample_rate)

        # Calculate total speech duration
        total_speech_duration = sum(end - start for start, end in speech_segments)

        # Determine if channel has significant speech
        has_speech = (
            rms_energy > self.config.detection.min_audio_energy_threshold
            and total_speech_duration > self.config.audio.min_speech_duration
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

    def calculate_audio_quality_metrics(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, float]:
        """Calculate audio quality metrics.

        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Audio sample rate

        Returns:
            Dictionary of audio quality metrics
        """
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
