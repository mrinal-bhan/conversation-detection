"""Configuration for conversation detection analysis."""

from dataclasses import dataclass, field, asdict
import os
import json
from typing import Optional, Dict, Any


@dataclass
class AudioConfig:
    """Audio processing configuration.

    Attributes:
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels (2 for stereo)
        hop_length: Number of samples between successive frames for librosa
        min_duration: Minimum duration for an audio file to be processed in seconds
        min_speech_duration: Minimum duration for a speech segment in seconds
        min_silence_duration: Minimum silence duration between segments in seconds
        caller_channel: Channel index for caller audio (0 = left)
        receiver_channel: Channel index for receiver audio (1 = right)
    """

    sample_rate: int = 16000
    channels: int = 2
    hop_length: int = 512
    min_duration: float = 1
    min_speech_duration: float = 0.15
    min_silence_duration: float = 0.2
    caller_channel: int = 0
    receiver_channel: int = 1


@dataclass
class ConversationDetectionParams:
    """Parameters for conversation detection algorithm.

    Attributes:
        min_conversation_duration: Minimum total conversation duration in seconds
        min_turns: Minimum number of speaker turns for valid conversation
        min_audio_energy_threshold: Minimum RMS energy threshold for valid audio
        conversation_detection_threshold: Confidence threshold for conversation detection
        speech_presence_weight: Weight for speech presence in confidence calculation
        turn_taking_weight: Weight for turn-taking in confidence calculation
        audio_quality_weight: Weight for audio quality in confidence calculation
        interaction_weight: Weight for interaction patterns in confidence calculation
    """

    min_conversation_duration: float = 2.0
    min_turns: int = 2
    min_audio_energy_threshold: float = 0.003
    conversation_detection_threshold: float = 0.40
    speech_presence_weight: float = 0.35
    turn_taking_weight: float = 0.25
    audio_quality_weight: float = 0.20
    interaction_weight: float = 0.20


@dataclass
class FilePaths:
    """File paths configuration.

    Attributes:
        input_recordings_dir: Directory containing input recordings
        output_dir: Directory for analysis results
        logs_dir: Directory for log files
        temp_dir: Directory for temporary files
    """

    input_recordings_dir: str = os.path.expanduser("~/myEqual/Azure/prod_recordings")
    output_dir: str = os.path.expanduser(
        "~/myEqual/Azure/conversation-detection/results/final_results"
    )
    logs_dir: str = "logs"
    temp_dir: str = "temp"


@dataclass
class ProcessingConfig:
    """Processing configuration.

    Attributes:
        batch_size: Number of recordings to process in a batch
        max_workers: Maximum number of parallel workers
        use_gpu: Whether to use GPU for processing
        device: Device to use for processing ('cuda', 'cpu', or 'auto')
    """

    batch_size: int = 16
    max_workers: int = 8
    use_gpu: bool = True
    device: str = "auto"  # 'cuda', 'cpu', or 'auto'


@dataclass
class ConversationDetectionConfig:
    """Complete configuration for conversation detection system.

    Attributes:
        audio: Audio processing configuration
        detection: Conversation detection parameters
        paths: File paths configuration
        processing: Processing configuration
        huggingface_token: Token for accessing Hugging Face models
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    detection: ConversationDetectionParams = field(
        default_factory=ConversationDetectionParams
    )
    paths: FilePaths = field(default_factory=FilePaths)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    huggingface_token: Optional[str] = field(
        default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN")
    )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConversationDetectionConfig":
        """Create configuration from a dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            ConversationDetectionConfig instance
        """
        audio_dict = config_dict.get("audio", {})
        detection_dict = config_dict.get("detection", {})
        paths_dict = config_dict.get("paths", {})
        processing_dict = config_dict.get("processing", {})

        return cls(
            audio=AudioConfig(**audio_dict) if audio_dict else AudioConfig(),
            detection=ConversationDetectionParams(**detection_dict)
            if detection_dict
            else ConversationDetectionParams(),
            paths=FilePaths(**paths_dict) if paths_dict else FilePaths(),
            processing=ProcessingConfig(**processing_dict)
            if processing_dict
            else ProcessingConfig(),
            huggingface_token=config_dict.get(
                "huggingface_token", os.getenv("HUGGINGFACE_TOKEN")
            ),
        )

    @classmethod
    def from_json(cls, json_path: str) -> "ConversationDetectionConfig":
        """Load configuration from a JSON file.

        Args:
            json_path: Path to JSON configuration file

        Returns:
            ConversationDetectionConfig instance
        """
        with open(json_path, "r") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        config_dict = asdict(self)
        # Ensure token isn't included in serialized output
        if "huggingface_token" in config_dict:
            config_dict["huggingface_token"] = None if self.huggingface_token else None
        return config_dict

    def save_to_json(self, json_path: str) -> None:
        """Save configuration to a JSON file.

        Args:
            json_path: Path to save JSON configuration
        """
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Create default configuration instance
default_config = ConversationDetectionConfig()
