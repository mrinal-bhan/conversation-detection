"""Configuration for conversation detection analysis."""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class ConversationDetectionConfig:
    """Configuration for conversation detection analysis.

    Attributes:
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels (2 for stereo)
        min_speech_duration: Minimum duration for a speech segment in seconds
        min_silence_duration: Minimum silence duration between segments in seconds
        min_conversation_duration: Minimum total conversation duration in seconds
        min_turns: Minimum number of speaker turns for valid conversation
        min_audio_energy_threshold: Minimum RMS energy threshold for valid audio
        conversation_detection_threshold: Confidence threshold for conversation detection
        caller_channel: Channel index for caller audio (0 = left)
        receiver_channel: Channel index for receiver audio (1 = right)
        input_recordings_dir: Directory containing input recordings
        output_dir: Directory for analysis results
        logs_dir: Directory for log files
        temp_dir: Directory for temporary files
        batch_size: Number of recordings to process in parallel
        max_workers: Maximum number of parallel workers
        huggingface_token: Token for accessing Hugging Face models
    """

    # Audio processing parameters
    sample_rate: int = 16000
    channels: int = 2
    min_speech_duration: float = (
        0.2  # Reduced from 0.3 to catch shorter speech segments
    )
    min_silence_duration: float = 0.3  # Reduced from 0.5 to detect quicker turn-taking

    # Conversation detection parameters
    min_conversation_duration: float = (
        2.0  # Reduced from 5.0 to handle shorter conversations
    )
    min_turns: int = 1  # Reduced from 3 to detect conversations with fewer turns
    min_audio_energy_threshold: float = 0.003  # Further reduced for better sensitivity
    conversation_detection_threshold: float = (
        0.40  # Further reduced from 0.45 to catch more edge cases
    )

    # Channel parameters
    caller_channel: int = 0
    receiver_channel: int = 1

    # File paths
    input_recordings_dir: str = "/Users/mrinal/myEqual/Azure/prod_recordings"
    output_dir: str = "/Users/mrinal/myEqual/Azure/conversation-detection/results/prod"
    logs_dir: str = "logs"
    temp_dir: str = "temp"

    # Analysis parameters
    batch_size: int = 8
    max_workers: int = 4

    # Hugging Face token for pyannote models
    huggingface_token: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")


# Default configuration instance
default_config = ConversationDetectionConfig()
