"""Data models for conversation detection."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import numpy as np


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
class TurnTakingAnalysis:
    """Analysis of turn-taking patterns between speakers.

    Attributes:
        total_turns: Total number of speaking turns
        turn_switches: Number of times the speaker changed
        avg_turn_duration: Average duration of speaking turns
        turn_rate: Rate of turns per second
        overlap_ratio: Ratio of overlapping speech to total duration
        response_times: List of times between turns
        speaker_balance: Ratio of speaking time between speakers (0-1, 1=perfectly balanced)
    """

    total_turns: int = 0
    turn_switches: int = 0
    avg_turn_duration: float = 0.0
    turn_rate: float = 0.0
    overlap_ratio: float = 0.0
    response_times: List[float] = field(default_factory=list)
    speaker_balance: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_turns": self.total_turns,
            "turn_switches": self.turn_switches,
            "avg_turn_duration": self.avg_turn_duration,
            "turn_rate": self.turn_rate,
            "overlap_ratio": self.overlap_ratio,
            "response_times": self.response_times,
            "speaker_balance": self.speaker_balance,
        }


@dataclass
class AudioQualityMetrics:
    """Audio quality measurements.

    Attributes:
        snr_db: Signal-to-noise ratio in dB
        spectral_centroid: Average spectral centroid
        dynamic_range: Dynamic range of the audio
    """

    snr_db: float = 0.0
    spectral_centroid: float = 0.0
    dynamic_range: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "snr_db": self.snr_db,
            "spectral_centroid": self.spectral_centroid,
            "dynamic_range": self.dynamic_range,
        }


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
        detection_scores: Individual component scores for detection
    """

    recording_id: str
    duration: float
    caller_analysis: ChannelAnalysis
    receiver_analysis: ChannelAnalysis
    conversation_detected: bool
    conversation_confidence: float
    turn_taking_analysis: TurnTakingAnalysis
    audio_quality_metrics: AudioQualityMetrics
    is_mono: bool = False
    detection_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis results to dictionary for serialization."""
        return {
            "recording_id": self.recording_id,
            "duration": self.duration,
            "conversation_detected": self.conversation_detected,
            "conversation_confidence": self.conversation_confidence,
            "caller_has_speech": self.caller_analysis.has_speech,
            "receiver_has_speech": self.receiver_analysis.has_speech,
            "caller_speech_duration": self.caller_analysis.total_speech_duration,
            "receiver_speech_duration": self.receiver_analysis.total_speech_duration,
            "turn_taking": self.turn_taking_analysis.as_dict(),
            "audio_quality": self.audio_quality_metrics.as_dict(),
            "is_mono": self.is_mono,
            "detection_scores": self.detection_scores,
        }


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
        true_positives: Number of true positives
        false_positives: Number of false positives
        true_negatives: Number of true negatives
        false_negatives: Number of false negatives
    """

    precision: float
    recall: float
    f1_score: float
    accuracy: float
    confusion_matrix: np.ndarray
    total_samples: int
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "total_samples": self.total_samples,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
        }
