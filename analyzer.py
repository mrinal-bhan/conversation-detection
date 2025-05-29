"""Conversation analysis for detecting conversation in call recordings."""

import logging
import traceback
from typing import Dict, List, Tuple
import numpy as np

from config import ConversationDetectionConfig
from models import (
    ChannelAnalysis,
    TurnTakingAnalysis,
    AudioQualityMetrics,
    ConversationAnalysis,
)


class ConversationAnalyzer:
    """Analyzer for conversation detection."""

    def __init__(self, config: ConversationDetectionConfig):
        """Initialize conversation analyzer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def analyze_turn_taking(
        self,
        caller_segments: List[Tuple[float, float]],
        receiver_segments: List[Tuple[float, float]],
    ) -> TurnTakingAnalysis:
        """Analyze turn-taking patterns between speakers.

        Args:
            caller_segments: List of (start, end) tuples for caller speech
            receiver_segments: List of (start, end) tuples for receiver speech

        Returns:
            TurnTakingAnalysis object
        """
        try:
            # Initialize empty analysis object
            analysis = TurnTakingAnalysis()

            if not caller_segments or not receiver_segments:
                return analysis

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

            # Populate the analysis object
            analysis.total_turns = len(turns)
            analysis.turn_switches = turn_switches
            analysis.avg_turn_duration = (
                np.mean([t[0] for t in turns]) if turns else 0.0
            )
            analysis.turn_rate = (
                len(turns) / total_duration if total_duration > 0 else 0.0
            )
            analysis.overlap_ratio = (
                overlap_duration / total_duration if total_duration > 0 else 0.0
            )
            analysis.response_times = response_times
            analysis.speaker_balance = min(caller_duration, receiver_duration) / max(
                caller_duration, receiver_duration, 1e-6
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing turn-taking: {str(e)}")
            self.logger.error(traceback.format_exc())
            return TurnTakingAnalysis()

    def determine_conversation_detection(
        self,
        caller_analysis: ChannelAnalysis,
        receiver_analysis: ChannelAnalysis,
        turn_taking: TurnTakingAnalysis,
        duration: float,
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Determine if a conversation occurred and calculate confidence.

        Args:
            caller_analysis: Analysis results for caller channel
            receiver_analysis: Analysis results for receiver channel
            turn_taking: Turn-taking analysis
            duration: Total duration of the recording

        Returns:
            Tuple of (is_conversation, confidence_score, component_scores)
        """
        try:
            # Initialize scoring components
            scores = {}

            # 1. Check minimum duration
            if duration < self.config.detection.min_conversation_duration:
                return False, 0.0, {}

            # 2. Speech presence score
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

            # 3. Turn-taking score
            turn_score = 0.0
            if turn_taking.total_turns >= self.config.detection.min_turns:
                # Consider turn rate and speaker balance
                turn_score = 0.6 * min(1.0, turn_taking.turn_rate * 10)
                turn_score += 0.4 * turn_taking.speaker_balance
            scores["turns"] = turn_score

            # 4. Audio quality score
            quality_score = 0.0
            for channel in [caller_analysis, receiver_analysis]:
                if (
                    channel.energy_stats["rms_energy"]
                    > self.config.detection.min_audio_energy_threshold
                ):
                    quality_score += 0.5
                if channel.energy_stats["estimated_snr"] > 10:  # 10 dB threshold
                    quality_score += 0.5
            scores["quality"] = min(1.0, quality_score)

            # 5. Interaction score
            interaction_score = 0.0
            if turn_taking.response_times:
                # Good response times between 0.1 and 2.0 seconds
                good_responses = sum(
                    1 for t in turn_taking.response_times if 0.1 <= t <= 2.0
                )
                interaction_score = good_responses / len(turn_taking.response_times)

                # Add bonus for balanced speech
                if turn_taking.speaker_balance > 0.3:  # If reasonably balanced
                    interaction_score = min(1.0, interaction_score * 1.2)
            scores["interaction"] = interaction_score

            # Calculate weighted final score with emphasis on speech presence and balance
            weights = {
                "speech": self.config.detection.speech_presence_weight,
                "turns": self.config.detection.turn_taking_weight,
                "quality": self.config.detection.audio_quality_weight,
                "interaction": self.config.detection.interaction_weight,
            }

            final_score = sum(
                score * weights[component] for component, score in scores.items()
            )

            # Add bonus for balanced speech with good SNR
            if (
                turn_taking.speaker_balance > 0.3
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
                and turn_taking.total_turns >= self.config.detection.min_turns
                and duration >= self.config.detection.min_conversation_duration
            )

            # If basic criteria are met and either:
            # 1. High confidence score OR
            # 2. Good balance between speakers with decent audio quality
            is_conversation = basic_conversation_criteria and (
                final_score >= self.config.detection.conversation_detection_threshold
                or (
                    turn_taking.speaker_balance > 0.3
                    and caller_analysis.energy_stats["estimated_snr"] > 20
                    and receiver_analysis.energy_stats["estimated_snr"] > 20
                )
            )

            # Log the decision factors
            self.logger.debug(
                f"Decision factors: score={final_score:.3f}, duration={duration:.1f}s, "
                f"caller_speech={caller_analysis.has_speech}, receiver_speech={receiver_analysis.has_speech}, "
                f"speaker_balance={turn_taking.speaker_balance:.2f}, "
                f"caller_snr={caller_analysis.energy_stats['estimated_snr']:.1f}, "
                f"receiver_snr={receiver_analysis.energy_stats['estimated_snr']:.1f}"
            )

            return is_conversation, final_score, scores

        except Exception as e:
            self.logger.error(f"Error determining conversation: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False, 0.0, {}

    def create_audio_quality_metrics(
        self, audio_quality_dict: Dict[str, float]
    ) -> AudioQualityMetrics:
        """Create AudioQualityMetrics object from dictionary.

        Args:
            audio_quality_dict: Dictionary of audio quality metrics

        Returns:
            AudioQualityMetrics object
        """
        return AudioQualityMetrics(
            snr_db=audio_quality_dict.get("snr_db", 0.0),
            spectral_centroid=audio_quality_dict.get("spectral_centroid", 0.0),
            dynamic_range=audio_quality_dict.get("dynamic_range", 0.0),
        )

    def analyze_conversation(
        self,
        recording_id: str,
        duration: float,
        caller_analysis: ChannelAnalysis,
        receiver_analysis: ChannelAnalysis,
        audio_quality_dict: Dict[str, float],
        is_mono: bool = False,
    ) -> ConversationAnalysis:
        """Analyze conversation between two speakers.

        Args:
            recording_id: Unique identifier for the recording
            duration: Total duration of the recording
            caller_analysis: Analysis results for caller channel
            receiver_analysis: Analysis results for receiver channel
            audio_quality_dict: Audio quality measurements
            is_mono: Whether the original recording was mono

        Returns:
            ConversationAnalysis object with results
        """
        # Analyze turn-taking patterns
        turn_taking = self.analyze_turn_taking(
            caller_analysis.speech_segments,
            receiver_analysis.speech_segments,
        )

        # Create audio quality metrics object
        audio_quality_metrics = self.create_audio_quality_metrics(audio_quality_dict)

        # Determine if conversation occurred
        conversation_detected, confidence, detection_scores = (
            self.determine_conversation_detection(
                caller_analysis,
                receiver_analysis,
                turn_taking,
                duration,
            )
        )

        # Adjust confidence for mono recordings
        if is_mono:
            confidence *= 0.7  # Reduce confidence for mono recordings

        # Create and return conversation analysis object
        return ConversationAnalysis(
            recording_id=recording_id,
            duration=duration,
            caller_analysis=caller_analysis,
            receiver_analysis=receiver_analysis,
            conversation_detected=conversation_detected,
            conversation_confidence=confidence,
            turn_taking_analysis=turn_taking,
            audio_quality_metrics=audio_quality_metrics,
            is_mono=is_mono,
            detection_scores=detection_scores,
        )
