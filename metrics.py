"""Metrics calculation for conversation detection evaluation."""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from config import ConversationDetectionConfig
from models import ConversationAnalysis, MetricsResult


class MetricsCalculator:
    """Calculate evaluation metrics for conversation detection."""

    def __init__(self, config: ConversationDetectionConfig):
        """Initialize metrics calculator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def generate_ground_truth_from_transcription_report(self) -> Dict[str, bool]:
        """Generate ground truth from transcription analysis report.

        Returns:
            Dictionary mapping recording IDs to ground truth values (True=conversation)
        """
        try:
            # Read transcription analysis report
            report_path = os.path.join(
                self.config.paths.input_recordings_dir,
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
                self.config.paths.input_recordings_dir,
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
                self.config.paths.output_dir, "ground_truth.json"
            )
            with open(ground_truth_path, "w") as f:
                json.dump(ground_truth, f, indent=2)

            self.logger.info(f"Ground truth saved to: {ground_truth_path}")
            return ground_truth

        except Exception as e:
            self.logger.error(f"Error generating ground truth: {e}")
            return {}

    def load_ground_truth(
        self, ground_truth_path: Optional[str] = None
    ) -> Dict[str, bool]:
        """Load ground truth from a JSON file.

        Args:
            ground_truth_path: Path to ground truth JSON file,
                              defaults to ground_truth.json in output directory

        Returns:
            Dictionary mapping recording IDs to ground truth values
        """
        if not ground_truth_path:
            ground_truth_path = os.path.join(
                self.config.paths.output_dir, "ground_truth.json"
            )

        if not os.path.exists(ground_truth_path):
            self.logger.warning(f"Ground truth file not found: {ground_truth_path}")
            return {}

        try:
            with open(ground_truth_path, "r") as f:
                ground_truth = json.load(f)

            self.logger.info(f"Loaded ground truth for {len(ground_truth)} recordings")
            return ground_truth
        except Exception as e:
            self.logger.error(f"Error loading ground truth: {e}")
            return {}

    def calculate_metrics(
        self, analyses: List[ConversationAnalysis], ground_truth: Dict[str, bool]
    ) -> MetricsResult:
        """Calculate evaluation metrics.

        Args:
            analyses: List of conversation analysis results
            ground_truth: Dictionary mapping recording IDs to ground truth values

        Returns:
            MetricsResult object with evaluation metrics

        Raises:
            ValueError: If no recordings found in both analyses and ground truth
        """
        # Align analyses with ground truth
        y_true = []
        y_pred = []
        matching_ids = []

        for analysis in analyses:
            if analysis.recording_id in ground_truth:
                y_true.append(ground_truth[analysis.recording_id])
                y_pred.append(analysis.conversation_detected)
                matching_ids.append(analysis.recording_id)

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

        # Calculate TP, FP, TN, FN
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Create and return metrics result
        result = MetricsResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            confusion_matrix=cm,
            total_samples=len(y_true),
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
        )

        # Log metrics
        self.logger.info(
            f"Evaluation metrics calculated for {len(matching_ids)} recordings"
        )
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1 Score: {f1:.4f}")
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Confusion Matrix: \n{cm}")

        return result

    def analyze_errors(
        self, analyses: List[ConversationAnalysis], ground_truth: Dict[str, bool]
    ) -> Tuple[List[str], List[str]]:
        """Analyze false positives and false negatives.

        Args:
            analyses: List of conversation analysis results
            ground_truth: Dictionary mapping recording IDs to ground truth values

        Returns:
            Tuple of (false_positive_ids, false_negative_ids)
        """
        false_positives = []
        false_negatives = []

        for analysis in analyses:
            if analysis.recording_id in ground_truth:
                ground_truth_value = ground_truth[analysis.recording_id]
                predicted_value = analysis.conversation_detected

                if predicted_value and not ground_truth_value:
                    false_positives.append(analysis.recording_id)
                elif not predicted_value and ground_truth_value:
                    false_negatives.append(analysis.recording_id)

        self.logger.info(
            f"Found {len(false_positives)} false positives and {len(false_negatives)} false negatives"
        )
        return false_positives, false_negatives

    def save_metrics(
        self, metrics: MetricsResult, output_path: Optional[str] = None
    ) -> None:
        """Save metrics to a JSON file.

        Args:
            metrics: MetricsResult object
            output_path: Path to save metrics, defaults to metrics.json in output directory
        """
        if not output_path:
            output_path = os.path.join(self.config.paths.output_dir, "metrics.json")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with open(output_path, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)

            self.logger.info(f"Metrics saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")

    def generate_metrics_report(
        self,
        metrics: MetricsResult,
        false_positives: List[str],
        false_negatives: List[str],
        output_path: Optional[str] = None,
    ) -> None:
        """Generate a detailed metrics report.

        Args:
            metrics: MetricsResult object
            false_positives: List of false positive recording IDs
            false_negatives: List of false negative recording IDs
            output_path: Path to save report, defaults to metrics_report.txt in output directory
        """
        if not output_path:
            output_path = os.path.join(
                self.config.paths.output_dir, "metrics_report.txt"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with open(output_path, "w") as f:
                f.write("# Conversation Detection Metrics Report\n\n")

                f.write("## Summary\n")
                f.write(f"Total recordings evaluated: {metrics.total_samples}\n")
                f.write(f"Precision: {metrics.precision:.4f}\n")
                f.write(f"Recall: {metrics.recall:.4f}\n")
                f.write(f"F1 Score: {metrics.f1_score:.4f}\n")
                f.write(f"Accuracy: {metrics.accuracy:.4f}\n\n")

                f.write("## Confusion Matrix\n")
                f.write("```\n")
                f.write(f"True Positives: {metrics.true_positives}\n")
                f.write(f"False Positives: {metrics.false_positives}\n")
                f.write(f"True Negatives: {metrics.true_negatives}\n")
                f.write(f"False Negatives: {metrics.false_negatives}\n")
                f.write("```\n\n")

                f.write("## Error Analysis\n")

                f.write("### False Positives\n")
                f.write("Recordings incorrectly classified as conversations:\n")
                for recording_id in false_positives[:10]:
                    f.write(f"- {recording_id}\n")
                if len(false_positives) > 10:
                    f.write(f"- ... and {len(false_positives) - 10} more\n")
                f.write("\n")

                f.write("### False Negatives\n")
                f.write("Conversations incorrectly classified as non-conversations:\n")
                for recording_id in false_negatives[:10]:
                    f.write(f"- {recording_id}\n")
                if len(false_negatives) > 10:
                    f.write(f"- ... and {len(false_negatives) - 10} more\n")

            self.logger.info(f"Metrics report saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error generating metrics report: {e}")
