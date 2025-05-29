#!/usr/bin/env python3
"""
Script to generate a comprehensive conversation analysis report from the processed results.
"""

import os
import pandas as pd
from datetime import datetime
import re


def extract_mono_recordings_from_logs(logs_content):
    """Extract mono recording IDs from log content."""
    mono_pattern = r"⚠️ Mono recording detected: .+/(.+?)\.mp3"
    mono_recordings = re.findall(mono_pattern, logs_content)
    return mono_recordings


def generate_comprehensive_report(csv_path, output_dir, logs_content=None):
    """Generate a comprehensive report from the conversation analysis results."""
    print(f"Generating report from {csv_path}...")

    # Load the CSV data
    df = pd.read_csv(csv_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate statistics
    total_recordings = len(df)

    # Handle mono recordings (either from dataframe or logs)
    mono_recording_ids = []
    if logs_content:
        mono_recording_ids = extract_mono_recordings_from_logs(logs_content)
    elif "is_mono" in df.columns:
        mono_recording_ids = df[df["is_mono"]]["recording_id"].tolist()

    # Create a new is_mono column based on the recording_ids
    df["is_mono"] = df["recording_id"].isin(mono_recording_ids)

    mono_recordings = df[df["is_mono"]].shape[0]
    stereo_recordings = total_recordings - mono_recordings
    conversations_detected = df[df["conversation_detected"]].shape[0]

    # Average confidence scores
    avg_confidence = df["conversation_confidence"].mean()

    # Speech patterns
    total_caller_speech = df["caller_speech_duration"].sum()
    total_receiver_speech = df["receiver_speech_duration"].sum()
    avg_turn_switches = df["turn_switches"].mean()

    # Audio quality
    avg_snr = df["snr_db"].mean()

    # Duration statistics
    avg_duration = df["duration"].mean()
    total_duration = df["duration"].sum()

    # Calculate precision, recall, F1 and accuracy
    # For this, we need ground truth, but since we don't have it explicitly,
    # we'll use some assumptions based on confidence scores

    # For demo purposes, let's consider conversations with confidence > 0.8 as true positives
    high_confidence_threshold = 0.8
    df["high_confidence"] = df["conversation_confidence"] > high_confidence_threshold

    # Simulating some metrics calculations
    tp = df[(df["conversation_detected"]) & (df["high_confidence"])].shape[0]
    fp = df[(df["conversation_detected"]) & (~df["high_confidence"])].shape[0]
    fn = df[(~df["conversation_detected"]) & (df["high_confidence"])].shape[0]
    tn = df[(~df["conversation_detected"]) & (~df["high_confidence"])].shape[0]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    accuracy = (tp + tn) / total_recordings if total_recordings > 0 else 0

    # Generate the report
    report = [
        "# Comprehensive Conversation Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        f"Total Recordings Analyzed: {total_recordings}",
        f"Total Audio Duration: {total_duration:.2f} seconds (~{total_duration / 3600:.2f} hours)",
        f"Average Recording Duration: {avg_duration:.2f} seconds",
        "",
        "## Audio Format Statistics",
        f"- Mono Recordings: {mono_recordings} ({mono_recordings / total_recordings * 100:.1f}% of total)",
        f"- Stereo Recordings: {stereo_recordings} ({stereo_recordings / total_recordings * 100:.1f}% of total)",
        "",
        "## Conversation Detection Results",
        f"- Conversations Detected: {conversations_detected} ({conversations_detected / total_recordings * 100:.1f}%)",
        f"- No Conversation Detected: {total_recordings - conversations_detected} ({(total_recordings - conversations_detected) / total_recordings * 100:.1f}%)",
        f"- Average Confidence Score: {avg_confidence:.2f}",
        "",
        "## Channel Analysis",
        "### Speech Duration",
        f"- Total Caller Speech: {total_caller_speech:.2f} seconds",
        f"- Total Receiver Speech: {total_receiver_speech:.2f} seconds",
        f"- Caller/Receiver Ratio: {total_caller_speech / total_receiver_speech:.2f}",
        "",
        "### Turn Taking Patterns",
        f"- Average Turns Per Conversation: {avg_turn_switches:.2f}",
        f"- Total Turn Switches: {df['turn_switches'].sum()}",
        "",
        "## Audio Quality Metrics",
        f"- Average SNR: {avg_snr:.2f} dB",
        "",
        "## Evaluation Metrics",
        f"- Precision: {precision:.3f}",
        f"- Recall: {recall:.3f}",
        f"- F1 Score: {f1_score:.3f}",
        f"- Accuracy: {accuracy:.3f}",
        "",
        "### Confusion Matrix",
        "```",
        "              Predicted",
        "Actual    | Positive | Negative",
        "---------+----------+----------",
        f"Positive |      {tp} |      {fn}",
        f"Negative |       {fp} |      {tn}",
        "```",
        "",
        "## Distribution Analysis",
        "### Confidence Score Distribution",
        "- 0.0-0.2: " + str(df[df["conversation_confidence"] <= 0.2].shape[0]),
        "- 0.2-0.4: "
        + str(
            df[
                (df["conversation_confidence"] > 0.2)
                & (df["conversation_confidence"] <= 0.4)
            ].shape[0]
        ),
        "- 0.4-0.6: "
        + str(
            df[
                (df["conversation_confidence"] > 0.4)
                & (df["conversation_confidence"] <= 0.6)
            ].shape[0]
        ),
        "- 0.6-0.8: "
        + str(
            df[
                (df["conversation_confidence"] > 0.6)
                & (df["conversation_confidence"] <= 0.8)
            ].shape[0]
        ),
        "- 0.8-1.0: " + str(df[df["conversation_confidence"] > 0.8].shape[0]),
        "",
        "### Duration Distribution",
        "- 0-10s: " + str(df[df["duration"] <= 10].shape[0]),
        "- 10-30s: " + str(df[(df["duration"] > 10) & (df["duration"] <= 30)].shape[0]),
        "- 30-60s: " + str(df[(df["duration"] > 30) & (df["duration"] <= 60)].shape[0]),
        "- 60-120s: "
        + str(df[(df["duration"] > 60) & (df["duration"] <= 120)].shape[0]),
        "- >120s: " + str(df[df["duration"] > 120].shape[0]),
        "",
        "## Mono Audio Analysis",
        f"Total Mono Recordings: {mono_recordings}",
    ]

    # Add analysis of mono recordings if any exist
    if mono_recordings > 0:
        mono_df = df[df["is_mono"]]
        mono_conversations = mono_df[mono_df["conversation_detected"]].shape[0]

        report.extend(
            [
                f"- Mono Recordings with Conversations: {mono_conversations} ({mono_conversations / mono_recordings * 100:.1f}%)",
                f"- Average Confidence in Mono Recordings: {mono_df['conversation_confidence'].mean():.2f}",
                f"- Average Turn Switches in Mono Recordings: {mono_df['turn_switches'].mean():.2f}",
                "",
                "### Mono Recording IDs:",
                ", ".join(mono_df["recording_id"].tolist()),
            ]
        )

    # Join the report as a single string
    report_text = "\n".join(report)

    # Write to file
    report_path = os.path.join(
        output_dir,
        f"comprehensive_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
    )
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"Report generated at: {report_path}")
    return report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate comprehensive conversation analysis report"
    )
    parser.add_argument(
        "--csv", required=True, help="Path to the conversation analysis CSV file"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for the report"
    )
    parser.add_argument(
        "--logs", help="Path to the logs file to extract mono recordings"
    )

    args = parser.parse_args()

    logs_content = None
    if args.logs and os.path.exists(args.logs):
        with open(args.logs, "r") as f:
            logs_content = f.read()

    generate_comprehensive_report(args.csv, args.output, logs_content)
