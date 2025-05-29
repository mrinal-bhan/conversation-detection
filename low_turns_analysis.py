#!/usr/bin/env python3
"""
Script to analyze conversation detection results from conversation_analysis1.csv,
focusing on calls with fewer than 5 turn switches and generating a markdown report.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")
sns.set_context("notebook", font_scale=1.2)

OUTPUT_DIR = "results/turn-count-analysis"


def load_data(csv_path):
    """Load the conversation analysis data."""
    df = pd.read_csv(csv_path)
    return df


def analyze_low_turns(df, max_turns=4):
    """Analyze conversations with low number of turn switches."""
    # Filter for calls with turn_switches < max_turns+1
    low_turns_df = df[df["turn_switches"] <= max_turns].copy()

    # Add a column for turn count category
    low_turns_df["turn_category"] = low_turns_df["turn_switches"].astype(str)

    return low_turns_df


def generate_plots(df, output_dir):
    """Generate plots for analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Distribution of conversation detection by turn count
    plt.figure(figsize=(12, 8))
    turn_counts = pd.crosstab(df["turn_switches"], df["conversation_detected"])
    turn_counts_pct = turn_counts.div(turn_counts.sum(axis=1), axis=0) * 100

    ax = turn_counts_pct.plot(kind="bar", stacked=True)
    plt.title("Conversation Detection Rate by Turn Count")
    plt.xlabel("Number of Turn Switches")
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.legend(["Not Detected", "Detected"])

    # Add count labels on bars
    for i, t in enumerate(turn_counts.index):
        for j, col in enumerate(turn_counts.columns):
            count = turn_counts.loc[t, col]
            pct = turn_counts_pct.loc[t, col]
            if pct > 5:  # Only show percentage if it's large enough
                ax.text(
                    i,
                    pct / 2 if j == 0 else (100 - pct / 2),
                    f"{count}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "conversation_detection_by_turns.png"), dpi=300
    )
    plt.close()

    # 2. Confidence score distribution by turn count
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        x="turn_switches",
        y="conversation_confidence",
        hue="conversation_detected",
        data=df,
    )
    plt.title("Confidence Score Distribution by Turn Count")
    plt.xlabel("Number of Turn Switches")
    plt.ylabel("Confidence Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_score_by_turns.png"), dpi=300)
    plt.close()

    # 3. Speech duration comparison
    plt.figure(figsize=(12, 10))

    # Calculate speaker balance ratio
    df["speech_balance"] = np.minimum(
        df["caller_speech_duration"], df["receiver_speech_duration"]
    ) / np.maximum(
        df["caller_speech_duration"], df["receiver_speech_duration"]
    ).replace(0, 1)

    # Define bubble size based on conversation duration
    bubble_size = (df["caller_speech_duration"] + df["receiver_speech_duration"]) * 10

    # Create scatter plot with bubbles
    plt.scatter(
        df["caller_speech_duration"],
        df["receiver_speech_duration"],
        s=bubble_size,
        c=df["conversation_detected"].map({True: "green", False: "red"}),
        alpha=0.6,
        edgecolors="w",
    )

    # Add diagonal line for equal speech
    max_val = max(
        df["caller_speech_duration"].max(), df["receiver_speech_duration"].max()
    )
    plt.plot([0, max_val], [0, max_val], "k--", alpha=0.5)

    plt.xlabel("Caller Speech Duration (s)")
    plt.ylabel("Receiver Speech Duration (s)")
    plt.title("Speech Duration by Speaker for Calls with < 5 Turn Switches")

    # Create custom legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=10,
            label="Conversation Detected",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="No Conversation Detected",
        ),
    ]
    plt.legend(handles=legend_elements)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speech_duration_comparison.png"), dpi=300)
    plt.close()


def generate_markdown_report(df, output_dir):
    """Generate a markdown report with statistics and plots."""
    # Calculate overall statistics
    total_calls = len(df)
    conversation_detected = df["conversation_detected"].sum()
    no_conversation = total_calls - conversation_detected
    detection_rate = (conversation_detected / total_calls) * 100

    # Statistics by turn count
    stats_by_turn = []
    for turn in sorted(df["turn_switches"].unique()):
        turn_df = df[df["turn_switches"] == turn]
        total_in_turn = len(turn_df)
        detected_in_turn = turn_df["conversation_detected"].sum()
        stats_by_turn.append(
            {
                "turn_switches": turn,
                "total_calls": total_in_turn,
                "conversations_detected": detected_in_turn,
                "detection_rate": detected_in_turn / total_in_turn * 100
                if total_in_turn > 0
                else 0,
                "avg_confidence": turn_df["conversation_confidence"].mean(),
                "avg_caller_speech": turn_df["caller_speech_duration"].mean(),
                "avg_receiver_speech": turn_df["receiver_speech_duration"].mean(),
            }
        )

    # Generate markdown content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown_content = f"""# Conversation Detection Analysis Report - Low Turns
Generated on: {timestamp}

## Overview

This report analyzes conversation detection results for calls with fewer than 5 turn switches.

### Key Statistics

- **Total Calls Analyzed**: {total_calls}
- **Conversations Detected**: {conversation_detected} ({detection_rate:.1f}%)
- **No Conversation Detected**: {no_conversation} ({100 - detection_rate:.1f}%)

## Analysis by Turn Count

| Turn Switches | Total Calls | Conversations Detected | Detection Rate | Avg Confidence | Avg Caller Speech | Avg Receiver Speech |
|--------------|-------------|----------------------|----------------|----------------|------------------|-------------------|
"""

    for stat in stats_by_turn:
        markdown_content += f"| {stat['turn_switches']} | {stat['total_calls']} | {stat['conversations_detected']} | {stat['detection_rate']:.1f}% | {stat['avg_confidence']:.3f} | {stat['avg_caller_speech']:.2f}s | {stat['avg_receiver_speech']:.2f}s |\n"

    markdown_content += """
## Visualizations

### 1. Conversation Detection Rate by Turn Count

This plot shows the distribution of conversation detection results for each turn count:

![Conversation Detection by Turns](conversation_detection_by_turns.png)

### 2. Confidence Score Distribution

This plot shows the distribution of confidence scores by turn count and detection status:

![Confidence Score Distribution](confidence_score_by_turns.png)

### 3. Speech Duration Analysis

This plot compares caller and receiver speech durations, with bubble size representing total conversation duration:

![Speech Duration Comparison](speech_duration_comparison.png)

## Key Findings

1. **Turn Count Impact**: The analysis shows how conversation detection rates vary with the number of turn switches.
2. **Speech Duration Patterns**: The speech duration comparison reveals patterns in caller vs receiver speech distribution.
3. **Confidence Score Trends**: The confidence score distribution helps understand the reliability of detection at different turn counts.

## Methodology

- Analysis focused on calls with 0-4 turn switches
- Conversation detection based on speech presence, turn-taking patterns, and duration
- Confidence scores reflect the certainty of conversation detection
"""

    # Save markdown report
    with open(os.path.join(output_dir, "analysis_report.md"), "w") as f:
        f.write(markdown_content)


def main():
    # Set paths
    csv_path = "results/full_analysis_500/combined_results.csv"
    output_dir = "results/full_analysis_500/low_turns_analysis"
    
    # Load data
    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} records")
    
    # Analyze low turns (< 5 turn switches)
    print("Analyzing calls with fewer than 5 turn switches...")
    low_turns_df = analyze_low_turns(df, max_turns=4)
    print(f"Found {len(low_turns_df)} calls with < 5 turn switches")
    
    # Generate plots
    print("Generating plots...")
    generate_plots(low_turns_df, output_dir)
    
    # Generate markdown report
    print("Generating markdown report...")
    generate_markdown_report(low_turns_df, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"- Plots saved as PNG files")
    print(f"- Report saved as: {output_dir}/low_turns_analysis_report.md")


if __name__ == "__main__":
    main()
