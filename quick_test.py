#!/usr/bin/env python3
"""
Quick Test Script for Conversation Detection System

This script runs a quick test of the conversation detection system on a small
subset of recordings to verify functionality and provide detailed feedback.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from conversation_detector import ConversationDetector
from config import ConversationDetectionConfig


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def get_huggingface_token(args_token: Optional[str] = None) -> Optional[str]:
    """Get Hugging Face token from various sources.

    Args:
        args_token: Token provided via command line arguments

    Returns:
        Token if found, None otherwise
    """
    # Try command line argument first
    if args_token:
        return args_token

    # Try environment variable
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token

    # Try .env file
    if os.path.exists(".env"):
        load_dotenv()
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token:
            return token

    return None


def main():
    """Run quick test of conversation detection system."""
    print("\n🚀 Conversation Detection System - Quick Test")
    print("=" * 60)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Quick test for conversation detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--recordings",
        type=str,
        default="../real-time-transcription/exotel_recordings",
        help="Path to recordings directory",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of recordings to analyze",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API token",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="quick_test_results",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Get Hugging Face token
    token = get_huggingface_token(args.token)
    if not token:
        print("\n❌ Hugging Face token not found!")
        print("Please provide the token in one of these ways:")
        print("1. Command line argument: --token YOUR_TOKEN")
        print("2. Environment variable: HUGGINGFACE_TOKEN")
        print("3. .env file with HUGGINGFACE_TOKEN=YOUR_TOKEN")
        return

    # Check recordings directory
    recordings_dir = Path(args.recordings)
    if not recordings_dir.exists():
        print(f"\n❌ Recordings directory not found: {recordings_dir}")
        return

    # Count available recordings
    recordings = list(recordings_dir.glob("*.mp3"))
    if not recordings:
        print(f"\n❌ No MP3 files found in {recordings_dir}")
        return

    print(f"\n📁 Found {len(recordings)} recordings")
    print(f"🎯 Testing with {args.samples} sample(s)")

    # Configure for quick test
    config = ConversationDetectionConfig(
        input_recordings_dir=str(recordings_dir),
        output_dir=args.output,
        max_workers=2,  # Reduced for testing
        conversation_detection_threshold=0.65,
        min_conversation_duration=3.0,  # Reduced for testing
        huggingface_token=token,
    )

    try:
        # Initialize detector
        print("\n🔧 Initializing detector...")
        start_time = time.time()
        detector = ConversationDetector(config)
        init_time = time.time() - start_time
        print(f"✅ Initialization completed in {format_duration(init_time)}")

        # Run analysis
        print("\n🧪 Running analysis...")
        start_time = time.time()
        analyses, metrics = detector.run_analysis(max_recordings=args.samples)
        analysis_time = time.time() - start_time

        if not analyses:
            print("\n❌ No analyses were completed successfully")
            return

        print(f"\n✅ Analysis completed in {format_duration(analysis_time)}")

        # Show detailed results
        print("\n📊 Results Summary:")
        print("-" * 60)

        for i, analysis in enumerate(analyses, 1):
            status = "✓" if analysis.conversation_detected else "✗"
            confidence = f"{analysis.conversation_confidence:.1%}"
            duration = format_duration(analysis.duration)

            print(f"\nRecording {i}: {analysis.recording_id}")
            print(
                f"{'Status:':<15} {status} {'(Conversation detected)' if analysis.conversation_detected else '(No conversation)'}"
            )
            print(f"{'Confidence:':<15} {confidence}")
            print(f"{'Duration:':<15} {duration}")
            print(f"{'Turns:':<15} {analysis.turn_taking_analysis['total_turns']}")

            # Channel details
            for channel in ["Caller", "Receiver"]:
                ch_analysis = (
                    analysis.caller_analysis
                    if channel == "Caller"
                    else analysis.receiver_analysis
                )
                speech_ratio = ch_analysis.total_speech_duration / analysis.duration
                print(f"\n{channel} Channel:")
                print(f"{'  Speech ratio:':<20} {speech_ratio:.1%}")
                print(
                    f"{'  Energy level:':<20} {ch_analysis.energy_stats['rms_energy']:.2e}"
                )
                print(
                    f"{'  SNR:':<20} {ch_analysis.energy_stats['estimated_snr']:.1f} dB"
                )

        # Show metrics if available
        if metrics:
            print("\n🎯 Performance Metrics:")
            print("-" * 60)
            print(f"{'Precision:':<15} {metrics.precision:.1%}")
            print(f"{'Recall:':<15} {metrics.recall:.1%}")
            print(f"{'F1-Score:':<15} {metrics.f1_score:.1%}")
            print(f"{'Accuracy:':<15} {metrics.accuracy:.1%}")
            print(f"{'Total samples:':<15} {metrics.total_samples}")

        print(f"\n📁 Detailed results saved to: {config.output_dir}")
        print("\n🎉 Quick test completed successfully!")
        print("\nTo run the full analysis:")
        print(
            "python conversation_detector.py --recordings <path> --token <huggingface_token>"
        )

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("1. Check if you provided a valid Hugging Face token")
        print("2. Verify the recordings directory path")
        print("3. Ensure all dependencies are installed")
        print("4. Check the logs for detailed error messages")


if __name__ == "__main__":
    main()
