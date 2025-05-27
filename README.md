# Conversation Detection System

An intelligent system that analyzes call recordings to identify meaningful conversations between callers and receivers. The system uses advanced audio analysis and machine learning to detect speech patterns, turn-taking behavior, and conversation quality.

## Understanding Conversation Detection

### What Makes a Conversation?

The system identifies a conversation based on four key factors:

1. **Speech Activity (35% of score)**
   - Detects when both caller and receiver are speaking
   - Measures how much each person talks
   - Evaluates the balance of speaking time between parties

2. **Turn-Taking (25% of score)**
   - Identifies when speakers alternate
   - Measures how smoothly the conversation flows
   - Analyzes response timing and interaction patterns

3. **Audio Quality (20% of score)**
   - Measures clarity of the audio signal
   - Evaluates background noise levels
   - Ensures both parties can be heard clearly

4. **Interaction Quality (20% of score)**
   - Assesses natural conversation flow
   - Measures response times between speakers
   - Evaluates overall engagement level

### Key Metrics Explained

- **Conversation Confidence**: Overall score (0-100%) indicating how likely a real conversation occurred
- **Speech Ratio**: Percentage of time each party speaks
- **Turn Switches**: Number of times the conversation switches between speakers
- **Signal Quality**: Audio clarity measurement in decibels (dB)
- **Speaker Balance**: How evenly balanced the conversation is between parties

### What Counts as a Conversation?

A recording is marked as a conversation when:
- Both parties have detectable speech
- At least one turn exchange occurs
- Recording is longer than 2 seconds
- Either:
  - Overall confidence score is above 40%
  - OR speakers are well-balanced with good audio quality

## Quick Start Guide

1. **Prepare Your Recordings**
   ```bash
   python exotel_recorder.py  # Download recordings
   ```

2. **Run Analysis**
   ```bash
   python quick_test.py --samples 5  # Test with 5 recordings
   ```

## Understanding Results

### Sample Output
```json
{
  "recording_id": "example123",
  "conversation_detected": true,
  "confidence": "85%",
  "duration": "45 seconds",
  "caller_speech": "40%",
  "receiver_speech": "35%",
  "turns": 12,
  "audio_quality": "Excellent"
}
```

### Quality Indicators

**Excellent Conversation**
- Confidence > 80%
- Balanced speech (30-70% each)
- Multiple turns
- Clear audio (SNR > 30dB)

**Good Conversation**
- Confidence > 60%
- Some speech imbalance acceptable
- At least one turn
- Decent audio (SNR > 20dB)

**Poor Quality**
- Confidence < 40%
- Very unbalanced speech
- No clear turns
- Poor audio quality

## Best Practices

1. **Audio Quality**
   - Ensure good recording quality
   - Minimize background noise
   - Use appropriate recording levels

2. **Recording Duration**
   - Allow sufficient time for interaction
   - Capture complete exchanges
   - Include context before and after main conversation

3. **Analysis Tips**
   - Process recordings soon after capture
   - Review results for patterns
   - Use insights to improve call quality

## Features

- **Automatic Ground Truth**: Generates ground truth from transcription analysis report
- **Voice Activity Detection**: Uses pyannote.audio for speech detection
- **Stereo Channel Analysis**: Separates caller and receiver channels
- **Comprehensive Metrics**: Precision, recall, F1-score evaluation
- **Easy to Use**: Simple command-line interface

## Quick Start

### 1. Install Dependencies

```bash
pip install -r simple_requirements.txt
```

### 2. Set HuggingFace Token (already configured)

The HuggingFace token is already configured in the code. If you need to change it:

```python
# In simple_conversation_detector.py, update the Config class:
huggingface_token: str = "your_token_here"
```

### 3. Run Quick Test

Test with 3 recordings:

```bash
python simple_quick_test.py
```

### 4. Run Full Analysis

Analyze all recordings:

```bash
python simple_conversation_detector.py
```

## How It Works

The system uses a multi-step process to detect conversations:

1. **Voice Activity Detection (VAD)**
   - Uses pyannote.audio's segmentation model to detect speech segments
   - Analyzes both caller and receiver channels independently
   - Identifies speech segments with duration > 0.2s

2. **Conversation Detection Scoring**
   The system calculates a confidence score (0-1) based on:
   - Speech presence (35%): Total speech duration and balance between speakers
   - Turn-taking (25%): Number of turns and speaker transitions
   - Audio quality (20%): Signal strength and SNR
   - Interaction (20%): Response timing and speaker balance

   A conversation is detected when:
   - Both caller and receiver have speech segments
   - At least 1 turn between speakers
   - Duration > 2 seconds
   - Confidence score > 0.40

## Configuration Options

```bash
# Custom recordings directory
python simple_conversation_detector.py --recordings-dir /path/to/recordings

# Limit recordings for testing
python simple_conversation_detector.py --max-recordings 10

# Adjust confidence threshold
python simple_conversation_detector.py --confidence-threshold 0.8

# Single-threaded for debugging
python simple_conversation_detector.py --max-workers 1 --verbose

# Custom output directory
python simple_conversation_detector.py --output-dir my_results
```

## File Structure

```
conversation-detection/
├── simple_conversation_detector.py  # Main system (single file)
├── simple_quick_test.py            # Quick test script
├── simple_requirements.txt         # Dependencies
├── SIMPLE_README.md                # This file
└── results/                        # Output directory
    ├── conversation_analysis_*.json # Detailed results
    ├── conversation_analysis_*.csv  # Summary results
    ├── ground_truth.json           # Generated ground truth
    ├── reports/                    # Performance reports
    └── plots/                      # Visualizations
```

## Troubleshooting

### Common Issues

1. **Missing Recordings Directory**
   ```
   Error: Recordings directory not found
   ```
   Solution: Ensure `../real-time-transcription/exotel_recordings/` exists or specify path with `--recordings-dir`

2. **No Ground Truth Generated**
   ```
   Warning: Could not generate ground truth
   ```
   Solution: Check that transcription analysis report and CSV files exist

3. **Memory Issues**
   ```
   Error: Out of memory
   ```
   Solution: Use `--max-workers 1` or `--max-recordings 10` for smaller batches

4. **HuggingFace Issues**
   ```
   Error: Authentication required
   ```
   Solution: The token is already configured, but ensure internet connectivity

### Performance Tips

- Use `--max-recordings 5` for quick testing
- Use `--max-workers 1` if experiencing crashes
- Check logs in `logs/` directory for detailed error information

## Example Usage

```bash
# Quick test
python simple_quick_test.py

# Full analysis
python simple_conversation_detector.py

# Debug mode
python simple_conversation_detector.py --max-recordings 5 --max-workers 1 --verbose

# High confidence threshold
python simple_conversation_detector.py --confidence-threshold 0.9
```

This simplified system provides the same core functionality as the multi-file version but in a single, easy-to-understand Python file. 