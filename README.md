# VideoAnalysis

**Enable AI Agents to "Watch" and Analyze Video Content**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 52 passed](https://img.shields.io/badge/tests-52%20passed-green.svg)]()

---

## Overview

**VideoAnalysis** is a powerful tool that enables AI agents to analyze video content by extracting frames, detecting scenes, performing OCR on visible text, and identifying movement through delta change detection.

### The Problem

AI agents cannot directly "watch" videos. When Logan asked WSL_CLIO to analyze a 45-minute debugging session video, the agent received only "cannot read binary files" errors. This tool bridges that gap.

### The Solution

VideoAnalysis provides structured analysis of video content:
- **Frame Extraction**: Sample frames at configurable intervals
- **Scene Detection**: Identify scene changes with timestamps
- **Delta Change Detection**: Detect movement using frame differencing
- **OCR Text Extraction**: Read visible text from frames
- **Key Moment Identification**: Find the most significant visual changes
- **Activity Timeline**: Track activity levels throughout the video

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Features](#features)
4. [CLI Commands](#cli-commands)
5. [Python API](#python-api)
6. [Configuration](#configuration)
7. [Output Format](#output-format)
8. [Dependencies](#dependencies)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)
11. [Contributing](#contributing)
12. [Credits](#credits)
13. [License](#license)

---

## Installation

### Prerequisites

**Required:**
- Python 3.9 or higher
- FFmpeg (for video processing)

**Optional:**
- Tesseract OCR (for text extraction)
- OpenCV (for enhanced scene detection)

### Install FFmpeg

**Windows:**
```bash
winget install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### Install VideoAnalysis

**From source:**
```bash
git clone https://github.com/DonkRonk17/VideoAnalysis.git
cd VideoAnalysis
pip install -e .
```

**With all features:**
```bash
pip install -e ".[full]"
```

**Minimal (just delta detection):**
```bash
pip install Pillow
```

---

## Quick Start

### Basic Video Analysis

```bash
# Analyze a video
videoanalysis analyze video.mp4

# Save results to JSON
videoanalysis analyze video.mp4 -o results.json

# Quick analysis (faster, less detail)
videoanalysis analyze video.mp4 --type quick
```

### Check Dependencies

```bash
videoanalysis check-deps
```

### Detect Movement

```bash
# Find key moments using delta change detection
videoanalysis delta video.mp4 --key-moments 10

# With activity timeline
videoanalysis delta video.mp4 --timeline
```

---

## Features

### 1. Frame Extraction

Extract frames at configurable intervals from any video.

```bash
# Extract frames every 30 seconds
videoanalysis frames video.mp4 -o ./frames/ -i 30

# Extract up to 50 frames
videoanalysis frames video.mp4 -o ./frames/ -m 50
```

**Supported formats:** MP4, MOV, AVI, MKV, WebM, WMV

### 2. Scene Detection

Detect scene changes using histogram comparison or I-frame analysis.

```bash
# Detect scenes with default threshold
videoanalysis scenes video.mp4

# Adjust sensitivity (higher = more scenes detected)
videoanalysis scenes video.mp4 -t 0.4
```

### 3. Delta Change Detection (Movement Analysis)

The delta change detector compares consecutive frames to identify:
- **Scene changes**: Major visual shifts (>50 mean delta)
- **Major movement**: Significant activity (30-50 mean delta)
- **Moderate movement**: Noticeable changes (15-30 mean delta)
- **Minor movement**: Subtle changes (<15 mean delta)

```bash
# Basic delta analysis
videoanalysis delta video.mp4

# Custom threshold
videoanalysis delta video.mp4 -t 20.0

# With activity timeline
videoanalysis delta video.mp4 --timeline
```

**How it works:**
1. Extracts frames at short intervals (default: 5 seconds)
2. Converts to grayscale
3. Computes pixel-by-pixel difference between consecutive frames
4. Calculates mean delta (average change) and change percentage
5. Classifies changes and identifies key moments

### 4. OCR Text Extraction

Extract text visible in video frames using Tesseract OCR.

```bash
# Extract text at specific timestamps
videoanalysis ocr video.mp4 --timestamps "00:05:00,00:10:00,00:15:00"
```

### 5. Key Moment Detection

Automatically find the most visually significant moments in a video.

```python
from videoanalysis import VideoAnalyzer

analyzer = VideoAnalyzer()
result = analyzer.analyze("video.mp4")

# Top 10 key moments
for moment in result.key_moments:
    print(f"[{moment.timestamp}] {moment.change_type}: {moment.description}")
```

### 6. Activity Timeline

Track activity levels throughout the video.

```python
for bucket in result.activity_timeline:
    print(f"{bucket.start_time}-{bucket.end_time}: {bucket.activity_level}")
```

---

## CLI Commands

### `analyze` - Full Video Analysis

```bash
videoanalysis analyze VIDEO [OPTIONS]

Options:
  -o, --output PATH          Output JSON file
  -t, --type TYPE            Analysis type (comprehensive, quick, frames_only, ocr_only, delta_only)
  -s, --sample-rate INT      Seconds between frames (default: 30)
  -m, --max-frames INT       Maximum frames to extract (default: 100)
  --scene-threshold FLOAT    Scene detection threshold 0.0-1.0 (default: 0.3)
  --ocr-confidence FLOAT     OCR confidence threshold 0.0-1.0 (default: 0.6)
  --delta-threshold FLOAT    Delta change threshold 0-255 (default: 15.0)
  --no-cleanup               Keep temporary files
  -v, --verbose              Verbose output
```

### `delta` - Movement Detection

```bash
videoanalysis delta VIDEO [OPTIONS]

Options:
  -o, --output PATH          Output JSON file
  -i, --interval INT         Sampling interval in seconds (default: 5)
  -t, --threshold FLOAT      Change threshold 0-255 (default: 15)
  --key-moments INT          Number of key moments (default: 10)
  --timeline                 Include activity timeline
```

### `frames` - Frame Extraction

```bash
videoanalysis frames VIDEO -o OUTPUT_DIR [OPTIONS]

Options:
  -i, --interval INT         Interval in seconds (default: 30)
  -m, --max-frames INT       Maximum frames (default: 100)
```

### `scenes` - Scene Detection

```bash
videoanalysis scenes VIDEO [OPTIONS]

Options:
  -o, --output PATH          Output JSON file
  -t, --threshold FLOAT      Detection threshold (default: 0.3)
```

### `ocr` - Text Extraction

```bash
videoanalysis ocr VIDEO --timestamps "HH:MM:SS,HH:MM:SS,..." [OPTIONS]

Options:
  -o, --output PATH          Output JSON file
```

### `check-deps` - Dependency Check

```bash
videoanalysis check-deps
```

---

## Python API

### Basic Usage

```python
from videoanalysis import VideoAnalyzer

# Create analyzer
analyzer = VideoAnalyzer(
    sample_interval=30,      # Frame every 30 seconds
    max_frames=100,          # Max 100 frames
    scene_threshold=0.3,     # Scene detection sensitivity
    ocr_confidence=0.6,      # OCR minimum confidence
    delta_threshold=15.0,    # Delta change threshold
)

# Analyze video
result = analyzer.analyze("video.mp4")

# Access results
print(f"Duration: {result.duration}")
print(f"Resolution: {result.resolution}")
print(f"Scenes: {len(result.scenes)}")
print(f"Key moments: {len(result.key_moments)}")
print(f"Text occurrences: {len(result.text_detected)}")
```

### Delta Change Detection Only

```python
from videoanalysis import DeltaChangeDetector, FrameSampler
from pathlib import Path
import tempfile

# Extract frames
temp_dir = Path(tempfile.mkdtemp())
sampler = FrameSampler(temp_dir)
frames = sampler.sample("video.mp4", interval_seconds=5)

# Detect changes
detector = DeltaChangeDetector(threshold=15.0)
changes = detector.detect_changes(frames)

# Find key moments
key_moments = detector.find_key_moments(frames, top_n=10)

# Get activity timeline
timeline = detector.get_activity_timeline(frames, bucket_size=5)
```

### Extract Metadata Only

```python
from videoanalysis import MetadataExtractor

metadata = MetadataExtractor.extract("video.mp4")
print(f"Duration: {metadata.duration_formatted}")
print(f"Resolution: {metadata.width}x{metadata.height}")
print(f"FPS: {metadata.fps}")
print(f"Codec: {metadata.codec}")
```

### Scene Detection Only

```python
from videoanalysis import SceneDetector

detector = SceneDetector(threshold=0.3)
scenes = detector.detect("video.mp4")

for scene in scenes:
    print(f"Scene {scene.scene_index}: {scene.start_time} - {scene.end_time}")
```

---

## Configuration

### Default Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `sample_interval` | 30 | Seconds between frame samples |
| `max_frames` | 100 | Maximum frames to extract |
| `scene_threshold` | 0.3 | Scene detection sensitivity (0.0-1.0) |
| `ocr_confidence` | 0.6 | Minimum OCR confidence (0.0-1.0) |
| `delta_threshold` | 15.0 | Delta change threshold (0-255) |
| `cleanup_temp` | True | Delete temp files after analysis |

### Adjusting for Different Video Types

**Screen recordings (high detail, infrequent changes):**
```python
analyzer = VideoAnalyzer(
    sample_interval=10,     # More frequent sampling
    delta_threshold=10.0,   # Lower threshold for subtle changes
)
```

**Action videos (frequent changes):**
```python
analyzer = VideoAnalyzer(
    sample_interval=5,      # Very frequent sampling
    delta_threshold=25.0,   # Higher threshold to reduce noise
)
```

**Long videos:**
```python
analyzer = VideoAnalyzer(
    sample_interval=60,     # Less frequent sampling
    max_frames=50,          # Fewer frames
)
```

---

## Output Format

### AnalysisResult Structure

```json
{
  "file_path": "/path/to/video.mp4",
  "file_name": "video.mp4",
  "duration": "00:45:00",
  "resolution": [1920, 1080],
  "fps": 30.0,
  "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
  "codec": "h264",
  "file_size_mb": 256.5,
  "scenes": [...],
  "frames": [...],
  "text_detected": [...],
  "delta_changes": [...],
  "key_moments": [...],
  "activity_timeline": [...],
  "summary": "00:45:00 video (1920x1080, 30.0fps, h264). 15 scenes detected...",
  "processing_time_seconds": 45.2,
  "tool_version": "1.0.0"
}
```

### Key Moment Structure

```json
{
  "timestamp": "00:15:30",
  "frame_index": 31,
  "change_type": "major_movement",
  "magnitude": 45.5,
  "change_percent": 35.2,
  "is_key_moment": true,
  "description": "Significant movement/activity (delta: 45.5, 35.2% pixels changed)"
}
```

### Activity Timeline Structure

```json
{
  "start_time": "00:00:00",
  "end_time": "00:02:30",
  "avg_delta": 12.5,
  "max_delta": 35.0,
  "activity_level": "medium",
  "frame_count": 5
}
```

---

## Dependencies

### Required
- **Python 3.9+**: Core runtime
- **FFmpeg**: Video processing (frame extraction, metadata)

### Optional
- **Pillow**: Delta change detection (highly recommended)
- **OpenCV**: Enhanced scene detection
- **pytesseract + Tesseract**: OCR text extraction

### Dependency Check

```bash
$ videoanalysis check-deps

VideoAnalysis Dependency Status
========================================

Required:
  [OK] ffmpeg: Video metadata and frame extraction
  [OK] ffprobe: Video metadata extraction

Optional:
  [OK] tesseract: OCR text extraction
```

---

## Examples

See [EXAMPLES.md](EXAMPLES.md) for 10+ detailed examples including:

1. Basic video analysis
2. Quick analysis mode
3. Delta change detection
4. Finding key moments
5. Activity timeline generation
6. Frame extraction
7. Scene detection
8. OCR text extraction
9. Python API usage
10. Integration with Team Brain tools

---

## Troubleshooting

### FFmpeg not found

```
Error: Missing required dependencies:
ffmpeg: Video metadata and frame extraction
```

**Solution:** Install FFmpeg:
- Windows: `winget install ffmpeg`
- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

### OCR not working

```
Warning: Tesseract not available - OCR skipped
```

**Solution:** Install Tesseract:
- Windows: `winget install tesseract`
- Linux: `sudo apt install tesseract-ocr`

### Video format not supported

**Solution:** Ensure the video is in a supported format (MP4, MOV, AVI, MKV, WebM). Convert if needed:
```bash
ffmpeg -i input.xyz -c:v libx264 output.mp4
```

### Memory issues with large videos

**Solution:** Reduce frame count and increase interval:
```bash
videoanalysis analyze large_video.mp4 -m 30 -s 120
```

---

## Privacy & Security

- **Local Processing Only**: All analysis happens on your machine
- **No External APIs**: Video content never leaves your system
- **Temp File Cleanup**: Extracted frames are deleted after analysis
- **No Telemetry**: No data collection or tracking

---

## Performance

| Video Duration | Expected Time | Memory Usage |
|----------------|---------------|--------------|
| 5 minutes | ~15 seconds | ~200 MB |
| 30 minutes | ~1 minute | ~500 MB |
| 1 hour | ~2 minutes | ~800 MB |
| 3 hours | ~5 minutes | ~1.5 GB |

*Times vary based on analysis type, frame count, and system specs.*

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`python -m pytest`)
5. Submit a pull request

---

## Credits

### Requested By
- **Logan Smith** (Metaphy LLC) - Original concept and tool request via CLIO
- **Delta Change Detection** - Logan's key insight: "don't forget about using just delta change to see video movement"

### Built By
- **ATLAS** (Team Brain) - Primary Developer

### Facilitated By
- **WSL_CLIO** (Team Brain) - Tool Request submission 2026-02-04
- **IRIS** (Team Brain) - Collaborative debugging session

### Part Of
- **Team Brain** - AI Agent Collaboration System
- **Metaphy LLC** - Metaphysics and Computing Research

### Inspired By
- Logan's need for AI agents to analyze video content
- [The historic debugging session](https://metaphysicsandcomputing.com/about-us/f/from-localhost-to-real-time-a-wsl-agents-journey-into-bch) where WSL_CLIO couldn't watch a 45-min video
- Logan's delta change detection insight - a foundational technique for future video analysis

---

## License

MIT License

Copyright (c) 2026 Metaphy LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

**For the Maximum Benefit of Life.**  
**One World. One Family. One Love.**

*Together for all time!*
