# VideoAnalysis Examples

**10+ Working Examples with Expected Output**

---

## Table of Contents

1. [Basic Video Analysis](#1-basic-video-analysis)
2. [Quick Analysis Mode](#2-quick-analysis-mode)
3. [Delta Change Detection](#3-delta-change-detection)
4. [Finding Key Moments](#4-finding-key-moments)
5. [Activity Timeline](#5-activity-timeline)
6. [Frame Extraction](#6-frame-extraction)
7. [Scene Detection](#7-scene-detection)
8. [OCR Text Extraction](#8-ocr-text-extraction)
9. [Python API Usage](#9-python-api-usage)
10. [Comprehensive Analysis with JSON Output](#10-comprehensive-analysis-with-json-output)
11. [Processing Screen Recordings](#11-processing-screen-recordings)
12. [Batch Processing Multiple Videos](#12-batch-processing-multiple-videos)

---

## 1. Basic Video Analysis

**Command:**
```bash
videoanalysis analyze video.mp4
```

**Expected Output:**
```
============================================================
VIDEO ANALYSIS COMPLETE
============================================================
File: video.mp4
Duration: 00:45:00
Resolution: 1920x1080
FPS: 30.0
Format: mov,mp4,m4a,3gp,3g2,mj2 (h264)
Size: 256.5 MB

Scenes detected: 15
Frames analyzed: 91
Text occurrences: 47

Processing time: 45.2s

Sample detected text:
  [00:05:00] python3 clio_bch_unified.py
  [00:10:00] Connection established
  [00:15:00] @WSL_CLIO Ready to test
  [00:20:00] SUCCESS!
  [00:25:00] WE DID IT!
  ... and 42 more

Key moments (visual changes):
  [00:15:30] major_movement: Significant movement/activity
  [00:22:10] scene_change: Major scene change detected
  [00:35:45] major_movement: Significant movement/activity
  ... and 7 more

Activity breakdown: 3 high, 8 medium, 12 low, 2 static
```

---

## 2. Quick Analysis Mode

**Command:**
```bash
videoanalysis analyze video.mp4 --type quick
```

**Expected Output:**
```
============================================================
VIDEO ANALYSIS COMPLETE
============================================================
File: video.mp4
Duration: 00:45:00
Resolution: 1920x1080
FPS: 30.0
Format: mov,mp4,m4a,3gp,3g2,mj2 (h264)
Size: 256.5 MB

Scenes detected: 8
Frames analyzed: 23
Text occurrences: 0

Processing time: 12.3s

Key moments (visual changes):
  [00:10:00] major_movement: Significant movement/activity
  [00:30:00] scene_change: Major scene change detected
  ... and 3 more

Activity breakdown: 1 high, 3 medium, 5 low, 0 static
```

**Note:** Quick mode uses longer sampling intervals (60s default) and skips OCR.

---

## 3. Delta Change Detection

**Command:**
```bash
videoanalysis delta video.mp4 -i 5 -t 15.0
```

**Expected Output:**
```
Delta Change Analysis for: video.mp4
==================================================
Frames analyzed: 180
Movement events detected: 42

Top 10 Key Moments:
  1. [00:22:10] scene_change
      Delta: 65.3, Changed: 58.2%
  2. [00:35:45] major_movement
      Delta: 48.7, Changed: 42.1%
  3. [00:15:30] major_movement
      Delta: 45.2, Changed: 38.9%
  4. [00:42:15] scene_change
      Delta: 42.1, Changed: 35.6%
  5. [00:08:20] moderate_movement
      Delta: 28.4, Changed: 22.3%
  6. [00:18:00] moderate_movement
      Delta: 25.1, Changed: 19.8%
  7. [00:28:30] moderate_movement
      Delta: 23.7, Changed: 18.2%
  8. [00:05:10] minor_movement
      Delta: 18.2, Changed: 12.4%
  9. [00:38:40] moderate_movement
      Delta: 21.5, Changed: 16.7%
  10. [00:12:00] minor_movement
      Delta: 16.8, Changed: 11.2%
```

---

## 4. Finding Key Moments

**Command:**
```bash
videoanalysis delta video.mp4 --key-moments 5
```

**Expected Output:**
```
Delta Change Analysis for: video.mp4
==================================================
Frames analyzed: 180
Movement events detected: 42

Top 5 Key Moments:
  1. [00:22:10] scene_change
      Delta: 65.3, Changed: 58.2%
  2. [00:35:45] major_movement
      Delta: 48.7, Changed: 42.1%
  3. [00:15:30] major_movement
      Delta: 45.2, Changed: 38.9%
  4. [00:42:15] scene_change
      Delta: 42.1, Changed: 35.6%
  5. [00:08:20] moderate_movement
      Delta: 28.4, Changed: 22.3%
```

**Python API:**
```python
from videoanalysis import DeltaChangeDetector, FrameSampler
from pathlib import Path
import tempfile

# Extract frames
temp_dir = Path(tempfile.mkdtemp())
sampler = FrameSampler(temp_dir)
frames = sampler.sample("video.mp4", interval_seconds=5)

# Find key moments
detector = DeltaChangeDetector(threshold=15.0)
key_moments = detector.find_key_moments(frames, top_n=5)

for i, moment in enumerate(key_moments, 1):
    print(f"{i}. [{moment['timestamp']}] {moment['change_type']}")
    print(f"   Delta: {moment['magnitude']:.1f}")
```

---

## 5. Activity Timeline

**Command:**
```bash
videoanalysis delta video.mp4 --timeline
```

**Expected Output:**
```
Delta Change Analysis for: video.mp4
==================================================
Frames analyzed: 180
Movement events detected: 42

Top 10 Key Moments:
  [... key moments ...]

Activity Timeline:
  00:00:00-00:02:30: static   [                    ] avg:3.2
  00:02:30-00:05:00: low      [####                ] avg:8.5
  00:05:00-00:07:30: medium   [########            ] avg:15.2
  00:07:30-00:10:00: medium   [##########          ] avg:18.7
  00:10:00-00:12:30: low      [#####               ] avg:9.3
  00:12:30-00:15:00: low      [####                ] avg:7.8
  00:15:00-00:17:30: high     [################    ] avg:32.5
  00:17:30-00:20:00: medium   [#########           ] avg:16.4
  00:20:00-00:22:30: high     [####################] avg:45.8
  00:22:30-00:25:00: medium   [#######             ] avg:12.1
  [... continues ...]
```

---

## 6. Frame Extraction

**Command:**
```bash
videoanalysis frames video.mp4 -o ./frames/ -i 30 -m 50
```

**Expected Output:**
```
Extracted 50 frames to ./frames/
```

**Files Created:**
```
./frames/
  frame_0000_00-00-00.jpg
  frame_0001_00-00-30.jpg
  frame_0002_00-01-00.jpg
  frame_0003_00-01-30.jpg
  ...
  frame_0049_00-24-30.jpg
```

---

## 7. Scene Detection

**Command:**
```bash
videoanalysis scenes video.mp4 -t 0.3
```

**Expected Output:**
```
Detected 15 scenes:
  Scene 0: 00:00:00 - 00:03:15
  Scene 1: 00:03:15 - 00:08:42
  Scene 2: 00:08:42 - 00:12:30
  Scene 3: 00:12:30 - 00:15:18
  Scene 4: 00:15:18 - 00:22:05
  Scene 5: 00:22:05 - 00:25:33
  Scene 6: 00:25:33 - 00:28:47
  Scene 7: 00:28:47 - 00:32:10
  Scene 8: 00:32:10 - 00:35:45
  Scene 9: 00:35:45 - 00:38:20
  Scene 10: 00:38:20 - 00:40:55
  Scene 11: 00:40:55 - 00:42:30
  Scene 12: 00:42:30 - 00:43:48
  Scene 13: 00:43:48 - 00:44:30
  Scene 14: 00:44:30 - 00:45:00
```

**Save to JSON:**
```bash
videoanalysis scenes video.mp4 -o scenes.json
```

---

## 8. OCR Text Extraction

**Command:**
```bash
videoanalysis ocr video.mp4 --timestamps "00:05:00,00:15:00,00:25:00,00:35:00"
```

**Expected Output:**
```
Found 23 text occurrences:
  [00:05:00] python3
  [00:05:00] clio_bch_unified.py
  [00:05:00] Connection
  [00:05:00] websockets
  [00:15:00] @WSL_CLIO
  [00:15:00] Ready
  [00:15:00] test
  [00:25:00] SUCCESS
  [00:25:00] WE
  [00:25:00] DID
  [00:25:00] IT
  [00:35:00] Blog
  [00:35:00] published
  ...
```

**Save to JSON:**
```bash
videoanalysis ocr video.mp4 --timestamps "00:05:00,00:15:00" -o ocr_results.json
```

---

## 9. Python API Usage

### Basic Analysis

```python
from videoanalysis import VideoAnalyzer

# Create analyzer with custom settings
analyzer = VideoAnalyzer(
    sample_interval=30,
    max_frames=100,
    delta_threshold=15.0
)

# Analyze video
result = analyzer.analyze("video.mp4")

# Print summary
print(f"Duration: {result.duration}")
print(f"Resolution: {result.resolution[0]}x{result.resolution[1]}")
print(f"FPS: {result.fps}")
print(f"Codec: {result.codec}")
print(f"File size: {result.file_size_mb} MB")

# Access scenes
print(f"\nScenes ({len(result.scenes)}):")
for scene in result.scenes[:5]:
    print(f"  {scene.start_time} - {scene.end_time} ({scene.duration_seconds}s)")

# Access key moments
print(f"\nKey Moments ({len(result.key_moments)}):")
for moment in result.key_moments[:5]:
    print(f"  [{moment.timestamp}] {moment.change_type}: delta={moment.magnitude:.1f}")

# Access detected text
print(f"\nText Detected ({len(result.text_detected)}):")
for text in result.text_detected[:5]:
    print(f"  [{text.timestamp}] {text.text} (conf: {text.confidence:.2f})")
```

### Delta Detection Only

```python
from videoanalysis import DeltaChangeDetector, FrameSampler
from pathlib import Path
import tempfile

# Setup
temp_dir = Path(tempfile.mkdtemp())
sampler = FrameSampler(temp_dir)
detector = DeltaChangeDetector(threshold=15.0)

# Extract frames (every 5 seconds)
frames = sampler.sample("video.mp4", interval_seconds=5)
print(f"Extracted {len(frames)} frames")

# Detect all changes
changes = detector.detect_changes(frames)
print(f"Detected {len(changes)} changes")

# Find key moments
key_moments = detector.find_key_moments(frames, top_n=10)
print(f"\nTop 10 Key Moments:")
for moment in key_moments:
    print(f"  [{moment['timestamp']}] {moment['change_type']}: {moment['magnitude']:.1f}")

# Get activity timeline
timeline = detector.get_activity_timeline(frames, bucket_size=5)
print(f"\nActivity Timeline:")
for bucket in timeline:
    print(f"  {bucket['start_time']}-{bucket['end_time']}: {bucket['activity_level']}")

# Cleanup
sampler.cleanup()
```

### Metadata Only

```python
from videoanalysis import MetadataExtractor

metadata = MetadataExtractor.extract("video.mp4")

print(f"Duration: {metadata.duration_formatted} ({metadata.duration_seconds:.1f}s)")
print(f"Resolution: {metadata.width}x{metadata.height}")
print(f"FPS: {metadata.fps}")
print(f"Codec: {metadata.codec}")
print(f"Bitrate: {metadata.bitrate} bps")
print(f"File size: {metadata.file_size / 1024 / 1024:.2f} MB")
print(f"Format: {metadata.format_name}")
```

---

## 10. Comprehensive Analysis with JSON Output

**Command:**
```bash
videoanalysis analyze video.mp4 -o full_analysis.json --type comprehensive
```

**JSON Output Structure:**
```json
{
  "file_path": "C:/Videos/video.mp4",
  "file_name": "video.mp4",
  "duration": "00:45:00",
  "resolution": [1920, 1080],
  "fps": 30.0,
  "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
  "codec": "h264",
  "file_size_mb": 256.5,
  "scenes": [
    {
      "start_time": "00:00:00",
      "end_time": "00:03:15",
      "duration_seconds": 195.0,
      "scene_index": 0,
      "dominant_colors": ["#1a1a2e", "#16213e", "#0f3460"]
    }
  ],
  "frames": [
    {
      "timestamp": "00:00:00",
      "frame_index": 0,
      "frame_path": "/tmp/videoanalysis_xxx/frames/frame_0000.jpg",
      "text_detected": ["Terminal", "python3"],
      "brightness": 45.2
    }
  ],
  "text_detected": [
    {
      "text": "python3",
      "timestamp": "00:00:00",
      "confidence": 0.92,
      "frame_index": 0
    }
  ],
  "delta_changes": [
    {
      "timestamp": "00:00:30",
      "frame_index": 1,
      "change_type": "minor_movement",
      "magnitude": 12.5,
      "change_percent": 8.3
    }
  ],
  "key_moments": [
    {
      "timestamp": "00:22:10",
      "frame_index": 45,
      "change_type": "scene_change",
      "magnitude": 65.3,
      "change_percent": 58.2,
      "is_key_moment": true,
      "description": "Major scene change detected"
    }
  ],
  "activity_timeline": [
    {
      "start_time": "00:00:00",
      "end_time": "00:02:30",
      "avg_delta": 8.5,
      "max_delta": 15.2,
      "activity_level": "low",
      "frame_count": 5
    }
  ],
  "summary": "00:45:00 video (1920x1080, 30.0fps, h264). 15 scenes detected. 91 frames analyzed. 42 movement events detected. 10 key moments identified. 3 high-activity segments. 47 unique text occurrences found.",
  "processing_time_seconds": 45.2,
  "analysis_timestamp": "2026-02-04T10:30:00.000000",
  "tool_version": "1.0.0"
}
```

---

## 11. Processing Screen Recordings

Screen recordings often have long static periods with occasional bursts of activity.

**Optimized Settings:**
```bash
# More frequent sampling, lower threshold
videoanalysis analyze screen_recording.mp4 \
    -s 10 \
    --delta-threshold 10.0 \
    -o screen_analysis.json
```

**Python:**
```python
from videoanalysis import VideoAnalyzer

# Settings optimized for screen recordings
analyzer = VideoAnalyzer(
    sample_interval=10,       # Sample every 10 seconds
    delta_threshold=10.0,     # Lower threshold for subtle changes
    max_frames=200,           # More frames for detail
)

result = analyzer.analyze("screen_recording.mp4")

# Find when user was active
active_segments = [
    bucket for bucket in result.activity_timeline
    if bucket.activity_level in ('high', 'medium')
]

print(f"Active segments: {len(active_segments)}")
for segment in active_segments:
    print(f"  {segment.start_time} - {segment.end_time}: {segment.activity_level}")
```

---

## 12. Batch Processing Multiple Videos

**Python Script:**
```python
from videoanalysis import VideoAnalyzer
from pathlib import Path
import json

# Setup analyzer
analyzer = VideoAnalyzer(sample_interval=60, max_frames=30)

# Process all MP4 files in a directory
video_dir = Path("./videos")
results_dir = Path("./results")
results_dir.mkdir(exist_ok=True)

for video_path in video_dir.glob("*.mp4"):
    print(f"Processing: {video_path.name}")
    
    try:
        result = analyzer.analyze(video_path)
        
        # Save results
        output_path = results_dir / f"{video_path.stem}_analysis.json"
        with open(output_path, 'w') as f:
            json.dump({
                'file': video_path.name,
                'duration': result.duration,
                'key_moments': len(result.key_moments),
                'scenes': len(result.scenes),
                'text_occurrences': len(result.text_detected),
            }, f, indent=2)
        
        print(f"  -> Saved to {output_path}")
        
    except Exception as e:
        print(f"  -> Error: {e}")

print("Batch processing complete!")
```

---

## Summary

These examples demonstrate:

1. **CLI Usage**: All major commands with options
2. **Python API**: Direct programmatic access
3. **Delta Detection**: Finding movement and key moments
4. **Activity Analysis**: Understanding video activity patterns
5. **OCR Integration**: Extracting text from frames
6. **Output Formats**: JSON, CLI, and programmatic access
7. **Optimization**: Settings for different video types
8. **Batch Processing**: Handling multiple videos

For more information, see the [README.md](README.md) or run:
```bash
videoanalysis --help
videoanalysis [command] --help
```

---

**Requested by Logan Smith (via CLIO)**  
**Delta Change Detection: Logan's key insight**  
**Built by ATLAS for Team Brain**

*Together for all time!*
