# VideoAnalysis - Quick Start Guides

## 5-Minute Guides for Every Team Brain Agent

**Document Version:** 1.0  
**Created:** February 6, 2026  
**Author:** ATLAS (Team Brain)  
**For:** Logan Smith / Metaphy LLC  

---

## 📖 ABOUT THESE GUIDES

Each Team Brain agent has a **5-minute quick-start guide** tailored to their role and workflows. Pick your guide and get started immediately.

**Choose your guide:**
- [Forge (Orchestrator)](#-forge-quick-start)
- [Atlas (Executor)](#-atlas-quick-start)
- [Clio (Linux Agent)](#-clio-quick-start)
- [Nexus (Multi-Platform)](#-nexus-quick-start)
- [Bolt (Free Executor)](#-bolt-quick-start)

---

## 🔥 FORGE QUICK START

**Role:** Orchestrator / Reviewer  
**Time:** 5 minutes  
**Goal:** Learn to use VideoAnalysis for reviewing agent work and analyzing screen recordings

### Step 1: Installation Check

```bash
# Verify VideoAnalysis is available
python -c "from videoanalysis import VideoAnalyzer; print('VideoAnalysis OK')"

# Check dependencies
python videoanalysis.py check-deps
```

**Expected Output:**
```
VideoAnalysis OK
```

### Step 2: First Use - Analyze a Video

```python
# In your Forge session
from videoanalysis import VideoAnalyzer

# Create analyzer with review-focused settings
analyzer = VideoAnalyzer(
    sample_interval=10,    # Detailed sampling
    ocr_confidence=0.5,    # Catch more text
    delta_threshold=10.0   # Sensitive to changes
)

# Analyze an agent's screen recording
result = analyzer.analyze("recording.mp4")
print(f"Duration: {result.duration}")
print(f"Key moments: {len(result.key_moments)}")
print(f"Summary: {result.summary}")
```

### Step 3: Review Workflow - Find Errors in Recordings

```python
# Forge reviewing agent work via video
result = analyzer.analyze("agent_session.mp4")

# Check for error messages in video frames
errors_found = []
for text in result.text_detected:
    if any(kw in text.text.lower() for kw in ['error', 'failed', 'exception']):
        errors_found.append(f"[{text.timestamp}] {text.text}")

if errors_found:
    print(f"[!] Found {len(errors_found)} potential errors:")
    for e in errors_found:
        print(f"  {e}")
else:
    print("[OK] No errors detected in recording")
```

### Step 4: Quick CLI Review

```bash
# Quick analysis from command line
python videoanalysis.py analyze recording.mp4 --type quick

# Focus on key moments
python videoanalysis.py delta recording.mp4 --key-moments 10 --timeline

# Extract text at specific timestamps
python videoanalysis.py ocr recording.mp4 --timestamps "00:01:00,00:05:00"
```

### Next Steps for Forge
1. Read [INTEGRATION_PLAN.md](INTEGRATION_PLAN.md) - Forge section
2. Try [EXAMPLES.md](EXAMPLES.md) - Example 5 (Review workflow)
3. Add video review to your orchestration checklist
4. Use delta detection to find key moments in long recordings

---

## ⚡ ATLAS QUICK START

**Role:** Executor / Builder  
**Time:** 5 minutes  
**Goal:** Learn to use VideoAnalysis for building tools and testing pipelines

### Step 1: Installation Check

```bash
python -c "from videoanalysis import VideoAnalyzer, DeltaChangeDetector; print('OK')"
```

### Step 2: First Use - Build a Video Pipeline

```python
# Atlas building custom video processing
from videoanalysis import VideoAnalyzer, DeltaChangeDetector, FrameSampler
from pathlib import Path

# Full analysis
analyzer = VideoAnalyzer()
result = analyzer.analyze("test_video.mp4")

# Access structured data
print(f"File: {result.file_name}")
print(f"Duration: {result.duration}")
print(f"Resolution: {result.resolution}")
print(f"FPS: {result.fps}")
print(f"Scenes: {len(result.scenes)}")
print(f"Key moments: {len(result.key_moments)}")
```

### Step 3: Use Individual Components

```python
from videoanalysis import DeltaChangeDetector, FrameSampler
from pathlib import Path

# Extract frames manually
sampler = FrameSampler(Path("./my_frames"))
frames = sampler.sample("video.mp4", interval_seconds=5)
print(f"Extracted {len(frames)} frames")

# Run delta detection on extracted frames
detector = DeltaChangeDetector(threshold=10.0)
changes = detector.detect_changes(frames)
moments = detector.find_key_moments(frames, top_n=10)
timeline = detector.get_activity_timeline(frames, bucket_size=5)

print(f"Changes detected: {len(changes)}")
print(f"Key moments: {len(moments)}")

# Cleanup
sampler.cleanup()
```

### Step 4: JSON Output for Integration

```bash
# Save analysis as JSON for other tools to consume
python videoanalysis.py analyze video.mp4 -o analysis.json

# Delta detection with timeline
python videoanalysis.py delta video.mp4 --timeline -o delta.json
```

### Next Steps for Atlas
1. Integrate into Holy Grail automation pipeline
2. Build custom analysis pipelines using components
3. Combine with AudioAnalysis for full media processing
4. Use JSON output as input for other tools

---

## 🐧 CLIO QUICK START

**Role:** Linux / Ubuntu Agent  
**Time:** 5 minutes  
**Goal:** Learn to use VideoAnalysis in Linux environment with CLI focus

### Step 1: Linux Installation

```bash
# Install FFmpeg (if not already installed)
sudo apt install ffmpeg

# Optional: Install Tesseract for OCR
sudo apt install tesseract-ocr

# Clone from GitHub
git clone https://github.com/DonkRonk17/VideoAnalysis.git
cd VideoAnalysis

# Verify
python3 videoanalysis.py check-deps
```

**Expected Output:**
```
Dependency Status Report
========================
Required:
  [OK] ffmpeg - Video metadata and frame extraction
  [OK] ffprobe - Video metadata extraction

Optional:
  [OK] tesseract - OCR text extraction from frames
```

### Step 2: First Use - CLI Analysis

```bash
# Full analysis
python3 videoanalysis.py analyze /path/to/video.mp4

# Quick analysis
python3 videoanalysis.py analyze /path/to/video.mp4 --type quick

# Save results
python3 videoanalysis.py analyze /path/to/video.mp4 -o results.json
```

### Step 3: Batch Processing Scripts

```bash
#!/bin/bash
# analyze_all.sh - Batch analyze videos

VIDEO_DIR="/home/clio/videos"
OUTPUT_DIR="/home/clio/analysis_results"
mkdir -p "$OUTPUT_DIR"

for video in "$VIDEO_DIR"/*.mp4; do
    filename=$(basename "$video" .mp4)
    echo "Analyzing: $filename"
    python3 videoanalysis.py analyze "$video" -o "$OUTPUT_DIR/${filename}.json"
done

echo "Done! Results in $OUTPUT_DIR"
```

### Step 4: Movement Detection on Security Footage

```bash
# Detect all movement with timeline
python3 videoanalysis.py delta /security/cam01.mp4 \
    --key-moments 20 \
    --timeline \
    --threshold 10 \
    -o /reports/cam01_movement.json

# High-sensitivity detection
python3 videoanalysis.py delta /security/cam01.mp4 \
    --threshold 5 \
    --interval 2 \
    --key-moments 50
```

### Step 5: Extract Text from Videos

```bash
# OCR at specific timestamps
python3 videoanalysis.py ocr /tutorials/setup.mp4 \
    --timestamps "00:01:00,00:05:00,00:10:00,00:15:00" \
    -o text_results.json
```

### Next Steps for Clio
1. Create shell aliases: `alias va='python3 /path/to/videoanalysis.py'`
2. Set up cron jobs for automated video analysis
3. Build monitoring scripts for security cameras
4. Report Linux-specific issues via Synapse

---

## 🌐 NEXUS QUICK START

**Role:** Multi-Platform Agent  
**Time:** 5 minutes  
**Goal:** Learn cross-platform usage of VideoAnalysis

### Step 1: Platform Detection and Setup

```python
import platform
from pathlib import Path

print(f"Platform: {platform.system()}")
print(f"Python: {platform.python_version()}")

# Platform-specific FFmpeg install
if platform.system() == "Windows":
    print("Install: winget install ffmpeg")
elif platform.system() == "Linux":
    print("Install: sudo apt install ffmpeg")
elif platform.system() == "Darwin":
    print("Install: brew install ffmpeg")
```

### Step 2: First Use - Cross-Platform Analysis

```python
from videoanalysis import VideoAnalyzer
from pathlib import Path

# VideoAnalyzer uses pathlib internally - works everywhere
analyzer = VideoAnalyzer()

# Analyze video with platform-agnostic path
video = Path("videos") / "test.mp4"
if video.exists():
    result = analyzer.analyze(str(video))
    print(f"Analysis complete: {result.file_name}")
    print(f"Platform: {platform.system()}")
    print(f"Duration: {result.duration}")
```

### Step 3: Platform-Specific Paths

```python
import platform
from pathlib import Path

# Smart path resolution
if platform.system() == "Windows":
    video_dir = Path.home() / "Videos"
elif platform.system() == "Linux":
    video_dir = Path.home() / "videos"
else:  # macOS
    video_dir = Path.home() / "Movies"

# Process all videos regardless of platform
for video in video_dir.glob("*.mp4"):
    analyzer = VideoAnalyzer(max_frames=50)
    result = analyzer.analyze(str(video))
    print(f"[{platform.system()}] {video.name}: {result.duration}")
```

### Step 4: Cross-Platform CLI

```bash
# Same commands work everywhere
python videoanalysis.py analyze video.mp4
python videoanalysis.py delta video.mp4 --timeline
python videoanalysis.py check-deps
```

### Platform Compatibility Notes

| Feature | Windows | Linux | macOS |
|---------|---------|-------|-------|
| FFmpeg | winget install | apt install | brew install |
| Tesseract | winget install | apt install | brew install |
| CLI | python videoanalysis.py | python3 videoanalysis.py | python3 videoanalysis.py |
| Paths | pathlib handles | pathlib handles | pathlib handles |
| Temp files | %TEMP% | /tmp | /tmp |

### Next Steps for Nexus
1. Test on all 3 platforms for consistency
2. Report platform-specific issues
3. Verify JSON output is identical cross-platform
4. Add to multi-platform testing workflows

---

## 🆓 BOLT QUICK START

**Role:** Free Executor (Cline + Grok)  
**Time:** 5 minutes  
**Goal:** Learn to use VideoAnalysis for batch processing without API costs

### Step 1: Verify Free Access

```bash
# No API key required - runs 100% locally!
python videoanalysis.py check-deps

# Zero cloud dependencies
python -c "from videoanalysis import VideoAnalyzer; print('Ready - $0 cost!')"
```

### Step 2: First Use - Free Analysis

```bash
# Run analysis - completely free
python videoanalysis.py analyze video.mp4

# Output goes to stdout or file - no network calls
python videoanalysis.py analyze video.mp4 -o results.json
```

### Step 3: Batch Processing (Save API Costs!)

```bash
# Process multiple videos in sequence
python videoanalysis.py analyze video1.mp4 -o result1.json
python videoanalysis.py analyze video2.mp4 -o result2.json
python videoanalysis.py analyze video3.mp4 -o result3.json

# Quick mode for faster batch processing
python videoanalysis.py analyze video1.mp4 --type quick -o quick1.json
```

### Step 4: Bulk Delta Detection

```bash
# Analyze movement in security/monitoring footage
python videoanalysis.py delta footage1.mp4 --key-moments 20 -o delta1.json
python videoanalysis.py delta footage2.mp4 --key-moments 20 -o delta2.json
```

### Cost Comparison

| Method | Cost | Speed |
|--------|------|-------|
| VideoAnalysis (local) | **$0.00** | Fast |
| Cloud video API | $0.05-0.50/min | Variable |
| Manual review | Time cost | Very slow |

### Next Steps for Bolt
1. Add to Cline task automation
2. Build batch processing scripts
3. Process all pending video tasks
4. Report any issues via Synapse

---

## 📚 ADDITIONAL RESOURCES

**For All Agents:**
- Full Documentation: [README.md](README.md)
- Detailed Examples: [EXAMPLES.md](EXAMPLES.md)
- Integration Plan: [INTEGRATION_PLAN.md](INTEGRATION_PLAN.md)
- Integration Examples: [INTEGRATION_EXAMPLES.md](INTEGRATION_EXAMPLES.md)
- Cheat Sheet: [CHEAT_SHEET.txt](CHEAT_SHEET.txt)

**Support:**
- GitHub Issues: https://github.com/DonkRonk17/VideoAnalysis/issues
- Synapse: Post in THE_SYNAPSE/active/
- Direct: Message ATLAS via SynapseLink

---

**Last Updated:** February 6, 2026  
**Maintained By:** ATLAS (Team Brain)  
**For:** Logan Smith / Metaphy LLC
