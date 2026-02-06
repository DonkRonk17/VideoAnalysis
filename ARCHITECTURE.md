# Architecture Design - VideoAnalysis

**Builder:** ATLAS
**Date:** 2026-02-04
**Protocol:** BUILD_PROTOCOL_V1.md Phase 3

---

## System Overview

```
+------------------+     +------------------+     +------------------+
|   VIDEO INPUT    | --> |  VIDEO ANALYZER  | --> |   JSON OUTPUT    |
| (MP4/MOV/AVI...) |     |  (Core Engine)   |     | (Structured)     |
+------------------+     +------------------+     +------------------+
                                  |
                                  v
                    +---------------------------+
                    |     PROCESSING PIPELINE   |
                    +---------------------------+
                    |                           |
          +---------+---------+       +---------+---------+
          |                   |       |                   |
          v                   v       v                   v
    +----------+        +----------+  +----------+  +----------+
    | METADATA |        |  FRAME   |  |   OCR    |  |  SCENE   |
    | EXTRACT  |        | SAMPLER  |  | ENGINE   |  | DETECTOR |
    +----------+        +----------+  +----------+  +----------+
```

---

## Core Components

### 1. VideoAnalyzer (Main Class)

**Purpose:** Orchestrates all video analysis operations
**File:** `videoanalysis.py`

**Inputs:**
- Video file path (str or Path)
- Analysis configuration (dict)
- Output directory (optional)

**Outputs:**
- AnalysisResult dataclass with all findings
- JSON file (if output_path specified)

**Tools Used:**
- PathBridge: Cross-platform path handling
- ConfigManager: Configuration management
- ErrorRecovery: Error handling

```python
@dataclass
class AnalysisResult:
    file_path: str
    duration: str
    resolution: Tuple[int, int]
    fps: float
    format: str
    scenes: List[Scene]
    frames: List[FrameAnalysis]
    text_detected: List[TextOccurrence]
    summary: str
    metadata: Dict[str, Any]
    processing_time: float
```

---

### 2. MetadataExtractor (Component)

**Purpose:** Extract video metadata using FFmpeg
**Inputs:** Video file path
**Outputs:** VideoMetadata dataclass

```python
@dataclass
class VideoMetadata:
    duration_seconds: float
    duration_formatted: str
    width: int
    height: int
    fps: float
    codec: str
    bitrate: int
    file_size: int
    format_name: str
```

**Tools Used:**
- ProcessWatcher: Monitor FFmpeg subprocess
- TimeSync: Validate timestamps

---

### 3. FrameSampler (Component)

**Purpose:** Extract frames at configurable intervals
**Inputs:** Video path, interval (seconds), max_frames
**Outputs:** List of extracted frame paths

**Algorithm:**
1. Calculate total frames needed based on duration
2. Use FFmpeg to extract frames at intervals
3. Save to temp directory with timestamps
4. Return list of frame paths

**Tools Used:**
- QuickBackup: Backup before extraction
- PathBridge: Temp directory handling

---

### 4. SceneDetector (Component)

**Purpose:** Detect scene changes using OpenCV
**Inputs:** Video path, threshold (0.0-1.0)
**Outputs:** List of Scene objects with timestamps

```python
@dataclass
class Scene:
    start_time: str
    end_time: str
    duration: float
    frame_path: Optional[str]
    description: str
    dominant_colors: List[str]
    is_key_moment: bool
```

**Algorithm:**
1. Compare consecutive frames using histogram difference
2. If difference > threshold, mark as scene change
3. Extract representative frame for each scene
4. Analyze frame for dominant colors

**Tools Used:**
- ProcessWatcher: Monitor OpenCV processing
- LogHunter: Track detection logs

---

### 5. OCREngine (Component)

**Purpose:** Extract text from video frames using Tesseract
**Inputs:** Frame image path
**Outputs:** List of TextOccurrence objects

```python
@dataclass
class TextOccurrence:
    text: str
    timestamp: str
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]]
    frame_path: str
```

**Algorithm:**
1. Pre-process frame (grayscale, contrast enhancement)
2. Run Tesseract OCR with language detection
3. Filter by confidence threshold (default: 60%)
4. Group text by location/timestamp

**Tools Used:**
- RegexLab: Pattern matching for text cleanup
- ErrorRecovery: Handle OCR failures

---

### 6. OutputFormatter (Component)

**Purpose:** Format analysis results for output
**Inputs:** AnalysisResult
**Outputs:** JSON, Markdown, or plain text

**Tools Used:**
- JSONQuery: Validate JSON structure
- DataConvert: Format conversion
- ContextCompressor: Compress large outputs

---

## Data Flow

```
1. INPUT VALIDATION
   +------------------+
   | VideoAnalyzer    |
   | .analyze(path)   |
   +--------+---------+
            |
            v
2. DEPENDENCY CHECK
   +------------------+
   | EnvGuard         |
   | .check_deps()    |
   +--------+---------+
            |
            v
3. METADATA EXTRACTION
   +------------------+
   | MetadataExtractor|
   | .extract()       |
   +--------+---------+
            |
            v
4. PARALLEL PROCESSING
   +------------------+------------------+------------------+
   |                  |                  |                  |
   v                  v                  v                  v
   FrameSampler    SceneDetector    OCREngine        (Future)
   .sample()       .detect()        .extract()       AudioAnalysis
   |                  |                  |
   +------------------+------------------+
            |
            v
5. RESULT AGGREGATION
   +------------------+
   | AnalysisResult   |
   | (dataclass)      |
   +--------+---------+
            |
            v
6. OUTPUT FORMATTING
   +------------------+
   | OutputFormatter  |
   | .format()        |
   +--------+---------+
            |
            v
7. SAVE & CLEANUP
   - Write JSON/Markdown
   - Delete temp frames
   - Return result
```

---

## Error Handling Strategy

### Error Categories

| Category | Example | Handling |
|----------|---------|----------|
| File Not Found | Video path doesn't exist | Raise FileNotFoundError with helpful message |
| Dependency Missing | FFmpeg not installed | Return DependencyError with install instructions |
| Codec Unsupported | Obscure codec | Attempt with generic decoder, warn user |
| Processing Failed | Frame extraction fails | Retry 3x, then partial result with warning |
| Memory Error | Video too large | Stream processing, reduce resolution |
| Permission Error | Can't write output | Use temp dir, warn user |

### Error Recovery Pattern

```python
def safe_process(func, *args, retries=3, **kwargs):
    """Error recovery wrapper using ErrorRecovery tool pattern."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except RecoverableError as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                raise
        except FatalError:
            raise  # Don't retry fatal errors
```

---

## Configuration Strategy

### Configuration File: `videoanalysis_config.json`

```json
{
    "version": "1.0.0",
    "defaults": {
        "sample_interval_seconds": 30,
        "max_frames": 100,
        "scene_threshold": 0.3,
        "ocr_confidence_threshold": 0.6,
        "ocr_languages": ["eng"],
        "output_format": "json",
        "cleanup_temp_files": true,
        "max_processing_time_seconds": 600
    },
    "paths": {
        "ffmpeg": "ffmpeg",
        "tesseract": "tesseract",
        "temp_dir": null
    },
    "logging": {
        "level": "INFO",
        "file": null
    }
}
```

### Configuration Validation

Using ConfigManager:
- Validate all required fields
- Apply defaults for missing values
- Check path existence for executables
- Verify numeric ranges

---

## CLI Interface

```bash
# Basic usage
videoanalysis analyze video.mp4

# With options
videoanalysis analyze video.mp4 \
    --sample-rate 30 \
    --output results.json \
    --format json \
    --verbose

# Quick mode (faster, less detail)
videoanalysis analyze video.mp4 --quick

# OCR only mode
videoanalysis ocr video.mp4 --timestamps "00:10:00,00:20:00"

# Scene detection only
videoanalysis scenes video.mp4 --threshold 0.4

# Frame extraction only
videoanalysis frames video.mp4 --interval 60 --output-dir ./frames/

# Check dependencies
videoanalysis check-deps
```

---

## Performance Considerations

### Memory Management
- Stream video processing (don't load entire file)
- Process frames in batches
- Delete temp files as we go
- Use generators for large result sets

### Processing Optimization
- Parallel frame extraction
- Skip similar consecutive frames
- Adaptive sampling (more frames at scene changes)
- Cache FFmpeg probe results

### Target Performance
- 45-min video in <5 minutes
- Memory usage <2GB for 1080p video
- CPU usage configurable (thread count)

---

## File Structure

```
VideoAnalysis/
+-- videoanalysis.py         # Main module
+-- test_videoanalysis.py    # Test suite (10+ unit, 5+ integration)
+-- README.md                # Documentation (400+ lines)
+-- EXAMPLES.md              # 10+ examples
+-- CHEAT_SHEET.txt          # Quick reference
+-- requirements.txt         # Dependencies
+-- setup.py                 # Package setup
+-- LICENSE                  # MIT License
+-- .gitignore               # Git ignores
+-- BUILD_COVERAGE_PLAN.md   # Phase 1 output
+-- BUILD_AUDIT.md           # Phase 2 output (this file)
+-- ARCHITECTURE.md          # Phase 3 output (this file)
+-- BUILD_LOG.md             # Implementation log
+-- BUILD_REPORT.md          # Phase 8 output
+-- branding/
|   +-- BRANDING_PROMPTS.md  # DALL-E prompts
+-- config/
|   +-- videoanalysis_config.json  # Default config
```

---

## Security Considerations

### Privacy Protection
- NO external API calls for video content
- All processing happens locally
- Temp files deleted after use
- No telemetry or logging of video content

### Input Validation
- Sanitize file paths (prevent traversal)
- Validate file is actually a video
- Check file size before processing
- Timeout long-running processes

---

## Future Extensibility

### Planned Features (v2.0)
- Audio analysis integration (when AudioAnalysis tool is built)
- Custom AI vision model integration (local models only)
- Video comparison (diff two videos)
- Batch processing with parallel videos
- Watch mode (monitor folder for new videos)

### Plugin Architecture
- Define abstract base class for analyzers
- Allow custom analyzers to be registered
- Support for custom output formatters

---

**BUILD_PROTOCOL_V1.md Phase 3: COMPLETE**

*Built by ATLAS for Team Brain*
*Together for all time!*
