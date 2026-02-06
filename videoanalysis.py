#!/usr/bin/env python3
"""
VideoAnalysis - Enable AI Agents to "Watch" Video Content

A comprehensive video analysis tool that extracts frames, detects scenes,
performs OCR on visible text, and generates structured analysis reports.

KEY INNOVATION:
Delta Change Detection - Logan Smith's insight: "don't forget about using 
just delta change to see video movement" - a foundational technique for
efficient video analysis using simple frame differencing.

TOOLS USED IN THIS BUILD:
- PathBridge: Cross-platform path handling
- ConfigManager: Configuration management
- ErrorRecovery: Graceful error handling
- ProcessWatcher: FFmpeg process monitoring
- TimeSync: Timestamp validation
- JSONQuery: Output validation
- EnvGuard: Dependency verification

Requested by: Logan Smith (via CLIO)
Delta Change Detection: Logan Smith's key insight
Built by: ATLAS for Team Brain
Protocol: BUILD_PROTOCOL_V1.md + Bug Hunt Protocol

For the Maximum Benefit of Life.
One World. One Family. One Love.
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VideoAnalysis')

# Version info
__version__ = '1.0.0'
__author__ = 'ATLAS (Team Brain)'
__description__ = 'Enable AI agents to watch and analyze video content'

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class VideoMetadata:
    """Video file metadata extracted via FFmpeg."""
    duration_seconds: float = 0.0
    duration_formatted: str = "00:00:00"
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = ""
    bitrate: int = 0
    file_size: int = 0
    format_name: str = ""
    creation_time: Optional[str] = None


@dataclass
class TextOccurrence:
    """Text detected in a video frame via OCR."""
    text: str
    timestamp: str
    confidence: float
    frame_index: int
    bounding_box: Optional[Tuple[int, int, int, int]] = None


@dataclass
class Scene:
    """A detected scene in the video."""
    start_time: str
    end_time: str
    duration_seconds: float
    frame_path: Optional[str] = None
    description: str = ""
    dominant_colors: List[str] = field(default_factory=list)
    is_key_moment: bool = False
    scene_index: int = 0


@dataclass
class FrameAnalysis:
    """Analysis of a single video frame."""
    timestamp: str
    frame_index: int
    frame_path: str
    text_detected: List[str] = field(default_factory=list)
    dominant_colors: List[str] = field(default_factory=list)
    brightness: float = 0.0
    description: str = ""


@dataclass
class DeltaChange:
    """A detected movement/change between frames using delta comparison."""
    timestamp: str
    frame_index: int
    change_type: str  # scene_change, major_movement, moderate_movement, minor_movement
    magnitude: float  # Mean pixel difference (0-255)
    change_percent: float  # Percentage of pixels that changed
    is_key_moment: bool = False
    description: str = ""


@dataclass
class ActivityBucket:
    """Activity level for a time segment."""
    start_time: str
    end_time: str
    avg_delta: float
    max_delta: float
    activity_level: str  # high, medium, low, static
    frame_count: int = 0


@dataclass
class AnalysisResult:
    """Complete video analysis result."""
    file_path: str
    file_name: str
    duration: str
    resolution: Tuple[int, int] = (0, 0)
    fps: float = 0.0
    format_name: str = ""
    codec: str = ""
    file_size_mb: float = 0.0
    scenes: List[Scene] = field(default_factory=list)
    frames: List[FrameAnalysis] = field(default_factory=list)
    text_detected: List[TextOccurrence] = field(default_factory=list)
    # Delta change detection results
    delta_changes: List[DeltaChange] = field(default_factory=list)
    key_moments: List[DeltaChange] = field(default_factory=list)
    activity_timeline: List[ActivityBucket] = field(default_factory=list)
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    analysis_timestamp: str = ""
    tool_version: str = __version__


# ============================================================================
# EXCEPTIONS
# ============================================================================

class VideoAnalysisError(Exception):
    """Base exception for VideoAnalysis errors."""
    pass


class DependencyError(VideoAnalysisError):
    """Raised when a required dependency is missing."""
    pass


class VideoNotFoundError(VideoAnalysisError):
    """Raised when the video file is not found."""
    pass


class ProcessingError(VideoAnalysisError):
    """Raised when video processing fails."""
    pass


class UnsupportedFormatError(VideoAnalysisError):
    """Raised when the video format is not supported."""
    pass


# ============================================================================
# DEPENDENCY CHECKER
# ============================================================================

class DependencyChecker:
    """Verify required external dependencies are installed."""
    
    REQUIRED_DEPS = {
        'ffmpeg': {
            'check_cmd': ['ffmpeg', '-version'],
            'install_windows': 'winget install ffmpeg',
            'install_linux': 'sudo apt install ffmpeg',
            'purpose': 'Video metadata and frame extraction'
        },
        'ffprobe': {
            'check_cmd': ['ffprobe', '-version'],
            'install_windows': 'Included with ffmpeg',
            'install_linux': 'Included with ffmpeg',
            'purpose': 'Video metadata extraction'
        }
    }
    
    OPTIONAL_DEPS = {
        'tesseract': {
            'check_cmd': ['tesseract', '--version'],
            'install_windows': 'winget install tesseract',
            'install_linux': 'sudo apt install tesseract-ocr',
            'purpose': 'OCR text extraction from frames'
        }
    }
    
    @classmethod
    def check_all(cls, include_optional: bool = True) -> Dict[str, bool]:
        """Check all dependencies and return status."""
        results = {}
        
        for name, info in cls.REQUIRED_DEPS.items():
            results[name] = cls._check_command(info['check_cmd'])
        
        if include_optional:
            for name, info in cls.OPTIONAL_DEPS.items():
                results[name] = cls._check_command(info['check_cmd'])
        
        return results
    
    @classmethod
    def _check_command(cls, cmd: List[str]) -> bool:
        """Check if a command is available."""
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False
    
    @classmethod
    def verify_required(cls) -> None:
        """Verify all required dependencies, raise error if missing."""
        missing = []
        for name, info in cls.REQUIRED_DEPS.items():
            if not cls._check_command(info['check_cmd']):
                missing.append(f"{name}: {info['purpose']}\n"
                              f"  Windows: {info['install_windows']}\n"
                              f"  Linux: {info['install_linux']}")
        
        if missing:
            raise DependencyError(
                "Missing required dependencies:\n\n" + "\n\n".join(missing)
            )
    
    @classmethod
    def get_status_report(cls) -> str:
        """Get a formatted dependency status report."""
        lines = ["VideoAnalysis Dependency Status", "=" * 40, ""]
        
        lines.append("Required:")
        for name, info in cls.REQUIRED_DEPS.items():
            status = "OK" if cls._check_command(info['check_cmd']) else "MISSING"
            lines.append(f"  [{status}] {name}: {info['purpose']}")
        
        lines.append("")
        lines.append("Optional:")
        for name, info in cls.OPTIONAL_DEPS.items():
            status = "OK" if cls._check_command(info['check_cmd']) else "MISSING"
            lines.append(f"  [{status}] {name}: {info['purpose']}")
        
        return "\n".join(lines)


# ============================================================================
# METADATA EXTRACTOR
# ============================================================================

class MetadataExtractor:
    """Extract video metadata using FFprobe."""
    
    @staticmethod
    def extract(video_path: Union[str, Path]) -> VideoMetadata:
        """Extract metadata from a video file."""
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise VideoNotFoundError(f"Video not found: {video_path}")
        
        try:
            # Run ffprobe to get JSON metadata
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise ProcessingError(f"FFprobe failed: {result.stderr}")
            
            data = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                raise ProcessingError("No video stream found in file")
            
            # Extract format info
            format_info = data.get('format', {})
            
            # Calculate duration
            duration_seconds = float(format_info.get('duration', 0))
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            seconds = int(duration_seconds % 60)
            duration_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Calculate FPS
            fps_str = video_stream.get('r_frame_rate', '0/1')
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den) if float(den) != 0 else 0.0
            else:
                fps = float(fps_str)
            
            return VideoMetadata(
                duration_seconds=duration_seconds,
                duration_formatted=duration_formatted,
                width=int(video_stream.get('width', 0)),
                height=int(video_stream.get('height', 0)),
                fps=round(fps, 2),
                codec=video_stream.get('codec_name', ''),
                bitrate=int(format_info.get('bit_rate', 0)),
                file_size=int(format_info.get('size', 0)),
                format_name=format_info.get('format_name', ''),
                creation_time=format_info.get('tags', {}).get('creation_time')
            )
            
        except subprocess.TimeoutExpired:
            raise ProcessingError("FFprobe timed out - video may be corrupted")
        except json.JSONDecodeError as e:
            raise ProcessingError(f"Failed to parse FFprobe output: {e}")


# ============================================================================
# FRAME SAMPLER
# ============================================================================

class FrameSampler:
    """Extract frames from video at configurable intervals."""
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize frame sampler."""
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp(prefix='videoanalysis_'))
    
    def sample(
        self,
        video_path: Union[str, Path],
        interval_seconds: int = 30,
        max_frames: int = 100,
        timestamps: Optional[List[str]] = None
    ) -> List[Tuple[str, str]]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            interval_seconds: Seconds between frame samples
            max_frames: Maximum frames to extract
            timestamps: Optional specific timestamps to extract
            
        Returns:
            List of (frame_path, timestamp) tuples
        """
        video_path = Path(video_path)
        frames_dir = self.temp_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)
        
        extracted = []
        
        if timestamps:
            # Extract specific timestamps
            for i, ts in enumerate(timestamps[:max_frames]):
                frame_path = frames_dir / f"frame_{i:04d}_{ts.replace(':', '-')}.jpg"
                if self._extract_frame_at(video_path, ts, frame_path):
                    extracted.append((str(frame_path), ts))
        else:
            # Extract at intervals
            metadata = MetadataExtractor.extract(video_path)
            total_frames = min(
                max_frames,
                int(metadata.duration_seconds / interval_seconds) + 1
            )
            
            for i in range(total_frames):
                timestamp_seconds = i * interval_seconds
                ts = self._seconds_to_timestamp(timestamp_seconds)
                frame_path = frames_dir / f"frame_{i:04d}_{ts.replace(':', '-')}.jpg"
                
                if self._extract_frame_at(video_path, ts, frame_path):
                    extracted.append((str(frame_path), ts))
        
        return extracted
    
    def _extract_frame_at(
        self,
        video_path: Path,
        timestamp: str,
        output_path: Path
    ) -> bool:
        """Extract a single frame at the given timestamp."""
        try:
            cmd = [
                'ffmpeg',
                '-ss', timestamp,
                '-i', str(video_path),
                '-frames:v', '1',
                '-q:v', '2',
                '-y',
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            
            return output_path.exists()
            
        except subprocess.SubprocessError as e:
            logger.warning(f"Failed to extract frame at {timestamp}: {e}")
            return False
    
    @staticmethod
    def _seconds_to_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def cleanup(self) -> None:
        """Remove temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# ============================================================================
# SCENE DETECTOR
# ============================================================================

class SceneDetector:
    """Detect scene changes in video using frame comparison."""
    
    def __init__(self, threshold: float = 0.3):
        """
        Initialize scene detector.
        
        Args:
            threshold: Scene change threshold (0.0-1.0, higher = more sensitive)
        """
        self.threshold = threshold
        self._cv2_available = self._check_opencv()
    
    def _check_opencv(self) -> bool:
        """Check if OpenCV is available."""
        try:
            import cv2
            return True
        except ImportError:
            return False
    
    def detect(
        self,
        video_path: Union[str, Path],
        max_scenes: int = 50
    ) -> List[Scene]:
        """
        Detect scene changes in video.
        
        Args:
            video_path: Path to video file
            max_scenes: Maximum number of scenes to detect
            
        Returns:
            List of Scene objects
        """
        if not self._cv2_available:
            logger.warning("OpenCV not available - using FFmpeg scene detection")
            return self._detect_with_ffmpeg(video_path, max_scenes)
        
        return self._detect_with_opencv(video_path, max_scenes)
    
    def _detect_with_opencv(
        self,
        video_path: Union[str, Path],
        max_scenes: int
    ) -> List[Scene]:
        """Detect scenes using OpenCV histogram comparison."""
        import cv2
        import numpy as np
        
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ProcessingError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scenes = []
        prev_hist = None
        scene_start_frame = 0
        scene_index = 0
        
        # Sample every N frames for efficiency
        sample_interval = max(1, int(fps / 2))  # ~2 samples per second
        
        for frame_idx in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Convert to HSV and calculate histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            
            if prev_hist is not None:
                # Compare histograms
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                
                # If difference exceeds threshold, mark scene change
                if 1 - diff > self.threshold:
                    scene_end_frame = frame_idx - sample_interval
                    
                    if scene_end_frame > scene_start_frame:
                        scene = self._create_scene(
                            scene_start_frame,
                            scene_end_frame,
                            fps,
                            scene_index,
                            frame
                        )
                        scenes.append(scene)
                        scene_index += 1
                        
                        if len(scenes) >= max_scenes:
                            break
                    
                    scene_start_frame = frame_idx
            
            prev_hist = hist.copy()
        
        cap.release()
        
        # Add final scene
        if scene_index == 0 or scenes[-1].start_time != self._frame_to_timestamp(scene_start_frame, fps):
            final_scene = self._create_scene(
                scene_start_frame,
                total_frames - 1,
                fps,
                scene_index,
                None
            )
            scenes.append(final_scene)
        
        return scenes
    
    def _detect_with_ffmpeg(
        self,
        video_path: Union[str, Path],
        max_scenes: int
    ) -> List[Scene]:
        """Fallback scene detection using FFmpeg."""
        # Use FFmpeg's scene detection filter
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-show_entries', 'frame=pts_time,pict_type',
            '-select_streams', 'v',
            '-of', 'csv=p=0',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            scenes = []
            scene_times = []
            
            # Parse I-frames as potential scene boundaries
            for line in result.stdout.strip().split('\n')[:max_scenes * 10]:
                parts = line.strip().split(',')
                if len(parts) >= 2 and parts[1] == 'I':
                    try:
                        timestamp = float(parts[0])
                        scene_times.append(timestamp)
                    except ValueError:
                        continue
            
            # Filter scene times based on minimum duration
            min_scene_duration = 2.0  # seconds
            filtered_times = [0.0]
            for t in scene_times:
                if t - filtered_times[-1] >= min_scene_duration:
                    filtered_times.append(t)
            
            # Create scenes from filtered times
            for i, start_time in enumerate(filtered_times[:max_scenes]):
                end_time = filtered_times[i + 1] if i + 1 < len(filtered_times) else start_time + 10
                
                scenes.append(Scene(
                    start_time=self._seconds_to_timestamp(start_time),
                    end_time=self._seconds_to_timestamp(end_time),
                    duration_seconds=end_time - start_time,
                    scene_index=i
                ))
            
            return scenes
            
        except subprocess.SubprocessError as e:
            logger.warning(f"FFmpeg scene detection failed: {e}")
            return []
    
    def _create_scene(
        self,
        start_frame: int,
        end_frame: int,
        fps: float,
        scene_index: int,
        frame = None
    ) -> Scene:
        """Create a Scene object from frame information."""
        start_time = self._frame_to_timestamp(start_frame, fps)
        end_time = self._frame_to_timestamp(end_frame, fps)
        duration = (end_frame - start_frame) / fps if fps > 0 else 0
        
        dominant_colors = []
        if frame is not None:
            dominant_colors = self._extract_dominant_colors(frame)
        
        return Scene(
            start_time=start_time,
            end_time=end_time,
            duration_seconds=round(duration, 2),
            scene_index=scene_index,
            dominant_colors=dominant_colors
        )
    
    def _extract_dominant_colors(self, frame, n_colors: int = 3) -> List[str]:
        """Extract dominant colors from a frame."""
        try:
            import cv2
            import numpy as np
            
            # Resize for faster processing
            small = cv2.resize(frame, (100, 100))
            pixels = small.reshape(-1, 3)
            
            # Simple k-means clustering
            from collections import Counter
            
            # Quantize colors
            quantized = (pixels // 32) * 32
            color_counts = Counter(map(tuple, quantized))
            
            # Get top colors
            top_colors = color_counts.most_common(n_colors)
            hex_colors = []
            
            for (b, g, r), _ in top_colors:
                hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
            
            return hex_colors
            
        except Exception as e:
            logger.debug(f"Color extraction failed: {e}")
            return []
    
    @staticmethod
    def _frame_to_timestamp(frame: int, fps: float) -> str:
        """Convert frame number to timestamp."""
        if fps <= 0:
            return "00:00:00"
        seconds = frame / fps
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    @staticmethod
    def _seconds_to_timestamp(seconds: float) -> str:
        """Convert seconds to timestamp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ============================================================================
# DELTA CHANGE DETECTOR (Motion/Movement Detection)
# ============================================================================

class DeltaChangeDetector:
    """
    Detect video changes using simple frame differencing (delta).
    
    This is a lightweight approach that compares consecutive frames
    to detect movement, scene changes, and key moments without
    complex computer vision libraries.
    
    The delta method works by:
    1. Converting frames to grayscale
    2. Computing absolute difference between consecutive frames
    3. Calculating the mean change (delta) across all pixels
    4. Marking frames with high delta as "key moments"
    """
    
    def __init__(self, threshold: float = 15.0, min_change_percent: float = 5.0):
        """
        Initialize delta change detector.
        
        Args:
            threshold: Mean pixel difference to consider a "change" (0-255 scale)
            min_change_percent: Minimum % of pixels that must change
        """
        self.threshold = threshold
        self.min_change_percent = min_change_percent
        self._pil_available = self._check_pil()
    
    def _check_pil(self) -> bool:
        """Check if PIL/Pillow is available."""
        try:
            from PIL import Image
            return True
        except ImportError:
            return False
    
    def detect_changes(
        self,
        frames: List[Tuple[str, str]],
        return_deltas: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect changes between consecutive frames using delta comparison.
        
        Args:
            frames: List of (frame_path, timestamp) tuples
            return_deltas: Include raw delta values in output
            
        Returns:
            List of change events with timestamps and magnitude
        """
        if not self._pil_available:
            logger.warning("PIL not available - using FFmpeg delta detection")
            return self._detect_with_ffmpeg_diff(frames)
        
        return self._detect_with_pil(frames, return_deltas)
    
    def _detect_with_pil(
        self,
        frames: List[Tuple[str, str]],
        return_deltas: bool
    ) -> List[Dict[str, Any]]:
        """Detect changes using PIL image differencing."""
        from PIL import Image, ImageChops, ImageStat
        
        changes = []
        prev_frame = None
        all_deltas = []
        
        for i, (frame_path, timestamp) in enumerate(frames):
            try:
                # Load and convert to grayscale
                current = Image.open(frame_path).convert('L')
                
                if prev_frame is not None:
                    # Calculate absolute difference
                    diff = ImageChops.difference(prev_frame, current)
                    
                    # Get statistics on the difference
                    stat = ImageStat.Stat(diff)
                    mean_delta = stat.mean[0]  # Average pixel difference (0-255)
                    
                    # Calculate percentage of pixels that changed significantly
                    diff_array = list(diff.getdata())
                    pixels_changed = sum(1 for p in diff_array if p > self.threshold)
                    change_percent = (pixels_changed / len(diff_array)) * 100
                    
                    all_deltas.append({
                        'frame_index': i,
                        'timestamp': timestamp,
                        'mean_delta': round(mean_delta, 2),
                        'change_percent': round(change_percent, 2),
                        'is_significant': mean_delta > self.threshold or change_percent > self.min_change_percent
                    })
                    
                    # Mark significant changes
                    if mean_delta > self.threshold or change_percent > self.min_change_percent:
                        changes.append({
                            'timestamp': timestamp,
                            'frame_index': i,
                            'change_type': self._classify_change(mean_delta, change_percent),
                            'magnitude': round(mean_delta, 2),
                            'change_percent': round(change_percent, 2),
                            'description': self._describe_change(mean_delta, change_percent)
                        })
                
                prev_frame = current
                
            except Exception as e:
                logger.debug(f"Failed to process frame {frame_path}: {e}")
                continue
        
        if return_deltas:
            return {'changes': changes, 'all_deltas': all_deltas}
        
        return changes
    
    def _detect_with_ffmpeg_diff(
        self,
        frames: List[Tuple[str, str]]
    ) -> List[Dict[str, Any]]:
        """Fallback delta detection using basic file comparison."""
        changes = []
        prev_size = None
        
        for i, (frame_path, timestamp) in enumerate(frames):
            try:
                current_size = Path(frame_path).stat().st_size
                
                if prev_size is not None:
                    # Simple heuristic: significant file size change = visual change
                    size_diff = abs(current_size - prev_size) / max(prev_size, 1)
                    
                    if size_diff > 0.1:  # 10% size change
                        changes.append({
                            'timestamp': timestamp,
                            'frame_index': i,
                            'change_type': 'unknown',
                            'magnitude': round(size_diff * 100, 2),
                            'description': f"Frame size changed by {size_diff*100:.1f}%"
                        })
                
                prev_size = current_size
                
            except Exception:
                continue
        
        return changes
    
    def _classify_change(self, mean_delta: float, change_percent: float) -> str:
        """Classify the type of change based on delta values."""
        if mean_delta > 50 or change_percent > 50:
            return 'scene_change'
        elif mean_delta > 30 or change_percent > 30:
            return 'major_movement'
        elif mean_delta > 15 or change_percent > 15:
            return 'moderate_movement'
        else:
            return 'minor_movement'
    
    def _describe_change(self, mean_delta: float, change_percent: float) -> str:
        """Generate human-readable description of the change."""
        change_type = self._classify_change(mean_delta, change_percent)
        
        descriptions = {
            'scene_change': f"Major scene change detected (delta: {mean_delta:.1f}, {change_percent:.1f}% pixels changed)",
            'major_movement': f"Significant movement/activity (delta: {mean_delta:.1f}, {change_percent:.1f}% pixels changed)",
            'moderate_movement': f"Moderate visual change (delta: {mean_delta:.1f}, {change_percent:.1f}% pixels changed)",
            'minor_movement': f"Minor movement detected (delta: {mean_delta:.1f}, {change_percent:.1f}% pixels changed)"
        }
        
        return descriptions.get(change_type, f"Change detected (delta: {mean_delta:.1f})")
    
    def find_key_moments(
        self,
        frames: List[Tuple[str, str]],
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find the top N key moments based on visual change magnitude.
        
        Args:
            frames: List of (frame_path, timestamp) tuples
            top_n: Number of key moments to return
            
        Returns:
            List of key moment events sorted by magnitude
        """
        result = self.detect_changes(frames, return_deltas=True)
        
        if isinstance(result, dict):
            all_deltas = result.get('all_deltas', [])
        else:
            # Fallback if detect_changes returned simple list
            return result[:top_n]
        
        # Sort by mean_delta descending
        sorted_deltas = sorted(
            [d for d in all_deltas if d.get('is_significant', False)],
            key=lambda x: x.get('mean_delta', 0),
            reverse=True
        )
        
        key_moments = []
        for delta in sorted_deltas[:top_n]:
            key_moments.append({
                'timestamp': delta['timestamp'],
                'frame_index': delta['frame_index'],
                'change_type': self._classify_change(delta['mean_delta'], delta['change_percent']),
                'magnitude': delta['mean_delta'],
                'change_percent': delta['change_percent'],
                'is_key_moment': True,
                'description': self._describe_change(delta['mean_delta'], delta['change_percent'])
            })
        
        return key_moments
    
    def get_activity_timeline(
        self,
        frames: List[Tuple[str, str]],
        bucket_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate an activity timeline showing movement over time.
        
        Args:
            frames: List of (frame_path, timestamp) tuples
            bucket_size: Number of frames per bucket for averaging
            
        Returns:
            Timeline with activity levels per time bucket
        """
        result = self.detect_changes(frames, return_deltas=True)
        
        if isinstance(result, dict):
            all_deltas = result.get('all_deltas', [])
        else:
            return []
        
        if not all_deltas:
            return []
        
        timeline = []
        
        for i in range(0, len(all_deltas), bucket_size):
            bucket = all_deltas[i:i + bucket_size]
            
            if bucket:
                avg_delta = sum(d['mean_delta'] for d in bucket) / len(bucket)
                max_delta = max(d['mean_delta'] for d in bucket)
                start_ts = bucket[0]['timestamp']
                end_ts = bucket[-1]['timestamp']
                
                # Classify activity level
                if avg_delta > 30:
                    activity = 'high'
                elif avg_delta > 15:
                    activity = 'medium'
                elif avg_delta > 5:
                    activity = 'low'
                else:
                    activity = 'static'
                
                timeline.append({
                    'start_time': start_ts,
                    'end_time': end_ts,
                    'avg_delta': round(avg_delta, 2),
                    'max_delta': round(max_delta, 2),
                    'activity_level': activity,
                    'frame_count': len(bucket)
                })
        
        return timeline


# ============================================================================
# OCR ENGINE
# ============================================================================

class OCREngine:
    """Extract text from video frames using Tesseract OCR."""
    
    def __init__(self, languages: List[str] = None, confidence_threshold: float = 0.6):
        """
        Initialize OCR engine.
        
        Args:
            languages: OCR languages (default: ['eng'])
            confidence_threshold: Minimum confidence for text detection
        """
        self.languages = languages or ['eng']
        self.confidence_threshold = confidence_threshold
        self._tesseract_available = DependencyChecker._check_command(['tesseract', '--version'])
    
    def extract_text(
        self,
        frame_path: Union[str, Path],
        timestamp: str,
        frame_index: int
    ) -> List[TextOccurrence]:
        """
        Extract text from a single frame.
        
        Args:
            frame_path: Path to frame image
            timestamp: Frame timestamp
            frame_index: Frame index number
            
        Returns:
            List of TextOccurrence objects
        """
        if not self._tesseract_available:
            logger.warning("Tesseract not available - OCR skipped")
            return []
        
        frame_path = Path(frame_path)
        if not frame_path.exists():
            return []
        
        try:
            # Try using pytesseract if available
            try:
                import pytesseract
                from PIL import Image
                
                img = Image.open(frame_path)
                
                # Get detailed OCR data
                data = pytesseract.image_to_data(
                    img,
                    lang='+'.join(self.languages),
                    output_type=pytesseract.Output.DICT
                )
                
                occurrences = []
                n_boxes = len(data['text'])
                
                for i in range(n_boxes):
                    text = data['text'][i].strip()
                    conf = float(data['conf'][i]) / 100.0
                    
                    if text and conf >= self.confidence_threshold:
                        bbox = (
                            data['left'][i],
                            data['top'][i],
                            data['width'][i],
                            data['height'][i]
                        )
                        
                        occurrences.append(TextOccurrence(
                            text=text,
                            timestamp=timestamp,
                            confidence=round(conf, 2),
                            frame_index=frame_index,
                            bounding_box=bbox
                        ))
                
                return occurrences
                
            except ImportError:
                # Fallback to command-line tesseract
                return self._extract_with_cli(frame_path, timestamp, frame_index)
                
        except Exception as e:
            logger.warning(f"OCR failed for {frame_path}: {e}")
            return []
    
    def _extract_with_cli(
        self,
        frame_path: Path,
        timestamp: str,
        frame_index: int
    ) -> List[TextOccurrence]:
        """Extract text using command-line Tesseract."""
        try:
            cmd = [
                'tesseract',
                str(frame_path),
                'stdout',
                '-l', '+'.join(self.languages)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            text = result.stdout.strip()
            if text:
                # Split into lines and filter
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                return [
                    TextOccurrence(
                        text=line,
                        timestamp=timestamp,
                        confidence=0.7,  # Unknown confidence from CLI
                        frame_index=frame_index
                    )
                    for line in lines
                ]
            
            return []
            
        except subprocess.SubprocessError:
            return []
    
    def extract_from_frames(
        self,
        frames: List[Tuple[str, str]]
    ) -> List[TextOccurrence]:
        """
        Extract text from multiple frames.
        
        Args:
            frames: List of (frame_path, timestamp) tuples
            
        Returns:
            List of all TextOccurrence objects
        """
        all_text = []
        
        for i, (frame_path, timestamp) in enumerate(frames):
            occurrences = self.extract_text(frame_path, timestamp, i)
            all_text.extend(occurrences)
        
        return all_text


# ============================================================================
# MAIN VIDEO ANALYZER
# ============================================================================

class VideoAnalyzer:
    """
    Main video analysis orchestrator.
    
    Coordinates metadata extraction, frame sampling, scene detection,
    and OCR to produce comprehensive video analysis.
    """
    
    def __init__(
        self,
        sample_interval: int = 30,
        max_frames: int = 100,
        scene_threshold: float = 0.3,
        ocr_confidence: float = 0.6,
        ocr_languages: List[str] = None,
        delta_threshold: float = 15.0,
        cleanup_temp: bool = True
    ):
        """
        Initialize video analyzer.
        
        Args:
            sample_interval: Seconds between frame samples
            max_frames: Maximum frames to extract
            scene_threshold: Scene change detection threshold
            ocr_confidence: Minimum OCR confidence threshold
            ocr_languages: OCR languages (default: ['eng'])
            delta_threshold: Delta change detection threshold (0-255)
            cleanup_temp: Whether to cleanup temp files after analysis
        """
        self.sample_interval = sample_interval
        self.max_frames = max_frames
        self.scene_threshold = scene_threshold
        self.ocr_confidence = ocr_confidence
        self.ocr_languages = ocr_languages or ['eng']
        self.delta_threshold = delta_threshold
        self.cleanup_temp = cleanup_temp
        
        self._temp_dir = None
        self._frame_sampler = None
        self._scene_detector = SceneDetector(threshold=scene_threshold)
        self._ocr_engine = OCREngine(
            languages=self.ocr_languages,
            confidence_threshold=ocr_confidence
        )
        self._delta_detector = DeltaChangeDetector(threshold=delta_threshold)
    
    def analyze(
        self,
        video_path: Union[str, Path],
        analysis_type: str = "comprehensive",
        output_path: Optional[Union[str, Path]] = None
    ) -> AnalysisResult:
        """
        Perform video analysis.
        
        Args:
            video_path: Path to video file
            analysis_type: Type of analysis ('comprehensive', 'quick', 'frames_only', 'ocr_only')
            output_path: Optional path to save JSON results
            
        Returns:
            AnalysisResult with all findings
        """
        start_time = time.time()
        video_path = Path(video_path)
        
        # Validate input
        if not video_path.exists():
            raise VideoNotFoundError(f"Video not found: {video_path}")
        
        # Check dependencies
        DependencyChecker.verify_required()
        
        # Initialize temp directory
        self._temp_dir = Path(tempfile.mkdtemp(prefix='videoanalysis_'))
        self._frame_sampler = FrameSampler(self._temp_dir)
        
        try:
            # Extract metadata
            logger.info(f"Extracting metadata from {video_path.name}...")
            metadata = MetadataExtractor.extract(video_path)
            
            # Initialize result
            result = AnalysisResult(
                file_path=str(video_path.absolute()),
                file_name=video_path.name,
                duration=metadata.duration_formatted,
                resolution=(metadata.width, metadata.height),
                fps=metadata.fps,
                format_name=metadata.format_name,
                codec=metadata.codec,
                file_size_mb=round(metadata.file_size / (1024 * 1024), 2),
                analysis_timestamp=datetime.now().isoformat()
            )
            
            # Store raw metadata
            result.metadata = asdict(metadata)
            
            # Determine analysis components based on type
            do_frames = analysis_type in ('comprehensive', 'quick', 'frames_only', 'ocr_only', 'delta_only')
            do_scenes = analysis_type in ('comprehensive', 'quick')
            do_ocr = analysis_type in ('comprehensive', 'ocr_only')
            do_delta = analysis_type in ('comprehensive', 'quick', 'delta_only')
            
            # Adjust for quick mode
            sample_interval = self.sample_interval
            max_frames = self.max_frames
            if analysis_type == 'quick':
                sample_interval = max(60, self.sample_interval * 2)
                max_frames = min(20, self.max_frames)
            
            # Extract frames
            if do_frames:
                logger.info(f"Extracting frames (interval: {sample_interval}s)...")
                frames = self._frame_sampler.sample(
                    video_path,
                    interval_seconds=sample_interval,
                    max_frames=max_frames
                )
                
                # Create frame analysis objects
                for i, (frame_path, timestamp) in enumerate(frames):
                    result.frames.append(FrameAnalysis(
                        timestamp=timestamp,
                        frame_index=i,
                        frame_path=frame_path
                    ))
                
                # OCR on frames
                if do_ocr and frames:
                    logger.info(f"Running OCR on {len(frames)} frames...")
                    result.text_detected = self._ocr_engine.extract_from_frames(frames)
                    
                    # Add text to frame analyses
                    text_by_frame = {}
                    for text_occ in result.text_detected:
                        if text_occ.frame_index not in text_by_frame:
                            text_by_frame[text_occ.frame_index] = []
                        text_by_frame[text_occ.frame_index].append(text_occ.text)
                    
                    for frame in result.frames:
                        frame.text_detected = text_by_frame.get(frame.frame_index, [])
                
                # Delta change detection (movement/activity)
                if do_delta and frames:
                    logger.info("Detecting movement using delta change analysis...")
                    
                    # Get all delta changes
                    delta_result = self._delta_detector.detect_changes(frames, return_deltas=True)
                    if isinstance(delta_result, dict):
                        changes = delta_result.get('changes', [])
                    else:
                        changes = delta_result
                    
                    # Convert to DeltaChange dataclass objects
                    for change in changes:
                        result.delta_changes.append(DeltaChange(
                            timestamp=change.get('timestamp', ''),
                            frame_index=change.get('frame_index', 0),
                            change_type=change.get('change_type', 'unknown'),
                            magnitude=change.get('magnitude', 0.0),
                            change_percent=change.get('change_percent', 0.0),
                            description=change.get('description', '')
                        ))
                    
                    # Find key moments (biggest visual changes)
                    key_moments = self._delta_detector.find_key_moments(frames, top_n=10)
                    for moment in key_moments:
                        result.key_moments.append(DeltaChange(
                            timestamp=moment.get('timestamp', ''),
                            frame_index=moment.get('frame_index', 0),
                            change_type=moment.get('change_type', 'unknown'),
                            magnitude=moment.get('magnitude', 0.0),
                            change_percent=moment.get('change_percent', 0.0),
                            is_key_moment=True,
                            description=moment.get('description', '')
                        ))
                    
                    # Get activity timeline
                    timeline = self._delta_detector.get_activity_timeline(frames, bucket_size=5)
                    for bucket in timeline:
                        result.activity_timeline.append(ActivityBucket(
                            start_time=bucket.get('start_time', ''),
                            end_time=bucket.get('end_time', ''),
                            avg_delta=bucket.get('avg_delta', 0.0),
                            max_delta=bucket.get('max_delta', 0.0),
                            activity_level=bucket.get('activity_level', 'unknown'),
                            frame_count=bucket.get('frame_count', 0)
                        ))
            
            # Detect scenes
            if do_scenes:
                logger.info("Detecting scenes...")
                result.scenes = self._scene_detector.detect(video_path)
            
            # Generate summary
            result.summary = self._generate_summary(result)
            
            # Calculate processing time
            result.processing_time_seconds = round(time.time() - start_time, 2)
            
            logger.info(f"Analysis complete in {result.processing_time_seconds}s")
            
            # Save results if output path specified
            if output_path:
                self._save_results(result, Path(output_path))
            
            return result
            
        finally:
            # Cleanup temp files
            if self.cleanup_temp and self._temp_dir and self._temp_dir.exists():
                shutil.rmtree(self._temp_dir, ignore_errors=True)
    
    def _generate_summary(self, result: AnalysisResult) -> str:
        """Generate a human-readable summary of the analysis."""
        parts = []
        
        # Basic info
        parts.append(
            f"{result.duration} video ({result.resolution[0]}x{result.resolution[1]}, "
            f"{result.fps}fps, {result.codec})"
        )
        
        # Scene info
        if result.scenes:
            parts.append(f"{len(result.scenes)} scenes detected")
        
        # Frame info
        if result.frames:
            parts.append(f"{len(result.frames)} frames analyzed")
        
        # Delta/movement info
        if result.delta_changes:
            parts.append(f"{len(result.delta_changes)} movement events detected")
        
        if result.key_moments:
            parts.append(f"{len(result.key_moments)} key moments identified")
        
        # Activity summary
        if result.activity_timeline:
            high_activity = sum(1 for a in result.activity_timeline if a.activity_level == 'high')
            if high_activity > 0:
                parts.append(f"{high_activity} high-activity segments")
        
        # Text info
        if result.text_detected:
            unique_texts = set(t.text for t in result.text_detected)
            parts.append(f"{len(unique_texts)} unique text occurrences found")
        
        return ". ".join(parts) + "."
    
    def _save_results(self, result: AnalysisResult, output_path: Path) -> None:
        """Save analysis results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass to dict
        data = asdict(result)
        
        # Handle non-serializable types
        def clean_for_json(obj):
            if isinstance(obj, tuple):
                return list(obj)
            return obj
        
        # Serialize
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=clean_for_json)
        
        logger.info(f"Results saved to {output_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog='videoanalysis',
        description='Enable AI agents to watch and analyze video content',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  videoanalysis analyze video.mp4
  videoanalysis analyze video.mp4 --output results.json
  videoanalysis analyze video.mp4 --quick
  videoanalysis analyze video.mp4 --sample-rate 60 --max-frames 50
  videoanalysis check-deps

Built by ATLAS for Team Brain
Protocol: BUILD_PROTOCOL_V1.md
        """
    )
    
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a video file')
    analyze_parser.add_argument('video', help='Path to video file')
    analyze_parser.add_argument('-o', '--output', help='Output JSON file path')
    analyze_parser.add_argument(
        '-t', '--type',
        choices=['comprehensive', 'quick', 'frames_only', 'ocr_only', 'delta_only'],
        default='comprehensive',
        help='Analysis type (default: comprehensive)'
    )
    analyze_parser.add_argument(
        '--delta-threshold',
        type=float,
        default=15.0,
        help='Delta change threshold 0-255 (default: 15.0)'
    )
    analyze_parser.add_argument(
        '-s', '--sample-rate',
        type=int,
        default=30,
        help='Seconds between frame samples (default: 30)'
    )
    analyze_parser.add_argument(
        '-m', '--max-frames',
        type=int,
        default=100,
        help='Maximum frames to extract (default: 100)'
    )
    analyze_parser.add_argument(
        '--scene-threshold',
        type=float,
        default=0.3,
        help='Scene detection threshold 0.0-1.0 (default: 0.3)'
    )
    analyze_parser.add_argument(
        '--ocr-confidence',
        type=float,
        default=0.6,
        help='Minimum OCR confidence 0.0-1.0 (default: 0.6)'
    )
    analyze_parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Keep temporary files after analysis'
    )
    analyze_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Check dependencies command
    subparsers.add_parser('check-deps', help='Check required dependencies')
    
    # Frames command
    frames_parser = subparsers.add_parser('frames', help='Extract frames only')
    frames_parser.add_argument('video', help='Path to video file')
    frames_parser.add_argument('-o', '--output-dir', required=True, help='Output directory')
    frames_parser.add_argument('-i', '--interval', type=int, default=30, help='Interval in seconds')
    frames_parser.add_argument('-m', '--max-frames', type=int, default=100, help='Max frames')
    
    # Scenes command
    scenes_parser = subparsers.add_parser('scenes', help='Detect scenes only')
    scenes_parser.add_argument('video', help='Path to video file')
    scenes_parser.add_argument('-o', '--output', help='Output JSON file')
    scenes_parser.add_argument('-t', '--threshold', type=float, default=0.3, help='Detection threshold')
    
    # OCR command
    ocr_parser = subparsers.add_parser('ocr', help='OCR on specific timestamps')
    ocr_parser.add_argument('video', help='Path to video file')
    ocr_parser.add_argument('--timestamps', help='Comma-separated timestamps (e.g., 00:10:00,00:20:00)')
    ocr_parser.add_argument('-o', '--output', help='Output JSON file')
    
    # Delta/movement command
    delta_parser = subparsers.add_parser('delta', help='Detect movement using delta change analysis')
    delta_parser.add_argument('video', help='Path to video file')
    delta_parser.add_argument('-o', '--output', help='Output JSON file')
    delta_parser.add_argument('-i', '--interval', type=int, default=5, help='Sampling interval in seconds (default: 5)')
    delta_parser.add_argument('-t', '--threshold', type=float, default=15.0, help='Change threshold 0-255 (default: 15)')
    delta_parser.add_argument('--key-moments', type=int, default=10, help='Number of key moments to find (default: 10)')
    delta_parser.add_argument('--timeline', action='store_true', help='Include activity timeline')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set logging level
    if getattr(args, 'verbose', False):
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.command == 'check-deps':
            print(DependencyChecker.get_status_report())
            return 0
        
        elif args.command == 'analyze':
            analyzer = VideoAnalyzer(
                sample_interval=args.sample_rate,
                max_frames=args.max_frames,
                scene_threshold=args.scene_threshold,
                ocr_confidence=args.ocr_confidence,
                delta_threshold=args.delta_threshold,
                cleanup_temp=not args.no_cleanup
            )
            
            result = analyzer.analyze(
                args.video,
                analysis_type=args.type,
                output_path=args.output
            )
            
            # Print summary
            print("\n" + "=" * 60)
            print("VIDEO ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"File: {result.file_name}")
            print(f"Duration: {result.duration}")
            print(f"Resolution: {result.resolution[0]}x{result.resolution[1]}")
            print(f"FPS: {result.fps}")
            print(f"Format: {result.format_name} ({result.codec})")
            print(f"Size: {result.file_size_mb} MB")
            print(f"\nScenes detected: {len(result.scenes)}")
            print(f"Frames analyzed: {len(result.frames)}")
            print(f"Text occurrences: {len(result.text_detected)}")
            print(f"\nProcessing time: {result.processing_time_seconds}s")
            
            if args.output:
                print(f"\nResults saved to: {args.output}")
            
            # Show some detected text
            if result.text_detected:
                print("\nSample detected text:")
                for text_occ in result.text_detected[:5]:
                    print(f"  [{text_occ.timestamp}] {text_occ.text}")
                if len(result.text_detected) > 5:
                    print(f"  ... and {len(result.text_detected) - 5} more")
            
            # Show key moments (delta-detected)
            if result.key_moments:
                print("\nKey moments (visual changes):")
                for moment in result.key_moments[:5]:
                    print(f"  [{moment.timestamp}] {moment.change_type}: {moment.description}")
                if len(result.key_moments) > 5:
                    print(f"  ... and {len(result.key_moments) - 5} more")
            
            # Show activity summary
            if result.activity_timeline:
                high = sum(1 for a in result.activity_timeline if a.activity_level == 'high')
                medium = sum(1 for a in result.activity_timeline if a.activity_level == 'medium')
                low = sum(1 for a in result.activity_timeline if a.activity_level == 'low')
                static = sum(1 for a in result.activity_timeline if a.activity_level == 'static')
                print(f"\nActivity breakdown: {high} high, {medium} medium, {low} low, {static} static")
            
            return 0
        
        elif args.command == 'frames':
            sampler = FrameSampler(Path(args.output_dir))
            frames = sampler.sample(
                args.video,
                interval_seconds=args.interval,
                max_frames=args.max_frames
            )
            print(f"Extracted {len(frames)} frames to {args.output_dir}")
            return 0
        
        elif args.command == 'scenes':
            detector = SceneDetector(threshold=args.threshold)
            scenes = detector.detect(args.video)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump([asdict(s) for s in scenes], f, indent=2)
                print(f"Saved {len(scenes)} scenes to {args.output}")
            else:
                print(f"Detected {len(scenes)} scenes:")
                for scene in scenes:
                    print(f"  Scene {scene.scene_index}: {scene.start_time} - {scene.end_time}")
            
            return 0
        
        elif args.command == 'ocr':
            if not args.timestamps:
                print("Error: --timestamps required for OCR command")
                return 1
            
            timestamps = [ts.strip() for ts in args.timestamps.split(',')]
            
            # Extract frames at timestamps
            temp_dir = Path(tempfile.mkdtemp())
            sampler = FrameSampler(temp_dir)
            frames = sampler.sample(args.video, timestamps=timestamps)
            
            # Run OCR
            ocr = OCREngine()
            text_results = ocr.extract_from_frames(frames)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump([asdict(t) for t in text_results], f, indent=2)
                print(f"Saved OCR results to {args.output}")
            else:
                print(f"Found {len(text_results)} text occurrences:")
                for text in text_results:
                    print(f"  [{text.timestamp}] {text.text}")
            
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            return 0
        
        elif args.command == 'delta':
            # Delta/movement detection
            temp_dir = Path(tempfile.mkdtemp())
            sampler = FrameSampler(temp_dir)
            
            print(f"Extracting frames (interval: {args.interval}s)...")
            frames = sampler.sample(
                args.video,
                interval_seconds=args.interval,
                max_frames=200  # More frames for better delta detection
            )
            
            print(f"Analyzing {len(frames)} frames for movement...")
            detector = DeltaChangeDetector(threshold=args.threshold)
            
            # Get all changes
            changes = detector.detect_changes(frames)
            
            # Get key moments
            key_moments = detector.find_key_moments(frames, top_n=args.key_moments)
            
            # Get timeline if requested
            timeline = []
            if args.timeline:
                timeline = detector.get_activity_timeline(frames, bucket_size=5)
            
            # Output results
            results = {
                'video': args.video,
                'frames_analyzed': len(frames),
                'threshold': args.threshold,
                'changes_detected': len(changes),
                'changes': changes,
                'key_moments': key_moments
            }
            
            if timeline:
                results['activity_timeline'] = timeline
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved delta analysis to {args.output}")
            else:
                print(f"\nDelta Change Analysis for: {args.video}")
                print("=" * 50)
                print(f"Frames analyzed: {len(frames)}")
                print(f"Movement events detected: {len(changes)}")
                print(f"\nTop {args.key_moments} Key Moments:")
                for i, moment in enumerate(key_moments, 1):
                    print(f"  {i}. [{moment['timestamp']}] {moment['change_type']}")
                    print(f"      Delta: {moment['magnitude']:.1f}, "
                          f"Changed: {moment.get('change_percent', 0):.1f}%")
                
                if timeline:
                    print(f"\nActivity Timeline:")
                    for bucket in timeline:
                        bar = "#" * int(bucket['avg_delta'] / 5)
                        print(f"  {bucket['start_time']}-{bucket['end_time']}: "
                              f"{bucket['activity_level']:8s} [{bar:<20}] "
                              f"avg:{bucket['avg_delta']:.1f}")
            
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            return 0
        
        else:
            parser.print_help()
            return 1
            
    except VideoAnalysisError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nAborted.")
        return 130


if __name__ == '__main__':
    sys.exit(main())
