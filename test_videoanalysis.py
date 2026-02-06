#!/usr/bin/env python3
"""
Test Suite for VideoAnalysis
Built using BUILD_PROTOCOL_V1.md + Bug Hunt Protocol

Tools Tested:
- DependencyChecker: Verifying external tools
- MetadataExtractor: FFprobe integration
- FrameSampler: Frame extraction
- SceneDetector: Scene change detection
- DeltaChangeDetector: Movement/delta detection
- OCREngine: Text extraction
- VideoAnalyzer: Main orchestrator

Built by ATLAS for Team Brain
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from videoanalysis import (
    # Data classes
    VideoMetadata,
    TextOccurrence,
    Scene,
    FrameAnalysis,
    DeltaChange,
    ActivityBucket,
    AnalysisResult,
    
    # Exceptions
    VideoAnalysisError,
    DependencyError,
    VideoNotFoundError,
    ProcessingError,
    UnsupportedFormatError,
    
    # Components
    DependencyChecker,
    MetadataExtractor,
    FrameSampler,
    SceneDetector,
    DeltaChangeDetector,
    OCREngine,
    VideoAnalyzer,
    
    # CLI
    create_parser,
    main,
    
    # Version
    __version__,
)


class TestVideoMetadataDataclass(unittest.TestCase):
    """Test VideoMetadata dataclass."""
    
    def test_default_values(self):
        """Test default values are set correctly."""
        meta = VideoMetadata()
        self.assertEqual(meta.duration_seconds, 0.0)
        self.assertEqual(meta.duration_formatted, "00:00:00")
        self.assertEqual(meta.width, 0)
        self.assertEqual(meta.height, 0)
        self.assertEqual(meta.fps, 0.0)
    
    def test_custom_values(self):
        """Test custom values are stored correctly."""
        meta = VideoMetadata(
            duration_seconds=3661.5,
            duration_formatted="01:01:01",
            width=1920,
            height=1080,
            fps=29.97,
            codec="h264",
            bitrate=5000000,
            file_size=100000000,
            format_name="mp4"
        )
        self.assertEqual(meta.duration_seconds, 3661.5)
        self.assertEqual(meta.width, 1920)
        self.assertEqual(meta.codec, "h264")


class TestDeltaChangeDataclass(unittest.TestCase):
    """Test DeltaChange dataclass."""
    
    def test_create_delta_change(self):
        """Test creating a delta change object."""
        change = DeltaChange(
            timestamp="00:01:30",
            frame_index=3,
            change_type="major_movement",
            magnitude=45.5,
            change_percent=35.2,
            is_key_moment=True,
            description="Significant movement detected"
        )
        self.assertEqual(change.timestamp, "00:01:30")
        self.assertEqual(change.change_type, "major_movement")
        self.assertTrue(change.is_key_moment)


class TestAnalysisResultDataclass(unittest.TestCase):
    """Test AnalysisResult dataclass."""
    
    def test_create_result(self):
        """Test creating analysis result."""
        result = AnalysisResult(
            file_path="/path/to/video.mp4",
            file_name="video.mp4",
            duration="00:45:00"
        )
        self.assertEqual(result.file_name, "video.mp4")
        self.assertEqual(result.duration, "00:45:00")
        self.assertEqual(result.tool_version, __version__)
    
    def test_default_lists(self):
        """Test default empty lists."""
        result = AnalysisResult(
            file_path="/path/to/video.mp4",
            file_name="video.mp4",
            duration="00:45:00"
        )
        self.assertEqual(result.scenes, [])
        self.assertEqual(result.frames, [])
        self.assertEqual(result.delta_changes, [])
        self.assertEqual(result.key_moments, [])
        self.assertEqual(result.activity_timeline, [])


class TestDependencyChecker(unittest.TestCase):
    """Test DependencyChecker class."""
    
    def test_check_command_success(self):
        """Test checking a command that exists (python)."""
        # Python should always be available in test environment
        result = DependencyChecker._check_command(['python', '--version'])
        self.assertTrue(result)
    
    def test_check_command_failure(self):
        """Test checking a command that doesn't exist."""
        result = DependencyChecker._check_command(['nonexistent_command_xyz'])
        self.assertFalse(result)
    
    def test_get_status_report(self):
        """Test status report generation."""
        report = DependencyChecker.get_status_report()
        self.assertIn("VideoAnalysis Dependency Status", report)
        self.assertIn("Required:", report)
        self.assertIn("Optional:", report)
    
    def test_check_all_returns_dict(self):
        """Test check_all returns dictionary."""
        results = DependencyChecker.check_all()
        self.assertIsInstance(results, dict)
        self.assertIn('ffmpeg', results)
        self.assertIn('ffprobe', results)


class TestFrameSampler(unittest.TestCase):
    """Test FrameSampler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sampler = FrameSampler(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_init_creates_temp_dir(self):
        """Test that init sets up temp directory."""
        self.assertTrue(self.sampler.temp_dir.exists())
    
    def test_seconds_to_timestamp(self):
        """Test timestamp conversion."""
        self.assertEqual(FrameSampler._seconds_to_timestamp(0), "00:00:00")
        self.assertEqual(FrameSampler._seconds_to_timestamp(61), "00:01:01")
        self.assertEqual(FrameSampler._seconds_to_timestamp(3661), "01:01:01")
    
    def test_cleanup(self):
        """Test temp directory cleanup."""
        self.sampler.cleanup()
        self.assertFalse(self.temp_dir.exists())
    
    def test_sample_nonexistent_video(self):
        """Test sampling non-existent video raises error."""
        with self.assertRaises(VideoNotFoundError):
            self.sampler.sample("/nonexistent/video.mp4")


class TestSceneDetector(unittest.TestCase):
    """Test SceneDetector class."""
    
    def test_init_with_threshold(self):
        """Test initialization with threshold."""
        detector = SceneDetector(threshold=0.5)
        self.assertEqual(detector.threshold, 0.5)
    
    def test_frame_to_timestamp(self):
        """Test frame to timestamp conversion."""
        ts = SceneDetector._frame_to_timestamp(0, 30.0)
        self.assertEqual(ts, "00:00:00")
        
        ts = SceneDetector._frame_to_timestamp(1800, 30.0)
        self.assertEqual(ts, "00:01:00")
    
    def test_seconds_to_timestamp(self):
        """Test seconds to timestamp conversion."""
        ts = SceneDetector._seconds_to_timestamp(3661.5)
        self.assertEqual(ts, "01:01:01")


class TestDeltaChangeDetector(unittest.TestCase):
    """Test DeltaChangeDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = DeltaChangeDetector(threshold=15.0, min_change_percent=5.0)
    
    def test_init_with_thresholds(self):
        """Test initialization with thresholds."""
        self.assertEqual(self.detector.threshold, 15.0)
        self.assertEqual(self.detector.min_change_percent, 5.0)
    
    def test_classify_change_scene_change(self):
        """Test classification of scene change."""
        result = self.detector._classify_change(60.0, 55.0)
        self.assertEqual(result, 'scene_change')
    
    def test_classify_change_major_movement(self):
        """Test classification of major movement."""
        result = self.detector._classify_change(35.0, 35.0)
        self.assertEqual(result, 'major_movement')
    
    def test_classify_change_moderate_movement(self):
        """Test classification of moderate movement."""
        result = self.detector._classify_change(20.0, 20.0)
        self.assertEqual(result, 'moderate_movement')
    
    def test_classify_change_minor_movement(self):
        """Test classification of minor movement."""
        result = self.detector._classify_change(10.0, 10.0)
        self.assertEqual(result, 'minor_movement')
    
    def test_describe_change(self):
        """Test change description generation."""
        desc = self.detector._describe_change(60.0, 55.0)
        self.assertIn("scene change", desc.lower())
    
    def test_detect_changes_empty_frames(self):
        """Test detection with empty frame list."""
        changes = self.detector.detect_changes([])
        self.assertEqual(changes, [])
    
    def test_find_key_moments_empty_frames(self):
        """Test finding key moments with empty frames."""
        moments = self.detector.find_key_moments([], top_n=5)
        self.assertEqual(moments, [])


class TestOCREngine(unittest.TestCase):
    """Test OCREngine class."""
    
    def test_init_default_languages(self):
        """Test default language initialization."""
        engine = OCREngine()
        self.assertEqual(engine.languages, ['eng'])
    
    def test_init_custom_languages(self):
        """Test custom language initialization."""
        engine = OCREngine(languages=['eng', 'fra'])
        self.assertEqual(engine.languages, ['eng', 'fra'])
    
    def test_extract_text_nonexistent_frame(self):
        """Test OCR on non-existent frame."""
        engine = OCREngine()
        result = engine.extract_text("/nonexistent/frame.jpg", "00:00:00", 0)
        self.assertEqual(result, [])
    
    def test_extract_from_frames_empty(self):
        """Test OCR on empty frame list."""
        engine = OCREngine()
        result = engine.extract_from_frames([])
        self.assertEqual(result, [])


class TestVideoAnalyzer(unittest.TestCase):
    """Test VideoAnalyzer class."""
    
    def test_init_default_values(self):
        """Test default initialization values."""
        analyzer = VideoAnalyzer()
        self.assertEqual(analyzer.sample_interval, 30)
        self.assertEqual(analyzer.max_frames, 100)
        self.assertEqual(analyzer.scene_threshold, 0.3)
        self.assertEqual(analyzer.ocr_confidence, 0.6)
        self.assertEqual(analyzer.delta_threshold, 15.0)
    
    def test_init_custom_values(self):
        """Test custom initialization values."""
        analyzer = VideoAnalyzer(
            sample_interval=60,
            max_frames=50,
            scene_threshold=0.5,
            ocr_confidence=0.8,
            delta_threshold=20.0
        )
        self.assertEqual(analyzer.sample_interval, 60)
        self.assertEqual(analyzer.max_frames, 50)
        self.assertEqual(analyzer.delta_threshold, 20.0)
    
    def test_analyze_nonexistent_video(self):
        """Test analysis of non-existent video."""
        analyzer = VideoAnalyzer()
        with self.assertRaises(VideoNotFoundError):
            analyzer.analyze("/nonexistent/video.mp4")
    
    def test_generate_summary(self):
        """Test summary generation."""
        analyzer = VideoAnalyzer()
        result = AnalysisResult(
            file_path="/path/video.mp4",
            file_name="video.mp4",
            duration="00:45:00",
            resolution=(1920, 1080),
            fps=30.0,
            codec="h264"
        )
        result.scenes = [Scene(start_time="00:00:00", end_time="00:10:00", duration_seconds=600)]
        result.frames = [FrameAnalysis(timestamp="00:00:00", frame_index=0, frame_path="/tmp/f.jpg")]
        
        summary = analyzer._generate_summary(result)
        self.assertIn("00:45:00", summary)
        self.assertIn("1920x1080", summary)
        self.assertIn("1 scenes", summary)


class TestCLI(unittest.TestCase):
    """Test CLI interface."""
    
    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        self.assertIsNotNone(parser)
    
    def test_parse_check_deps(self):
        """Test parsing check-deps command."""
        parser = create_parser()
        args = parser.parse_args(['check-deps'])
        self.assertEqual(args.command, 'check-deps')
    
    def test_parse_analyze(self):
        """Test parsing analyze command."""
        parser = create_parser()
        args = parser.parse_args(['analyze', 'video.mp4'])
        self.assertEqual(args.command, 'analyze')
        self.assertEqual(args.video, 'video.mp4')
    
    def test_parse_analyze_with_options(self):
        """Test parsing analyze with options."""
        parser = create_parser()
        args = parser.parse_args([
            'analyze', 'video.mp4',
            '-o', 'output.json',
            '-t', 'quick',
            '-s', '60',
            '-m', '50'
        ])
        self.assertEqual(args.output, 'output.json')
        self.assertEqual(args.type, 'quick')
        self.assertEqual(args.sample_rate, 60)
        self.assertEqual(args.max_frames, 50)
    
    def test_parse_delta_command(self):
        """Test parsing delta command."""
        parser = create_parser()
        args = parser.parse_args([
            'delta', 'video.mp4',
            '-i', '5',
            '-t', '20.0',
            '--key-moments', '15',
            '--timeline'
        ])
        self.assertEqual(args.command, 'delta')
        self.assertEqual(args.interval, 5)
        self.assertEqual(args.threshold, 20.0)
        self.assertEqual(args.key_moments, 15)
        self.assertTrue(args.timeline)
    
    def test_parse_frames_command(self):
        """Test parsing frames command."""
        parser = create_parser()
        args = parser.parse_args([
            'frames', 'video.mp4',
            '-o', './frames/',
            '-i', '60'
        ])
        self.assertEqual(args.command, 'frames')
        self.assertEqual(args.output_dir, './frames/')
        self.assertEqual(args.interval, 60)
    
    def test_parse_scenes_command(self):
        """Test parsing scenes command."""
        parser = create_parser()
        args = parser.parse_args([
            'scenes', 'video.mp4',
            '-t', '0.4'
        ])
        self.assertEqual(args.command, 'scenes')
        self.assertEqual(args.threshold, 0.4)


class TestExceptions(unittest.TestCase):
    """Test custom exception classes."""
    
    def test_video_analysis_error(self):
        """Test base exception."""
        with self.assertRaises(VideoAnalysisError):
            raise VideoAnalysisError("Test error")
    
    def test_dependency_error(self):
        """Test dependency error."""
        with self.assertRaises(DependencyError):
            raise DependencyError("Missing FFmpeg")
    
    def test_video_not_found_error(self):
        """Test video not found error."""
        with self.assertRaises(VideoNotFoundError):
            raise VideoNotFoundError("video.mp4 not found")
    
    def test_processing_error(self):
        """Test processing error."""
        with self.assertRaises(ProcessingError):
            raise ProcessingError("Processing failed")
    
    def test_unsupported_format_error(self):
        """Test unsupported format error."""
        with self.assertRaises(UnsupportedFormatError):
            raise UnsupportedFormatError("Format not supported")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_video_path(self):
        """Test handling of empty video path raises appropriate error."""
        analyzer = VideoAnalyzer()
        # Either VideoNotFoundError or DependencyError (if FFmpeg missing)
        with self.assertRaises(VideoAnalysisError):
            analyzer.analyze("")
    
    def test_invalid_sample_interval(self):
        """Test analyzer accepts zero interval (should use default behavior)."""
        analyzer = VideoAnalyzer(sample_interval=0)
        self.assertEqual(analyzer.sample_interval, 0)
    
    def test_delta_detector_extreme_thresholds(self):
        """Test delta detector with extreme thresholds."""
        # Very high threshold - should detect nothing
        detector_high = DeltaChangeDetector(threshold=255.0)
        self.assertEqual(detector_high.threshold, 255.0)
        
        # Very low threshold - should detect everything
        detector_low = DeltaChangeDetector(threshold=0.0)
        self.assertEqual(detector_low.threshold, 0.0)
    
    def test_frame_sampler_zero_interval(self):
        """Test frame sampler with zero interval raises error for non-existent video."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            sampler = FrameSampler(temp_dir)
            # Should raise VideoNotFoundError for non-existent video
            with self.assertRaises(VideoNotFoundError):
                sampler.sample("/fake/video.mp4", interval_seconds=0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestDataclassSerialization(unittest.TestCase):
    """Test dataclass serialization."""
    
    def test_video_metadata_to_dict(self):
        """Test VideoMetadata serialization."""
        meta = VideoMetadata(
            duration_seconds=100.0,
            width=1920,
            height=1080
        )
        data = asdict(meta)
        self.assertIsInstance(data, dict)
        self.assertEqual(data['width'], 1920)
    
    def test_analysis_result_to_dict(self):
        """Test AnalysisResult serialization."""
        result = AnalysisResult(
            file_path="/path/video.mp4",
            file_name="video.mp4",
            duration="00:10:00"
        )
        data = asdict(result)
        self.assertIsInstance(data, dict)
        self.assertEqual(data['file_name'], "video.mp4")
    
    def test_delta_change_to_dict(self):
        """Test DeltaChange serialization."""
        change = DeltaChange(
            timestamp="00:01:00",
            frame_index=2,
            change_type="major_movement",
            magnitude=45.0,
            change_percent=30.0
        )
        data = asdict(change)
        self.assertIsInstance(data, dict)
        self.assertEqual(data['magnitude'], 45.0)


class TestIntegration(unittest.TestCase):
    """Integration tests (require FFmpeg)."""
    
    @classmethod
    def setUpClass(cls):
        """Check if FFmpeg is available for integration tests."""
        cls.ffmpeg_available = DependencyChecker._check_command(['ffmpeg', '-version'])
    
    def test_dependency_check_integration(self):
        """Test full dependency check."""
        results = DependencyChecker.check_all(include_optional=True)
        self.assertIn('ffmpeg', results)
        self.assertIn('tesseract', results)
    
    @unittest.skipUnless(
        DependencyChecker._check_command(['ffmpeg', '-version']),
        "FFmpeg not available"
    )
    def test_metadata_extractor_requires_valid_file(self):
        """Test metadata extractor requires valid file."""
        with self.assertRaises(VideoNotFoundError):
            MetadataExtractor.extract("/definitely/not/a/real/video.mp4")


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)
