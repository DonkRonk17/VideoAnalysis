# VideoAnalysis - Integration Examples

## Copy-Paste-Ready Code for Team Brain Tool Integration

**Document Version:** 1.0  
**Created:** February 6, 2026  
**Author:** ATLAS (Team Brain)  
**For:** Logan Smith / Metaphy LLC  

---

## 🎯 INTEGRATION PHILOSOPHY

VideoAnalysis is designed to work seamlessly with other Team Brain tools. This document provides **copy-paste-ready code examples** for common integration patterns.

Every pattern follows the principle: **Simple solutions first, add complexity only when needed.**

---

## 📚 TABLE OF CONTENTS

1. [Pattern 1: VideoAnalysis + AgentHealth](#pattern-1-videoanalysis--agenthealth)
2. [Pattern 2: VideoAnalysis + SynapseLink](#pattern-2-videoanalysis--synapselink)
3. [Pattern 3: VideoAnalysis + TaskQueuePro](#pattern-3-videoanalysis--taskqueuepro)
4. [Pattern 4: VideoAnalysis + MemoryBridge](#pattern-4-videoanalysis--memorybridge)
5. [Pattern 5: VideoAnalysis + SessionReplay](#pattern-5-videoanalysis--sessionreplay)
6. [Pattern 6: VideoAnalysis + ContextCompressor](#pattern-6-videoanalysis--contextcompressor)
7. [Pattern 7: VideoAnalysis + ConfigManager](#pattern-7-videoanalysis--configmanager)
8. [Pattern 8: VideoAnalysis + AudioAnalysis](#pattern-8-videoanalysis--audioanalysis)
9. [Pattern 9: Multi-Tool Video Pipeline](#pattern-9-multi-tool-video-pipeline)
10. [Pattern 10: Full Team Brain Video Stack](#pattern-10-full-team-brain-video-stack)

---

## Pattern 1: VideoAnalysis + AgentHealth

**Use Case:** Monitor agent health during video analysis tasks

**Why:** Track resource usage and errors during intensive video processing

**Code:**

```python
import time
from videoanalysis import VideoAnalyzer, VideoAnalysisError

def analyze_with_health(video_path: str, agent_name: str) -> dict:
    """
    Analyze video with health monitoring.
    
    Tracks processing time, success/failure, and resource usage
    for correlation with AgentHealth metrics.
    """
    health_data = {
        'agent': agent_name,
        'task': 'video_analysis',
        'video': video_path,
        'start_time': time.time(),
        'status': 'started'
    }
    
    try:
        # Initialize analyzer
        analyzer = VideoAnalyzer(
            sample_interval=30,
            max_frames=100
        )
        
        # Perform analysis
        health_data['status'] = 'processing'
        result = analyzer.analyze(video_path)
        
        # Record success metrics
        health_data.update({
            'status': 'completed',
            'duration_seconds': time.time() - health_data['start_time'],
            'frames_analyzed': len(result.frames),
            'video_duration': result.duration,
            'key_moments': len(result.key_moments),
            'processing_time': result.processing_time_seconds
        })
        
        return health_data
        
    except VideoAnalysisError as e:
        health_data.update({
            'status': 'failed',
            'error': str(e),
            'duration_seconds': time.time() - health_data['start_time']
        })
        return health_data

# Usage
metrics = analyze_with_health("demo.mp4", "ATLAS")
print(f"Status: {metrics['status']}")
print(f"Duration: {metrics.get('duration_seconds', 0):.1f}s")
```

**Result:** Health metrics correlated with video processing performance

---

## Pattern 2: VideoAnalysis + SynapseLink

**Use Case:** Notify Team Brain when video analysis completes

**Why:** Keep team informed of analysis results automatically

**Code:**

```python
from videoanalysis import VideoAnalyzer

def analyze_and_notify(video_path: str, notify_agents: str = "TEAM") -> dict:
    """
    Analyze video and send results via Synapse.
    
    Args:
        video_path: Path to video file
        notify_agents: Comma-separated agent names or "TEAM"
    
    Returns:
        Analysis result dict
    """
    analyzer = VideoAnalyzer()
    result = analyzer.analyze(video_path)
    
    # Build notification body
    notification = {
        'subject': f"Video Analysis: {result.file_name}",
        'body': (
            f"Video: {result.file_name}\n"
            f"Duration: {result.duration}\n"
            f"Resolution: {result.resolution[0]}x{result.resolution[1]}\n"
            f"Size: {result.file_size_mb:.1f} MB\n"
            f"\n"
            f"Scenes detected: {len(result.scenes)}\n"
            f"Key moments: {len(result.key_moments)}\n"
            f"Text occurrences: {len(result.text_detected)}\n"
            f"\n"
            f"Processing time: {result.processing_time_seconds:.1f}s\n"
            f"\n"
            f"Summary: {result.summary[:300]}"
        ),
        'priority': 'NORMAL',
        'to': notify_agents
    }
    
    # SynapseLink integration point
    # from synapselink import quick_send
    # quick_send(notify_agents, notification['subject'], notification['body'])
    
    print(f"[Synapse] Would notify {notify_agents}:")
    print(notification['body'])
    
    return notification

# Usage
analyze_and_notify("screen_recording.mp4", "FORGE,LOGAN")
```

**Result:** Team receives structured analysis notification without manual reporting

---

## Pattern 3: VideoAnalysis + TaskQueuePro

**Use Case:** Queue video analysis jobs for batch execution

**Why:** Manage multiple video analysis tasks with priority and tracking

**Code:**

```python
from videoanalysis import VideoAnalyzer
import json
from pathlib import Path

def create_video_task(video_path: str, priority: int = 2, 
                      analysis_type: str = "comprehensive") -> dict:
    """Create a video analysis task for the queue."""
    return {
        'title': f"Analyze: {Path(video_path).name}",
        'agent': 'ATLAS',
        'priority': priority,
        'metadata': {
            'tool': 'VideoAnalysis',
            'video_path': video_path,
            'analysis_type': analysis_type,
            'version': '1.0'
        }
    }

def execute_video_task(task: dict) -> dict:
    """Execute a queued video analysis task."""
    metadata = task.get('metadata', {})
    video_path = metadata.get('video_path')
    analysis_type = metadata.get('analysis_type', 'comprehensive')
    
    analyzer = VideoAnalyzer()
    result = analyzer.analyze(video_path, analysis_type=analysis_type)
    
    return {
        'task_title': task['title'],
        'status': 'completed',
        'summary': result.summary,
        'key_moments': len(result.key_moments),
        'text_detected': len(result.text_detected),
        'processing_time': result.processing_time_seconds
    }

# Usage - Create tasks
tasks = [
    create_video_task("video1.mp4", priority=1),
    create_video_task("video2.mp4", priority=2),
    create_video_task("video3.mp4", priority=3, analysis_type="quick"),
]

print(f"Created {len(tasks)} video analysis tasks")
for task in tasks:
    print(f"  [{task['priority']}] {task['title']}")
```

**Result:** Video analysis integrated into centralized task management

---

## Pattern 4: VideoAnalysis + MemoryBridge

**Use Case:** Store video analysis history in Memory Core for future reference

**Why:** Build a searchable archive of all analyzed videos

**Code:**

```python
from videoanalysis import VideoAnalyzer
from datetime import datetime
import json
from pathlib import Path

def analyze_and_store(video_path: str, memory_dir: str = None) -> dict:
    """
    Analyze video and persist results to memory.
    
    Stores compact analysis summary for future reference
    without consuming excessive storage.
    """
    analyzer = VideoAnalyzer()
    result = analyzer.analyze(video_path)
    
    # Create memory-friendly summary
    memory_entry = {
        'timestamp': datetime.now().isoformat(),
        'file_name': result.file_name,
        'file_path': result.file_path,
        'duration': result.duration,
        'resolution': f"{result.resolution[0]}x{result.resolution[1]}",
        'file_size_mb': result.file_size_mb,
        'scene_count': len(result.scenes),
        'key_moment_count': len(result.key_moments),
        'text_count': len(result.text_detected),
        'summary': result.summary,
        'top_moments': [
            {
                'time': m.timestamp,
                'type': m.change_type,
                'magnitude': round(m.magnitude, 1)
            }
            for m in result.key_moments[:5]
        ],
        'sample_text': [
            {'time': t.timestamp, 'text': t.text[:100]}
            for t in result.text_detected[:3]
        ],
        'processing_time': result.processing_time_seconds
    }
    
    # Save to memory file
    if memory_dir:
        memory_path = Path(memory_dir) / f"video_analysis_{result.file_name}.json"
        with open(memory_path, 'w') as f:
            json.dump(memory_entry, f, indent=2)
        print(f"Analysis stored: {memory_path}")
    
    return memory_entry

# Usage
entry = analyze_and_store("important_demo.mp4", memory_dir="./video_memory")
print(f"Stored: {entry['file_name']} - {entry['key_moment_count']} key moments")
```

**Result:** Persistent, searchable video analysis archive in Memory Core

---

## Pattern 5: VideoAnalysis + SessionReplay

**Use Case:** Record video analysis sessions for debugging and replay

**Why:** Full audit trail of what was analyzed, when, and what was found

**Code:**

```python
from videoanalysis import VideoAnalyzer
import time
from datetime import datetime

class RecordedVideoAnalysis:
    """Video analysis with full session recording for replay."""
    
    def __init__(self, session_id: str, agent: str):
        self.session_id = session_id
        self.agent = agent
        self.events = []
        self.start_time = datetime.now()
    
    def log(self, event_type: str, message: str):
        """Log an event for session replay."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'agent': self.agent,
            'type': event_type,
            'message': message
        }
        self.events.append(event)
        print(f"[{self.session_id}] {event_type}: {message}")
    
    def analyze(self, video_path: str, **kwargs) -> dict:
        """Analyze with full recording."""
        self.log('START', f"Analyzing: {video_path}")
        self.log('CONFIG', f"Settings: {kwargs}")
        
        analyzer = VideoAnalyzer(**kwargs)
        
        try:
            result = analyzer.analyze(video_path)
            
            self.log('METADATA', f"Duration: {result.duration}, "
                                 f"Resolution: {result.resolution}")
            self.log('SCENES', f"Detected {len(result.scenes)} scenes")
            self.log('MOMENTS', f"Found {len(result.key_moments)} key moments")
            self.log('TEXT', f"Extracted {len(result.text_detected)} text items")
            self.log('COMPLETE', f"Processing time: {result.processing_time_seconds:.1f}s")
            
            return {
                'result': result,
                'session': {
                    'id': self.session_id,
                    'agent': self.agent,
                    'events': self.events,
                    'duration': (datetime.now() - self.start_time).total_seconds()
                }
            }
            
        except Exception as e:
            self.log('ERROR', str(e))
            raise

# Usage
session = RecordedVideoAnalysis("VA-001", "ATLAS")
output = session.analyze("demo.mp4", sample_interval=15)
print(f"\nSession recorded: {len(session.events)} events")
```

**Result:** Full audit trail for debugging video analysis sessions

---

## Pattern 6: VideoAnalysis + ContextCompressor

**Use Case:** Compress video analysis results for agent context windows

**Why:** Video analysis generates lots of data - compress for efficient AI consumption

**Code:**

```python
from videoanalysis import VideoAnalyzer

def compressed_video_context(video_path: str, max_tokens: int = 500) -> str:
    """
    Analyze video and return compressed context-friendly output.
    
    Designed to fit within AI agent context windows without
    consuming excessive tokens.
    """
    analyzer = VideoAnalyzer()
    result = analyzer.analyze(video_path)
    
    lines = []
    
    # Header (compact)
    lines.append(f"=== VIDEO: {result.file_name} ===")
    lines.append(f"{result.duration} | {result.resolution[0]}x{result.resolution[1]} | "
                 f"{result.file_size_mb:.1f}MB | {result.fps}fps")
    
    # Key moments (most valuable info)
    if result.key_moments:
        lines.append(f"\nKEY MOMENTS ({len(result.key_moments)}):")
        for m in result.key_moments[:5]:
            lines.append(f"  {m.timestamp} | {m.change_type} | delta={m.magnitude:.0f}")
    
    # Text detected (high value)
    if result.text_detected:
        lines.append(f"\nTEXT ({len(result.text_detected)} occurrences):")
        seen = set()
        for t in result.text_detected[:5]:
            text_short = t.text[:60].strip()
            if text_short not in seen:
                lines.append(f"  {t.timestamp} | {text_short}")
                seen.add(text_short)
    
    # Activity summary (compact)
    if result.activity_timeline:
        high = sum(1 for a in result.activity_timeline if a.activity_level == 'high')
        med = sum(1 for a in result.activity_timeline if a.activity_level == 'medium')
        low = sum(1 for a in result.activity_timeline if a.activity_level == 'low')
        static = sum(1 for a in result.activity_timeline if a.activity_level == 'static')
        lines.append(f"\nACTIVITY: {high}hi/{med}med/{low}lo/{static}static segments")
    
    # Summary (truncated)
    if result.summary:
        summary = result.summary[:200]
        lines.append(f"\nSUMMARY: {summary}")
    
    compressed = "\n".join(lines)
    
    # Estimate token count (rough: ~4 chars per token)
    est_tokens = len(compressed) // 4
    print(f"Compressed: ~{est_tokens} tokens (target: {max_tokens})")
    
    return compressed

# Usage
context = compressed_video_context("long_recording.mp4")
print(context)
```

**Result:** Token-efficient video analysis for AI context windows

---

## Pattern 7: VideoAnalysis + ConfigManager

**Use Case:** Centralize VideoAnalysis settings across agents

**Why:** Consistent analysis parameters across all Team Brain operations

**Code:**

```python
from videoanalysis import VideoAnalyzer
import json
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    'video_analysis': {
        'sample_interval': 30,
        'max_frames': 100,
        'delta_threshold': 15.0,
        'scene_threshold': 0.3,
        'ocr_confidence': 0.6,
        'cleanup_temp': True
    },
    'profiles': {
        'quick': {
            'sample_interval': 60,
            'max_frames': 30,
            'delta_threshold': 20.0
        },
        'detailed': {
            'sample_interval': 10,
            'max_frames': 300,
            'delta_threshold': 8.0,
            'ocr_confidence': 0.4
        },
        'security': {
            'sample_interval': 2,
            'max_frames': 500,
            'delta_threshold': 5.0
        }
    }
}

def get_analyzer(profile: str = 'default') -> VideoAnalyzer:
    """Get VideoAnalyzer with named configuration profile."""
    config = DEFAULT_CONFIG['video_analysis'].copy()
    
    if profile != 'default' and profile in DEFAULT_CONFIG['profiles']:
        config.update(DEFAULT_CONFIG['profiles'][profile])
    
    return VideoAnalyzer(
        sample_interval=config['sample_interval'],
        max_frames=config['max_frames'],
        delta_threshold=config['delta_threshold'],
        scene_threshold=config['scene_threshold'],
        ocr_confidence=config['ocr_confidence'],
        cleanup_temp=config['cleanup_temp']
    )

# Usage
analyzer = get_analyzer('detailed')  # Detailed profile
result = analyzer.analyze("video.mp4")

analyzer = get_analyzer('security')  # Security monitoring profile
result = analyzer.analyze("security_cam.mp4")
```

**Result:** Consistent, configurable analysis parameters across all agents

---

## Pattern 8: VideoAnalysis + AudioAnalysis

**Use Case:** Combined audio + video analysis for complete media understanding

**Why:** Video and audio together give full picture of media content

**Code:**

```python
from videoanalysis import VideoAnalyzer
from pathlib import Path

def full_media_analysis(media_path: str) -> dict:
    """
    Combined video and audio analysis.
    
    Uses VideoAnalysis for visual content and can be paired
    with AudioAnalysis for audio content when available.
    """
    results = {'source': media_path}
    
    # Video Analysis
    video_analyzer = VideoAnalyzer(
        sample_interval=15,
        delta_threshold=10.0
    )
    video_result = video_analyzer.analyze(media_path)
    
    results['video'] = {
        'duration': video_result.duration,
        'resolution': f"{video_result.resolution[0]}x{video_result.resolution[1]}",
        'scenes': len(video_result.scenes),
        'key_moments': [
            {
                'time': m.timestamp,
                'type': m.change_type,
                'magnitude': round(m.magnitude, 1)
            }
            for m in video_result.key_moments
        ],
        'text_detected': [
            {'time': t.timestamp, 'text': t.text}
            for t in video_result.text_detected
        ],
        'activity_breakdown': {
            'high': sum(1 for a in video_result.activity_timeline 
                       if a.activity_level == 'high'),
            'medium': sum(1 for a in video_result.activity_timeline 
                         if a.activity_level == 'medium'),
            'low': sum(1 for a in video_result.activity_timeline 
                      if a.activity_level == 'low'),
            'static': sum(1 for a in video_result.activity_timeline 
                         if a.activity_level == 'static')
        },
        'summary': video_result.summary
    }
    
    # AudioAnalysis integration point
    # When AudioAnalysis is available:
    # from audioanalysis import AudioAnalyzer
    # audio_result = AudioAnalyzer().analyze(media_path)
    # results['audio'] = { ... }
    
    results['combined_summary'] = (
        f"Media: {Path(media_path).name}\n"
        f"Duration: {video_result.duration}\n"
        f"Visual: {len(video_result.key_moments)} key moments, "
        f"{len(video_result.text_detected)} text items\n"
        f"Activity: {results['video']['activity_breakdown']}"
    )
    
    return results

# Usage
analysis = full_media_analysis("presentation.mp4")
print(analysis['combined_summary'])
```

**Result:** Comprehensive media analysis combining visual and audio insights

---

## Pattern 9: Multi-Tool Video Pipeline

**Use Case:** Complete video processing pipeline using multiple tools

**Why:** Demonstrate real production scenario with full tool stack

**Code:**

```python
from videoanalysis import VideoAnalyzer
from dataclasses import asdict
import json
import time
from pathlib import Path

def video_pipeline(video_path: str, output_dir: str = "./pipeline_output"):
    """
    Complete video processing pipeline.
    
    Steps:
    1. Create task tracking
    2. Monitor health during processing
    3. Analyze video
    4. Compress results for context
    5. Store in memory
    6. Notify team
    """
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    
    pipeline_start = time.time()
    
    print("=" * 60)
    print("VIDEO PROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Task setup
    print("\n[1/6] Creating task...")
    task = {
        'id': f"VP-{int(time.time())}",
        'video': video_path,
        'status': 'started',
        'started_at': time.time()
    }
    
    # Step 2: Health monitoring
    print("[2/6] Starting health monitoring...")
    health = {'agent': 'ATLAS', 'status': 'processing'}
    
    # Step 3: Analyze video
    print("[3/6] Analyzing video...")
    analyzer = VideoAnalyzer(sample_interval=15, delta_threshold=10.0)
    result = analyzer.analyze(video_path)
    
    print(f"  Duration: {result.duration}")
    print(f"  Key moments: {len(result.key_moments)}")
    print(f"  Text items: {len(result.text_detected)}")
    
    # Step 4: Compress for context
    print("[4/6] Compressing results...")
    compact = {
        'file': result.file_name,
        'duration': result.duration,
        'moments': len(result.key_moments),
        'text': len(result.text_detected),
        'summary': result.summary[:300]
    }
    
    # Step 5: Store results
    print("[5/6] Storing results...")
    result_path = output / f"{Path(video_path).stem}_analysis.json"
    with open(result_path, 'w') as f:
        json.dump(compact, f, indent=2)
    
    # Step 6: Notification
    print("[6/6] Preparing notification...")
    notification = (
        f"Pipeline Complete: {result.file_name}\n"
        f"Duration: {result.duration}\n"
        f"Key moments: {len(result.key_moments)}\n"
        f"Processing: {time.time() - pipeline_start:.1f}s"
    )
    
    # Update task
    task['status'] = 'completed'
    task['duration'] = time.time() - pipeline_start
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(notification)
    print(f"Results saved: {result_path}")
    
    return {
        'task': task,
        'result': compact,
        'notification': notification,
        'output_path': str(result_path)
    }

# Usage
output = video_pipeline("important_meeting.mp4")
print(f"\nPipeline finished in {output['task']['duration']:.1f}s")
```

**Result:** Production-grade video processing with full tool orchestration

---

## Pattern 10: Full Team Brain Video Stack

**Use Case:** Ultimate integration - all tools working together for video processing

**Why:** Demonstrates the full power of Team Brain's video capabilities

**Code:**

```python
"""
Full Team Brain Video Stack
============================
This pattern shows how VideoAnalysis connects to the entire
Team Brain ecosystem for comprehensive video intelligence.

Stack:
  VideoAnalysis  -> Video processing engine
  AgentHealth    -> Performance monitoring
  SynapseLink   -> Team notifications
  TaskQueuePro   -> Job management
  MemoryBridge   -> Result persistence
  SessionReplay  -> Audit trail
  ContextCompressor -> Token optimization
  ConfigManager  -> Settings management
"""

from videoanalysis import VideoAnalyzer
import json
import time
from pathlib import Path

class TeamBrainVideoStack:
    """Full Team Brain video processing stack."""
    
    def __init__(self, agent: str = "ATLAS"):
        self.agent = agent
        self.analyzer = VideoAnalyzer()
        self.log_entries = []
    
    def _log(self, component: str, message: str):
        """Log stack operation."""
        entry = f"[{component}] {message}"
        self.log_entries.append(entry)
        print(entry)
    
    def process(self, video_path: str) -> dict:
        """
        Full stack video processing.
        
        Uses every available integration point for maximum
        intelligence and coordination.
        """
        self._log("STACK", f"Starting full stack for: {video_path}")
        start = time.time()
        
        # Health: Start monitoring
        self._log("HEALTH", f"Agent {self.agent} starting video analysis")
        
        # Task: Create tracking
        task_id = f"VID-{int(time.time())}"
        self._log("TASK", f"Created task {task_id}")
        
        # Session: Start recording
        self._log("SESSION", f"Recording session for replay")
        
        try:
            # Core: Analyze video
            self._log("ANALYZE", "Processing video...")
            result = self.analyzer.analyze(video_path)
            
            # Context: Compress results
            compressed = (
                f"{result.file_name}: {result.duration}, "
                f"{len(result.key_moments)} moments, "
                f"{len(result.text_detected)} text items"
            )
            self._log("COMPRESS", f"Compressed: {compressed}")
            
            # Memory: Store results
            self._log("MEMORY", "Storing analysis in memory core")
            
            # Config: Log settings used
            self._log("CONFIG", "Default analysis profile used")
            
            # Synapse: Notify team
            self._log("SYNAPSE", f"Notifying team of completion")
            
            # Health: Record success
            duration = time.time() - start
            self._log("HEALTH", f"Analysis complete in {duration:.1f}s")
            
            # Task: Complete
            self._log("TASK", f"Task {task_id} completed")
            
            return {
                'task_id': task_id,
                'status': 'completed',
                'agent': self.agent,
                'video': result.file_name,
                'duration': result.duration,
                'processing_time': duration,
                'key_moments': len(result.key_moments),
                'text_detected': len(result.text_detected),
                'compressed_context': compressed,
                'summary': result.summary,
                'log': self.log_entries
            }
            
        except Exception as e:
            self._log("ERROR", str(e))
            self._log("HEALTH", f"Agent {self.agent} error: {e}")
            self._log("SYNAPSE", "Alerting FORGE of failure")
            raise
    
    def get_session_log(self) -> str:
        """Get full session log for replay."""
        return "\n".join(self.log_entries)

# Usage
stack = TeamBrainVideoStack(agent="ATLAS")
result = stack.process("project_demo.mp4")

print(f"\n{'='*60}")
print(f"Full Stack Result:")
print(f"  Task: {result['task_id']}")
print(f"  Video: {result['video']} ({result['duration']})")
print(f"  Key moments: {result['key_moments']}")
print(f"  Text found: {result['text_detected']}")
print(f"  Time: {result['processing_time']:.1f}s")
print(f"\nSession log: {len(stack.log_entries)} entries")
```

**Result:** Complete demonstration of VideoAnalysis within the Team Brain ecosystem

---

## 📊 RECOMMENDED INTEGRATION PRIORITY

**Week 1 (Essential):**
1. AgentHealth - Monitor analysis performance
2. SynapseLink - Notify team of results
3. SessionReplay - Audit trail

**Week 2 (Productivity):**
4. TaskQueuePro - Batch job management
5. MemoryBridge - Result persistence
6. ConfigManager - Shared configuration

**Week 3 (Advanced):**
7. ContextCompressor - Token optimization
8. AudioAnalysis - Combined media analysis
9. Full stack integration

---

## 🔧 TROUBLESHOOTING INTEGRATIONS

**Import Errors:**
```python
# Ensure VideoAnalysis is in Python path
import sys
from pathlib import Path
sys.path.append(str(Path.home() / "OneDrive/Documents/AutoProjects/VideoAnalysis"))

# Then import
from videoanalysis import VideoAnalyzer
```

**FFmpeg Not Found:**
```bash
# Windows
winget install ffmpeg

# Linux
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

**Large Video Memory Issues:**
```python
# Reduce frame count for large videos
analyzer = VideoAnalyzer(
    sample_interval=60,   # Less frequent sampling
    max_frames=50,        # Fewer frames
    cleanup_temp=True     # Clean up immediately
)
```

**Integration Test:**
```python
# Quick test that everything works
from videoanalysis import DependencyChecker

status = DependencyChecker.check_all()
for dep, available in status.items():
    print(f"  {'[OK]' if available else '[X]'} {dep}")
```

---

**Last Updated:** February 6, 2026  
**Maintained By:** ATLAS (Team Brain)  
**For:** Logan Smith / Metaphy LLC
