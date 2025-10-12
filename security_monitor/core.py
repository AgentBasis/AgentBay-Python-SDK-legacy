"""
Core Security Monitoring Framework

Provides the foundational classes and functionality for security monitoring:
- SecurityMonitor: Main monitoring orchestrator
- SecurityEvent: Event data structure
- SecurityMetrics: Metrics collection and analysis
"""

import threading
import json
import time
from typing import Dict, List, Any, Optional, Callable, Union, Iterable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import hashlib
import uuid

from .detectors import DetectionResult, SecurityDetector


class SecurityEventType(Enum):
    """Types of security events that can be detected."""
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    SOCIAL_ENGINEERING = "social_engineering"
    MANIPULATION_ATTEMPT = "manipulation_attempt"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    POLICY_VIOLATION = "policy_violation"
    DATA_LEAK_RISK = "data_leak_risk"
    COMPLIANCE_VIOLATION = "compliance_violation"
    ANOMALY_DETECTED = "anomaly_detected"
    SECURITY_ALERT = "security_alert"


class SecurityLevel(Enum):
    """Security alert levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Represents a security event detected during monitoring."""
    
    # Event identification
    event_id: str
    event_type: SecurityEventType
    timestamp: datetime
    
    # Event details
    agent_id: str
    session_id: Optional[str]
    user_id: Optional[str]
    
    # Security assessment
    risk_score: float  # 0.0 to 1.0
    severity: SecurityLevel
    confidence: float  # 0.0 to 1.0
    
    # Event data
    description: str
    raw_data: Dict[str, Any]
    context: Dict[str, Any]
    
    # Detection metadata
    detector_name: str
    detection_rules: List[str]
    
    # Response tracking
    response_required: bool = False
    response_taken: Optional[str] = None
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        event_dict = asdict(self)
        # Convert enum values to strings
        event_dict['event_type'] = self.event_type.value
        event_dict['severity'] = self.severity.value
        # Convert datetime to ISO string
        event_dict['timestamp'] = self.timestamp.isoformat()
        return event_dict
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityEvent':
        """Create SecurityEvent from dictionary."""
        # Convert string values back to enums
        data['event_type'] = SecurityEventType(data['event_type'])
        data['severity'] = SecurityLevel(data['severity'])
        # Convert ISO string back to datetime
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class SecurityMetrics:
    """Security monitoring metrics and statistics."""
    
    # Time period
    start_time: datetime
    end_time: datetime
    
    # Event counts
    total_events: int = 0
    events_by_type: Dict[str, int] = None
    events_by_severity: Dict[str, int] = None
    
    # Risk assessment
    average_risk_score: float = 0.0
    max_risk_score: float = 0.0
    high_risk_events: int = 0
    
    # Agent statistics
    agents_monitored: int = 0
    sessions_monitored: int = 0
    unique_users: int = 0
    
    # Detection performance
    detection_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    response_time_avg: float = 0.0
    
    def __post_init__(self):
        if self.events_by_type is None:
            self.events_by_type = {}
        if self.events_by_severity is None:
            self.events_by_severity = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        metrics_dict = asdict(self)
        metrics_dict['start_time'] = self.start_time.isoformat()
        metrics_dict['end_time'] = self.end_time.isoformat()
        return metrics_dict


class SecurityMonitor:
    """
    Main security monitoring orchestrator.
    
    Coordinates security monitoring components and provides a unified interface
    for tracking security events and metrics.
    """
    
    def __init__(
        self,
        risk_threshold: float = 0.7,
        alert_webhook: Optional[str] = None,
        privacy_mode: bool = False,
        max_events_memory: int = 10000,
        metrics_interval: int = 300,  # 5 minutes
        detectors: Optional[Iterable[SecurityDetector]] = None,
        **kwargs
    ):
        """
        Initialize SecurityMonitor.
        
        Args:
            risk_threshold: Risk score threshold for alerts (0.0-1.0)
            alert_webhook: Webhook URL for security alerts
            privacy_mode: Enable privacy mode for minimal data collection
            max_events_memory: Maximum number of events to keep in memory
            metrics_interval: Interval for metrics calculation (seconds)
            detectors: Optional iterable of SecurityDetector instances
        """
        self.risk_threshold = risk_threshold
        self.alert_webhook = alert_webhook
        self.privacy_mode = privacy_mode
        self.max_events_memory = max_events_memory
        self.metrics_interval = metrics_interval
        
        # Internal state
        self._monitoring_active = False
        self._components = []
        self._event_handlers = []
        self._events = []
        self._metrics_history = []
        self._lock = threading.Lock()
        self._metrics_thread = None
        
        # Event storage
        self._events_by_id = {}
        self._events_by_agent = {}
        self._events_by_session = {}
        
        # Statistics
        self._start_time = None
        self._total_interactions = 0
        self._unique_agents = set()
        self._unique_sessions = set()
        self._unique_users = set()

        # Detectors
        self._detectors: List[SecurityDetector] = list(detectors or [])
        
        print(f"🔒 SecurityMonitor initialized (threshold: {risk_threshold})")
    
    def add_component(self, component) -> None:
        """Add a monitoring component (context monitor, compliance monitor, etc.)."""
        self._components.append(component)
        # Set up bidirectional communication
        if hasattr(component, 'set_security_monitor'):
            component.set_security_monitor(self)
        print(f"   Added component: {component.__class__.__name__}")
    
    def add_event_handler(self, handler: Callable[[SecurityEvent], None]) -> None:
        """Add an event handler function."""
        self._event_handlers.append(handler)
    
    def register_detector(self, detector: SecurityDetector) -> None:
        """Register a security detector."""
        if detector not in self._detectors:
            self._detectors.append(detector)
    
    def register_detectors(self, detectors: Iterable[SecurityDetector]) -> None:
        """Register multiple security detectors."""
        for detector in detectors:
            self.register_detector(detector)
    
    def start_monitoring(self) -> None:
        """Start security monitoring."""
        if self._monitoring_active:
            print("⚠️  Security monitoring already active")
            return
        
        self._monitoring_active = True
        self._start_time = datetime.now(timezone.utc)
        
        # Start components
        for component in self._components:
            if hasattr(component, 'start'):
                component.start()
        
        # Start metrics collection thread
        self._start_metrics_thread()
        
        print("✅ Security monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop security monitoring."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        
        # Stop components
        for component in self._components:
            if hasattr(component, 'stop'):
                component.stop()
        
        # Stop metrics thread
        if self._metrics_thread and self._metrics_thread.is_alive():
            self._metrics_thread.join(timeout=1)
        
        print("🛑 Security monitoring stopped")
    
    def track_interaction(
        self,
        agent_id: str,
        user_input: str,
        agent_response: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track an agent-user interaction for security analysis.
        
        Args:
            agent_id: Identifier for the agent
            user_input: User's input to the agent
            agent_response: Agent's response
            session_id: Optional session identifier
            user_id: Optional user identifier
            metadata: Optional additional metadata
        """
        if not self._monitoring_active:
            return
        
        # Update statistics
        self._total_interactions += 1
        self._unique_agents.add(agent_id)
        if session_id:
            self._unique_sessions.add(session_id)
        if user_id:
            self._unique_users.add(user_id)
        
        metadata = metadata or {}
        
        self._run_detectors(
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            user_input=user_input,
            agent_response=agent_response,
        )
        
        # Create interaction data
        interaction_data = {
            'agent_id': agent_id,
            'user_input': user_input if not self.privacy_mode else self._hash_content(user_input),
            'agent_response': agent_response if not self.privacy_mode else self._hash_content(agent_response),
            'session_id': session_id,
            'user_id': user_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metadata': metadata
        }
        
        # Pass to components for analysis
        for component in self._components:
            if hasattr(component, 'analyze_interaction'):
                try:
                    component.analyze_interaction(interaction_data)
                except Exception as e:
                    print(f"⚠️  Error in component {component.__class__.__name__}: {e}")

    def _run_detectors(
        self,
        *,
        agent_id: str,
        session_id: Optional[str],
        user_id: Optional[str],
        metadata: Dict[str, Any],
        user_input: Optional[str],
        agent_response: Optional[str],
    ) -> None:
        """Execute registered detectors against interaction content."""
        if not self._detectors:
            return
        
        base_context = {
            "agent_id": agent_id,
            "session_id": session_id,
            "user_id": user_id,
            "metadata": metadata,
        }
        
        surfaces = [
            ("user_input", user_input),
            ("agent_response", agent_response),
        ]
        
        for surface, text in surfaces:
            if not text:
                continue
            
            context = dict(base_context)
            context["surface"] = surface
            
            for detector in self._detectors:
                try:
                    result = detector.detect(text, context=context)
                except Exception as exc:
                    print(f"⚠️  Detector {detector.name} failed: {exc}")
                    continue
                
                self._handle_detection_result(
                    detector=detector,
                    result=result,
                    base_context=base_context,
                    surface=surface,
                    text=text,
                )

    def _handle_detection_result(
        self,
        *,
        detector: SecurityDetector,
        result: DetectionResult,
        base_context: Dict[str, Any],
        surface: str,
        text: str,
    ) -> None:
        """Convert detector output into security events."""
        if not isinstance(result, DetectionResult) or not result.detected:
            return
        
        event_type = self._map_threat_type(result.threat_type)
        detector_metadata = dict(result.metadata or {})
        raw_data = {
            "surface": surface,
            "text_signature": self._hash_content(text),
            "indicators": list(result.indicators),
            "detector_metadata": detector_metadata,
        }
        
        if not self.privacy_mode and result.threat_type != "pii":
            raw_data["content_excerpt"] = text[:200]
        
        context_payload = dict(base_context)
        context_payload["surface"] = surface
        context_payload["detector"] = detector.name
        context_payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        event = create_security_event(
            event_type=event_type,
            agent_id=base_context.get("agent_id", "unknown"),
            risk_score=result.risk_score,
            description=result.description,
            detector_name=detector.name,
            raw_data=raw_data,
            session_id=base_context.get("session_id"),
            user_id=base_context.get("user_id"),
            context=context_payload,
            detection_rules=result.indicators,
        )
        self.record_event(event)

    def _map_threat_type(self, threat_type: str) -> SecurityEventType:
        """Map detector threat types to security event enums."""
        mapping = {
            "prompt_injection": SecurityEventType.PROMPT_INJECTION,
            "jailbreak": SecurityEventType.JAILBREAK_ATTEMPT,
            "social_engineering": SecurityEventType.SOCIAL_ENGINEERING,
            "manipulation": SecurityEventType.MANIPULATION_ATTEMPT,
            "pii": SecurityEventType.DATA_LEAK_RISK,
        }
        return mapping.get(threat_type, SecurityEventType.ANOMALY_DETECTED)
    
    def record_event(self, event: SecurityEvent) -> None:
        """Record a security event."""
        with self._lock:
            # Add to events list
            self._events.append(event)
            
            # Maintain memory limit
            if len(self._events) > self.max_events_memory:
                removed_event = self._events.pop(0)
                self._events_by_id.pop(removed_event.event_id, None)
            
            # Index event
            self._events_by_id[event.event_id] = event
            
            # Index by agent
            if event.agent_id not in self._events_by_agent:
                self._events_by_agent[event.agent_id] = []
            self._events_by_agent[event.agent_id].append(event)
            
            # Index by session
            if event.session_id:
                if event.session_id not in self._events_by_session:
                    self._events_by_session[event.session_id] = []
                self._events_by_session[event.session_id].append(event)
        
        # Process event
        self._process_event(event)
        
        # Notify handlers
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"⚠️  Error in event handler: {e}")
    
    def get_events(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type: Optional[SecurityEventType] = None,
        severity: Optional[SecurityLevel] = None,
        limit: Optional[int] = None
    ) -> List[SecurityEvent]:
        """Get security events with optional filtering."""
        with self._lock:
            events = self._events.copy()
        
        # Apply filters
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if severity:
            events = [e for e in events if e.severity == severity]
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            events = events[:limit]
        
        return events
    
    def get_metrics(self, time_range: Optional[int] = None) -> SecurityMetrics:
        """
        Get security monitoring metrics.
        
        Args:
            time_range: Time range in seconds (None for all time)
        """
        end_time = datetime.now(timezone.utc)
        start_time = self._start_time or end_time
        
        if time_range:
            start_time = end_time - timedelta(seconds=time_range)
        
        # Filter events by time range
        relevant_events = [
            e for e in self._events
            if start_time <= e.timestamp <= end_time
        ]
        
        # Calculate metrics
        events_by_type = {}
        events_by_severity = {}
        risk_scores = []
        
        for event in relevant_events:
            # Count by type
            event_type_str = event.event_type.value
            events_by_type[event_type_str] = events_by_type.get(event_type_str, 0) + 1
            
            # Count by severity
            severity_str = event.severity.value
            events_by_severity[severity_str] = events_by_severity.get(severity_str, 0) + 1
            
            # Collect risk scores
            risk_scores.append(event.risk_score)
        
        # Calculate risk statistics
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        max_risk = max(risk_scores) if risk_scores else 0.0
        high_risk_events = len([s for s in risk_scores if s >= self.risk_threshold])
        
        return SecurityMetrics(
            start_time=start_time,
            end_time=end_time,
            total_events=len(relevant_events),
            events_by_type=events_by_type,
            events_by_severity=events_by_severity,
            average_risk_score=avg_risk,
            max_risk_score=max_risk,
            high_risk_events=high_risk_events,
            agents_monitored=len(self._unique_agents),
            sessions_monitored=len(self._unique_sessions),
            unique_users=len(self._unique_users)
        )
    
    def _process_event(self, event: SecurityEvent) -> None:
        """Process a security event (alerts, logging, etc.)."""
        # Check if event requires immediate attention
        if event.risk_score >= self.risk_threshold or event.severity == SecurityLevel.CRITICAL:
            self._handle_high_risk_event(event)
        
        # Log event
        print(f"🚨 Security Event: {event.event_type.value} | Risk: {event.risk_score:.2f} | Agent: {event.agent_id}")
        
        if not self.privacy_mode:
            print(f"   Description: {event.description}")
    
    def _handle_high_risk_event(self, event: SecurityEvent) -> None:
        """Handle high-risk security events."""
        print(f"🔥 HIGH RISK EVENT DETECTED: {event.event_type.value}")
        print(f"   Risk Score: {event.risk_score:.2f}")
        print(f"   Severity: {event.severity.value}")
        print(f"   Agent: {event.agent_id}")
        
        # Send webhook alert if configured
        if self.alert_webhook:
            self._send_webhook_alert(event)
    
    def _send_webhook_alert(self, event: SecurityEvent) -> None:
        """Send webhook alert for high-risk events."""
        try:
            import requests
            
            payload = {
                'event_type': event.event_type.value,
                'risk_score': event.risk_score,
                'severity': event.severity.value,
                'agent_id': event.agent_id,
                'timestamp': event.timestamp.isoformat(),
                'description': event.description
            }
            
            requests.post(self.alert_webhook, json=payload, timeout=10)
            print(f"   ✅ Alert sent to webhook")
            
        except Exception as e:
            print(f"   ❌ Failed to send webhook alert: {e}")
    
    def _hash_content(self, content: str) -> str:
        """Hash content for privacy mode."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _start_metrics_thread(self) -> None:
        """Start background thread for metrics collection."""
        def metrics_loop():
            while self._monitoring_active:
                try:
                    metrics = self.get_metrics(time_range=self.metrics_interval)
                    self._metrics_history.append(metrics)
                    
                    # Keep only last 100 metrics snapshots
                    if len(self._metrics_history) > 100:
                        self._metrics_history.pop(0)
                    
                    time.sleep(self.metrics_interval)
                    
                except Exception as e:
                    print(f"⚠️  Error in metrics collection: {e}")
                    time.sleep(60)  # Wait before retrying
        
        self._metrics_thread = threading.Thread(target=metrics_loop, daemon=True)
        self._metrics_thread.start()
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


def create_security_event(
    event_type: SecurityEventType,
    agent_id: str,
    risk_score: float,
    description: str,
    detector_name: str,
    raw_data: Dict[str, Any],
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    detection_rules: Optional[List[str]] = None
) -> SecurityEvent:
    """Helper function to create SecurityEvent instances."""
    
    # Determine severity based on risk score
    if risk_score >= 0.9:
        severity = SecurityLevel.CRITICAL
    elif risk_score >= 0.7:
        severity = SecurityLevel.HIGH
    elif risk_score >= 0.4:
        severity = SecurityLevel.MEDIUM
    else:
        severity = SecurityLevel.LOW
    
    return SecurityEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        timestamp=datetime.now(timezone.utc),
        agent_id=agent_id,
        session_id=session_id,
        user_id=user_id,
        risk_score=risk_score,
        severity=severity,
        confidence=min(risk_score + 0.1, 1.0),  # Simple confidence calculation
        description=description,
        raw_data=raw_data,
        context=context or {},
        detector_name=detector_name,
        detection_rules=detection_rules or []
    )
