#!/usr/bin/env python3
"""
Security Add-on Module for AI Agent Tracking SDK

This module provides security features as a pure add-on without modifying
the core SDK. It implements:

1. Tamper detection through file checksum verification
2. Unclosed session metrics for security monitoring  
3. Security flags injection into session events
4. Security event queueing and backend communication

Usage:
    # Wrap existing tracker with security
    base_tracker = AgentPerformanceTracker(...)
    secure_tracker = SecurityWrapper(base_tracker, enable_security=True)
    
    # Use normally - security features work automatically
    session_id = secure_tracker.start_conversation("agent1", "user1")
"""

import os
import hashlib
import logging
import threading
import asyncio
import time
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque
import requests
import aiohttp

@dataclass
class SecurityFlags:
    """Security flags for session events"""
    tamper_detected: bool = False
    pii_detected: bool = False  # For future implementation
    compliance_violation: bool = False  # For future implementation
    
    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)

@dataclass
class UnclosedSessionInfo:
    """Information about unclosed sessions for security metrics"""
    session_id: str
    start_time: str
    agent_id: str
    duration_hours: float

@dataclass
class SecurityMetricEvent:
    """Security metric event data structure"""
    event_type: str
    timestamp: str
    agent_id: str
    client_id: str
    unclosed_count: int
    unclosed_sessions: List[Dict[str, Any]]

@dataclass
class TamperDetectionEvent:
    """Tamper detection event data structure"""
    event_type: str
    timestamp: str
    agent_id: str
    client_id: str
    sdk_version: str
    checksum_expected: str
    checksum_actual: str
    modified_files: List[str]

@dataclass
class SecurityAPIResponse:
    """Security API response structure"""
    success: bool
    status_code: int = 0
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SecurityManager:
    """Manages security features for the AI Agent Tracking SDK"""
    
    def __init__(self, client_id: str, sdk_version: str = "1.2.1", 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Security Manager
        
        Args:
            client_id: Unique identifier for this SDK client instance
            sdk_version: Version of the SDK
            logger: Optional logger instance
        """
        self.client_id = client_id
        self.sdk_version = sdk_version
        self.logger = logger or logging.getLogger(__name__)
        
        # Security state
        self._tamper_detected = False
        self._security_lock = threading.RLock()
        
        # Expected checksums for SDK files
        self._expected_checksums: Dict[str, str] = {}
        self._sdk_files: List[str] = []
        
        # Initialize tamper detection
        self._discover_sdk_files()
        self._generate_expected_checksums()
        self._perform_initial_tamper_check()
    
    def _discover_sdk_files(self):
        """Discover main SDK Python files for checksum verification"""
        try:
            # Get the directory where this security.py file is located
            sdk_dir = Path(__file__).parent
            
            # List of main SDK files to monitor
            main_files = [
                'AgentOper.py',
                'AgentPerform.py', 
                'security.py',
                '__init__.py'
            ]
            
            for filename in main_files:
                file_path = sdk_dir / filename
                if file_path.exists():
                    self._sdk_files.append(str(file_path))
                    self.logger.debug(f"Added SDK file for monitoring: {filename}")
            
            self.logger.info(f"Discovered {len(self._sdk_files)} SDK files for tamper detection")
            
        except Exception as e:
            self.logger.error(f"Error discovering SDK files: {e}")
    
    def _generate_expected_checksums(self):
        """Generate expected SHA256 checksums for SDK files"""
        try:
            for file_path in self._sdk_files:
                checksum = self._calculate_file_checksum(file_path)
                if checksum:
                    filename = Path(file_path).name
                    self._expected_checksums[filename] = checksum
                    self.logger.debug(f"Generated checksum for {filename}: {checksum[:16]}...")
            
            self.logger.info(f"Generated checksums for {len(self._expected_checksums)} files")
            
        except Exception as e:
            self.logger.error(f"Error generating expected checksums: {e}")
    
    def _calculate_file_checksum(self, file_path: str) -> Optional[str]:
        """Calculate SHA256 checksum for a file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating checksum for {file_path}: {e}")
            return None
    
    def _perform_initial_tamper_check(self):
        """Perform initial tamper detection during initialization"""
        modified_files = self._check_tampering()
        if modified_files:
            with self._security_lock:
                self._tamper_detected = True
            self.logger.critical(f"TAMPER DETECTION: {len(modified_files)} files modified: {modified_files}")
        else:
            self.logger.info("Tamper detection: All files verified successfully")
    
    def _check_tampering(self) -> List[str]:
        """Check for tampering and return list of modified files"""
        try:
            modified_files = []
            
            for file_path in self._sdk_files:
                filename = Path(file_path).name
                expected_checksum = self._expected_checksums.get(filename)
                
                if not expected_checksum:
                    continue
                
                current_checksum = self._calculate_file_checksum(file_path)
                
                if current_checksum != expected_checksum:
                    modified_files.append(filename)
                    self.logger.warning(f"Tamper detected in {filename}")
            
            return modified_files
                
        except Exception as e:
            self.logger.error(f"Error during tamper detection: {e}")
            return []
    
    def create_tamper_detection_event(self, agent_id: str) -> TamperDetectionEvent:
        """Create a tamper detection event"""
        modified_files = self._check_tampering()
        
        # Get the first modified file for checksum comparison
        first_modified = modified_files[0] if modified_files else "unknown"
        expected_checksum = self._expected_checksums.get(first_modified, "unknown")
        
        # Calculate current checksum for the first modified file
        actual_checksum = "unknown"
        if modified_files:
            for file_path in self._sdk_files:
                if Path(file_path).name == first_modified:
                    actual_checksum = self._calculate_file_checksum(file_path) or "unknown"
                    break
        
        return TamperDetectionEvent(
            event_type="tamper_detected",
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            client_id=self.client_id,
            sdk_version=self.sdk_version,
            checksum_expected=expected_checksum,
            checksum_actual=actual_checksum,
            modified_files=modified_files
        )
    
    def create_unclosed_sessions_metric(self, agent_id: str, 
                                       unclosed_sessions: List[Dict[str, Any]]) -> SecurityMetricEvent:
        """Create an unclosed sessions security metric event"""
        return SecurityMetricEvent(
            event_type="unclosed_sessions",
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            client_id=self.client_id,
            unclosed_count=len(unclosed_sessions),
            unclosed_sessions=unclosed_sessions
        )
    
    def get_security_flags(self) -> SecurityFlags:
        """Get current security flags for session events"""
        with self._security_lock:
            return SecurityFlags(
                tamper_detected=self._tamper_detected,
                pii_detected=False,  # Future implementation
                compliance_violation=False  # Future implementation
            )
    
    def is_tamper_detected(self) -> bool:
        """Check if tamper has been detected"""
        with self._security_lock:
            return self._tamper_detected
    
    def recheck_tampering(self) -> List[str]:
        """Manually recheck for tampering and return list of modified files"""
        modified_files = self._check_tampering()
        if modified_files:
            with self._security_lock:
                self._tamper_detected = True
        return modified_files
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security status summary"""
        with self._security_lock:
            return {
                "client_id": self.client_id,
                "sdk_version": self.sdk_version,
                "tamper_detected": self._tamper_detected,
                "monitored_files": len(self._sdk_files),
                "files_with_checksums": len(self._expected_checksums),
                "last_check_time": datetime.now().isoformat(),
                "security_flags": self.get_security_flags().to_dict()
            }

class SecurityAPIClient:
    """Handles security-specific API communications"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize Security API Client"""
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        self._session: Optional[requests.Session] = None
        self._async_session: Optional[aiohttp.ClientSession] = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with API key"""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> SecurityAPIResponse:
        """Make HTTP request to security endpoint"""
        try:
            if not self._session:
                self._session = requests.Session()
            
            url = f"{self.base_url}{endpoint}"
            headers = self._get_headers()
            
            response = self._session.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            return SecurityAPIResponse(
                success=response.status_code == 200,
                status_code=response.status_code,
                data=response.json() if response.content else None,
                error=None if response.status_code == 200 else f"HTTP {response.status_code}"
            )
            
        except Exception as e:
            self.logger.error(f"Security API request failed: {e}")
            return SecurityAPIResponse(
                success=False,
                status_code=0,
                error=str(e)
            )
    
    async def _make_request_async(self, method: str, endpoint: str, 
                                data: Optional[Dict] = None) -> SecurityAPIResponse:
        """Make async HTTP request to security endpoint"""
        try:
            if not self._async_session:
                self._async_session = aiohttp.ClientSession()
            
            url = f"{self.base_url}{endpoint}"
            headers = self._get_headers()
            
            async with self._async_session.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                response_data = None
                if response.content_length and response.content_length > 0:
                    response_data = await response.json()
                
                return SecurityAPIResponse(
                    success=response.status == 200,
                    status_code=response.status,
                    data=response_data,
                    error=None if response.status == 200 else f"HTTP {response.status}"
                )
                
        except Exception as e:
            self.logger.error(f"Security API async request failed: {e}")
            return SecurityAPIResponse(
                success=False,
                status_code=0,
                error=str(e)
            )
    
    def send_tamper_detection(self, event: TamperDetectionEvent) -> SecurityAPIResponse:
        """Send tamper detection event to backend"""
        return self._make_request("POST", "/security/tamper", asdict(event))
    
    def send_unclosed_sessions_metric(self, event: SecurityMetricEvent) -> SecurityAPIResponse:
        """Send unclosed sessions metric to backend"""
        return self._make_request("POST", "/security/metrics", asdict(event))
    
    async def send_tamper_detection_async(self, event: TamperDetectionEvent) -> SecurityAPIResponse:
        """Send tamper detection event to backend (async)"""
        return await self._make_request_async("POST", "/security/tamper", asdict(event))
    
    async def send_unclosed_sessions_metric_async(self, event: SecurityMetricEvent) -> SecurityAPIResponse:
        """Send unclosed sessions metric to backend (async)"""
        return await self._make_request_async("POST", "/security/metrics", asdict(event))
    
    def close(self):
        """Close synchronous session"""
        if self._session:
            self._session.close()
            self._session = None
    
    async def close_async(self):
        """Close asynchronous session"""
        if self._async_session:
            await self._async_session.close()
            self._async_session = None

class SecurityWrapper:
    """
    Security wrapper that adds security features to existing trackers
    
    This wrapper intercepts method calls to add security functionality
    without modifying the original tracker classes.
    """
    
    def __init__(self, tracker, enable_security: bool = True, 
                 client_id: Optional[str] = None,
                 security_check_interval: int = 300,  # 5 minutes
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Security Wrapper
        
        Args:
            tracker: The base tracker to wrap (AgentPerformanceTracker or AgentOperationsTracker)
            enable_security: Whether to enable security features
            client_id: Unique client identifier for security tracking
            security_check_interval: Interval for periodic security checks (seconds)
            logger: Optional logger instance
        """
        self.tracker = tracker
        self.enable_security = enable_security
        self.logger = logger or logging.getLogger(__name__)
        
        # Generate client ID if not provided
        self.client_id = client_id or f"security_client_{uuid.uuid4().hex[:12]}"
        
        # Initialize security components if enabled
        if self.enable_security:
            try:
                # Get base URL and API key from wrapped tracker
                base_url = getattr(tracker, 'base_url', 'https://api.example.com')
                api_key = getattr(tracker.api_client, '_api_key', None) if hasattr(tracker, 'api_client') else None
                
                self.security_manager = SecurityManager(
                    client_id=self.client_id,
                    sdk_version="1.2.1",
                    logger=self.logger
                )
                
                self.security_api = SecurityAPIClient(
                    base_url=base_url,
                    api_key=api_key,
                    logger=self.logger
                )
                
                # Track unclosed sessions
                self._unclosed_sessions: Dict[str, Dict[str, Any]] = {}
                self._sessions_lock = threading.RLock()
                
                # Offline event queue
                self._offline_queue = deque()
                self._queue_lock = threading.RLock()
                self._backend_available = True
                
                # Start periodic security checks
                self._start_security_daemon(security_check_interval)
                
                # Send initial tamper detection if needed
                if self.security_manager.is_tamper_detected():
                    self._handle_tamper_detection("system_init")
                
                self.logger.info(f"Security wrapper enabled for client: {self.client_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize security features: {e}")
                self.security_manager = None
                self.security_api = None
                self.enable_security = False
        else:
            self.security_manager = None
            self.security_api = None
            self.logger.info("Security wrapper disabled")
        
        # Initialize daemon control
        self._daemon_stop_event = threading.Event()
        self._daemon_thread: Optional[threading.Thread] = None
    
    def _start_security_daemon(self, interval: int):
        """Start the security monitoring daemon"""
        if not self.enable_security:
            return
            
        self._daemon_thread = threading.Thread(
            target=self._security_daemon,
            args=(interval,),
            daemon=True
        )
        self._daemon_thread.start()
        self.logger.info("Security monitoring daemon started")
    
    def _security_daemon(self, interval: int):
        """Security monitoring daemon that runs periodic checks"""
        while not self._daemon_stop_event.is_set():
            try:
                # Check for tampering
                if self.security_manager:
                    modified_files = self.security_manager.recheck_tampering()
                    if modified_files:
                        self._handle_tamper_detection("periodic_check")
                
                # Send unclosed sessions metric
                self._send_unclosed_sessions_metric()
                
                # Process offline queue
                self._process_offline_queue()
                
                # Wait for next cycle
                if self._daemon_stop_event.wait(timeout=interval):
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in security daemon: {e}")
                if self._daemon_stop_event.wait(timeout=60):  # Wait 1 minute on error
                    break
    
    def _handle_tamper_detection(self, agent_id: str):
        """Handle tamper detection by sending alert to backend"""
        if not self.security_manager or not self.security_api:
            return
        
        try:
            event = self.security_manager.create_tamper_detection_event(agent_id)
            
            response = self.security_api.send_tamper_detection(event)
            if response.success:
                self.logger.critical(f"SECURITY ALERT: Tamper detection sent for {len(event.modified_files)} files")
            else:
                self.logger.error(f"Failed to send tamper detection: {response.error}")
                self._queue_offline_event('tamper_detected', asdict(event))
        
        except Exception as e:
            self.logger.error(f"Error handling tamper detection: {e}")
    
    def _send_unclosed_sessions_metric(self):
        """Send unclosed sessions security metric"""
        if not self.security_manager or not self.security_api:
            return
        
        try:
            with self._sessions_lock:
                if not self._unclosed_sessions:
                    return  # No unclosed sessions to report
                
                unclosed_list = []
                for session_id, session_data in self._unclosed_sessions.items():
                    start_time = datetime.fromisoformat(session_data['start_time'])
                    duration_hours = (datetime.now() - start_time).total_seconds() / 3600
                    
                    unclosed_list.append({
                        "session_id": session_id,
                        "start_time": session_data['start_time'],
                        "agent_id": session_data['agent_id'],
                        "duration_hours": round(duration_hours, 2)
                    })
                
                if unclosed_list:
                    event = self.security_manager.create_unclosed_sessions_metric(
                        agent_id="security_daemon",
                        unclosed_sessions=unclosed_list
                    )
                    
                    response = self.security_api.send_unclosed_sessions_metric(event)
                    if response.success:
                        self.logger.warning(f"SECURITY METRIC: {len(unclosed_list)} unclosed sessions reported")
                    else:
                        self.logger.error(f"Failed to send unclosed sessions metric: {response.error}")
                        self._queue_offline_event('unclosed_sessions', asdict(event))
        
        except Exception as e:
            self.logger.error(f"Error sending unclosed sessions metric: {e}")
    
    def _queue_offline_event(self, event_type: str, event_data: Dict[str, Any]):
        """Queue event for later transmission when backend is available"""
        with self._queue_lock:
            self._offline_queue.append({
                'type': event_type,
                'data': event_data,
                'timestamp': datetime.now().isoformat(),
                'retry_count': 0
            })
            self._backend_available = False
            self.logger.info(f"Queued {event_type} event for offline transmission")
    
    def _process_offline_queue(self):
        """Process queued events when backend becomes available"""
        if not self._offline_queue:
            return
        
        with self._queue_lock:
            while self._offline_queue:
                event = self._offline_queue.popleft()
                
                try:
                    if event['type'] == 'tamper_detected':
                        response = self.security_api.send_tamper_detection(
                            TamperDetectionEvent(**event['data'])
                        )
                    elif event['type'] == 'unclosed_sessions':
                        response = self.security_api.send_unclosed_sessions_metric(
                            SecurityMetricEvent(**event['data'])
                        )
                    else:
                        continue
                    
                    if response.success:
                        self.logger.info(f"Replayed {event['type']} event successfully")
                        self._backend_available = True
                    else:
                        # Requeue with retry limit
                        event['retry_count'] += 1
                        if event['retry_count'] < 3:
                            self._offline_queue.appendleft(event)
                        else:
                            self.logger.error(f"Dropping {event['type']} event after 3 retries")
                        break
                
                except Exception as e:
                    self.logger.error(f"Error processing offline event: {e}")
                    break
    
    def _inject_security_flags(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Inject security flags into data payload"""
        if self.enable_security and self.security_manager:
            security_flags = self.security_manager.get_security_flags()
            data["security_flags"] = security_flags.to_dict()
        return data
    
    def _track_session_start(self, session_id: str, agent_id: str):
        """Track session start for unclosed sessions monitoring"""
        if not self.enable_security:
            return
        
        with self._sessions_lock:
            self._unclosed_sessions[session_id] = {
                'agent_id': agent_id,
                'start_time': datetime.now().isoformat(),
                'status': 'active'
            }
    
    def _track_session_end(self, session_id: str):
        """Track session end - remove from unclosed sessions"""
        if not self.enable_security:
            return
        
        with self._sessions_lock:
            if session_id in self._unclosed_sessions:
                del self._unclosed_sessions[session_id]
    
    # ============ WRAPPER METHODS FOR AgentPerformanceTracker ============
    
    def start_conversation(self, agent_id: str, user_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start conversation with security features"""
        # Call original method with correct signature
        session_id = self.tracker.start_conversation(agent_id, user_id, metadata)
        
        # Add security tracking
        if session_id and self.enable_security:
            self._track_session_start(session_id, agent_id)
        
        return session_id
    
    def end_conversation(self, session_id: str, quality_score = None,
                        user_feedback: Optional[str] = None,
                        message_count: Optional[int] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """End conversation with security features"""
        # Call original method
        result = self.tracker.end_conversation(session_id, quality_score, user_feedback, message_count, metadata)
        
        # Add security tracking
        if result and self.enable_security:
            self._track_session_end(session_id)
        
        return result
    
    def record_failed_session(self, session_id: str, error_message: str,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Record failed session with security features"""
        # Call original method
        result = self.tracker.record_failed_session(session_id, error_message, metadata)
        
        # Add security tracking
        if result and self.enable_security:
            self._track_session_end(session_id)
        
        return result
    
    # ============ ASYNC WRAPPER METHODS ============
    
    async def start_conversation_async(self, agent_id: str, user_id: Optional[str] = None,
                                     metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start conversation with security features (async)"""
        session_id = await self.tracker.start_conversation_async(agent_id, user_id, metadata)
        
        if session_id and self.enable_security:
            self._track_session_start(session_id, agent_id)
        
        return session_id
    
    async def end_conversation_async(self, session_id: str, quality_score = None,
                                   user_feedback: Optional[str] = None,
                                   message_count: Optional[int] = None,
                                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """End conversation with security features (async)"""
        result = await self.tracker.end_conversation_async(session_id, quality_score, user_feedback, message_count, metadata)
        
        if result and self.enable_security:
            self._track_session_end(session_id)
        
        return result
    
    # ============ PASSTHROUGH METHODS ============
    
    def __getattr__(self, name):
        """Pass through any other method calls to the wrapped tracker"""
        # Avoid infinite recursion by checking if attribute exists in wrapper first
        if name.startswith('_daemon_') or name in ['_sessions_lock', '_offline_queue', '_queue_lock']:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        return getattr(self.tracker, name)
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get comprehensive security statistics"""
        if not self.enable_security:
            return {"security_enabled": False}
        
        with self._sessions_lock:
            unclosed_count = len(self._unclosed_sessions)
        
        with self._queue_lock:
            queue_size = len(self._offline_queue)
        
        base_stats = {
            "security_enabled": True,
            "client_id": self.client_id,
            "unclosed_sessions_count": unclosed_count,
            "offline_queue_size": queue_size,
            "backend_available": self._backend_available,
            "daemon_running": self._daemon_thread.is_alive() if self._daemon_thread else False
        }
        
        if self.security_manager:
            base_stats.update(self.security_manager.get_security_summary())
        
        return base_stats
    
    def close(self):
        """Close security wrapper and underlying tracker"""
        # Stop security daemon
        if self._daemon_thread and self._daemon_thread.is_alive():
            self._daemon_stop_event.set()
            self._daemon_thread.join(timeout=5)
        
        # Close security API client
        if self.security_api:
            self.security_api.close()
        
        # Close underlying tracker
        if hasattr(self.tracker, 'close'):
            self.tracker.close()
        
        self.logger.info("Security wrapper closed")
    
    async def close_async(self):
        """Close security wrapper and underlying tracker (async)"""
        # Stop security daemon
        if self._daemon_thread and self._daemon_thread.is_alive():
            self._daemon_stop_event.set()
            self._daemon_thread.join(timeout=5)
        
        # Close security API client
        if self.security_api:
            await self.security_api.close_async()
        
        # Close underlying tracker
        if hasattr(self.tracker, 'close_async'):
            await self.tracker.close_async()
        elif hasattr(self.tracker, 'close'):
            self.tracker.close()
        
        self.logger.info("Security wrapper closed (async)")

# ============ FACTORY FUNCTIONS ============

def create_secure_performance_tracker(base_url: str, api_key: Optional[str] = None,
                                     enable_security: bool = True,
                                     client_id: Optional[str] = None,
                                     **tracker_kwargs) -> SecurityWrapper:
    """
    Factory function to create a secure AgentPerformanceTracker
    
    Args:
        base_url: API base URL
        api_key: API authentication key
        enable_security: Whether to enable security features
        client_id: Unique client identifier for security tracking
        **tracker_kwargs: Additional arguments for AgentPerformanceTracker
    
    Returns:
        SecurityWrapper wrapping AgentPerformanceTracker
    """
    from .AgentPerform import AgentPerformanceTracker
    
    base_tracker = AgentPerformanceTracker(base_url, api_key, **tracker_kwargs)
    return SecurityWrapper(base_tracker, enable_security, client_id)

def create_secure_operations_tracker(base_url: str, api_key: Optional[str] = None,
                                    enable_security: bool = True,
                                    client_id: Optional[str] = None,
                                    **tracker_kwargs) -> SecurityWrapper:
    """
    Factory function to create a secure AgentOperationsTracker
    
    Args:
        base_url: API base URL
        api_key: API authentication key
        enable_security: Whether to enable security features
        client_id: Unique client identifier for security tracking
        **tracker_kwargs: Additional arguments for AgentOperationsTracker
    
    Returns:
        SecurityWrapper wrapping AgentOperationsTracker
    """
    from .AgentOper import AgentOperationsTracker
    
    base_tracker = AgentOperationsTracker(base_url, api_key, **tracker_kwargs)
    return SecurityWrapper(base_tracker, enable_security, client_id)
