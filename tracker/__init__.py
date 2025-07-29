"""
AI Agent Tracking SDK

This package provides comprehensive tracking capabilities for AI Agents:

Agent Operations Tracker:
- Agent registration and status management  
- Activity logging and monitoring
- Real-time agent operations tracking

Agent Performance Tracker:
- Conversation quality metrics
- Success rate calculations  
- Response time tracking
- Failed session management
- Sliding TTL session tracking with automatic cleanup

Key Features:
- Secure API communication
- Async/sync support
- Sliding TTL for active session management
- Thread-safe operations
- Comprehensive logging

Note: Session tracking uses sliding TTL - sessions are kept alive
as long as they are being accessed within the TTL window.
"""

from .AgentOper import (
    AgentOperationsTracker,
    AgentStatus,
    APIResponse,
    SecureLogger,
    SecureAPIClient,
    AgentRegistrationData,
    AgentStatusData,
    ActivityLogData
)

from .AgentPerform import (
    AgentPerformanceTracker,
    ConversationQuality,
    ConversationStartData,
    ConversationEndData,
    PerformanceMetricsQuery,
    SessionInfo
)

__version__ = "1.2.1"
__all__ = [
    # Agent Operations
    'AgentOperationsTracker',
    'AgentStatus',
    'AgentRegistrationData',
    'AgentStatusData',
    'ActivityLogData',
    
    # Agent Performance
    'AgentPerformanceTracker',
    'ConversationQuality',
    'ConversationStartData',
    'ConversationEndData',
    'PerformanceMetricsQuery',
    'SessionInfo',
    
    # Shared Components
    'APIResponse',
    'SecureLogger',
    'SecureAPIClient'
]

# Note: Session tracking is now handled entirely by the backend.
# The SDK no longer maintains local session state for lighter memory usage. 