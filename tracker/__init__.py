"""
Agent Operations Tracking SDK

A library for tracking AI Agent operations and metrics.
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

__version__ = "1.0.0"
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