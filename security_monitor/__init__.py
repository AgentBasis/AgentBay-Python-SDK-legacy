"""
Security Monitor - Advanced Agent Security Monitoring System

A comprehensive security monitoring solution for AI agents providing:

Context-Aware Security Monitoring:
- Conversation context tracking and analysis
- Prompt injection and jailbreak detection
- Social engineering pattern recognition
- Manipulation attempt identification

Compliance and Governance:
- Regulatory compliance tracking
- Data retention and privacy controls
- Audit trail generation and reporting
- Risk assessment and scoring

Features:
- Non-intrusive passive monitoring
- Real-time security event detection
- Configurable alert thresholds
- Integration with existing telemetry systems
- Privacy-first design with configurable data collection

Quick Start:
    from MYSDK.security_monitor import SecurityMonitor
    
    # Initialize security monitoring
    monitor = SecurityMonitor()
    monitor.start_monitoring()
    
    # Monitor a conversation
    monitor.track_conversation(
        agent_id="agent_1",
        user_input="Hello, can you help me?",
        agent_response="Of course! How can I assist you?"
    )
"""

from .core import SecurityMonitor, SecurityEvent, SecurityMetrics
from .context_monitor import ContextAwareMonitor, ConversationContext
from .compliance import ComplianceMonitor, ComplianceReport
from .detectors import (
    PromptInjectionDetector,
    JailbreakDetector, 
    SocialEngineeringDetector,
    ManipulationDetector,
    PIIDetector,
    create_default_detectors,
)
from .analyzers import (
    ContentAnalyzer,
    RiskAssessment,
    BehaviorAnalyzer
)
from .reporting import SecurityReporter, AuditLogger
from .telemetry import SecurityTelemetryEmitter, create_logging_telemetry_handler

# Version info
__version__ = "1.0.0"
__author__ = "AgentBay Security Team"
__description__ = "Advanced Agent Security Monitoring System"

# Export all public classes and functions
__all__ = [
    # Core monitoring
    "SecurityMonitor",
    "SecurityEvent", 
    "SecurityMetrics",
    
    # Context-aware monitoring
    "ContextAwareMonitor",
    "ConversationContext",
    
    # Compliance and governance
    "ComplianceMonitor",
    "ComplianceReport",
    
    # Security detectors
    "PromptInjectionDetector",
    "JailbreakDetector",
    "SocialEngineeringDetector", 
    "ManipulationDetector",
    "PIIDetector",
    "create_default_detectors",
    
    # Analyzers
    "ContentAnalyzer",
    "RiskAssessment",
    "BehaviorAnalyzer",
    
    # Reporting
    "SecurityReporter",
    "AuditLogger",

    # Telemetry
    "SecurityTelemetryEmitter",
    "create_logging_telemetry_handler",
    
    # Quick setup
    "quick_setup_security",
    "setup_security_monitoring",
]


def quick_setup_security(
    enable_context_monitoring: bool = True,
    enable_compliance_tracking: bool = True,
    risk_threshold: float = 0.7,
    alert_webhook: str = None,
    privacy_mode: bool = False,
    **kwargs
):
    """
    Quick setup for security monitoring with default configuration.
    
    Args:
        enable_context_monitoring: Enable conversation context tracking
        enable_compliance_tracking: Enable compliance and governance features
        risk_threshold: Risk score threshold for alerts (0.0-1.0)
        alert_webhook: Webhook URL for security alerts
        privacy_mode: Enable privacy mode for minimal data collection
        **kwargs: Additional configuration options
            - detectors: Optional iterable of pre-configured detectors
            - telemetry_handler: Optional callable for detector telemetry
            - monitor_config/context_config/compliance_config: nested settings
    """
    print("🔒 Initializing Security Monitor...")
    
    # Initialize core security monitor
    monitor_config = dict(kwargs.get('monitor_config', {}))
    telemetry_handler = kwargs.get('telemetry_handler')
    if telemetry_handler is None:
        telemetry_handler = monitor_config.pop('telemetry_handler', None)
    
    detectors = kwargs.get('detectors') or monitor_config.pop('detectors', None)
    if detectors is None:
        detectors = create_default_detectors(telemetry_handler=telemetry_handler)
    
    monitor = SecurityMonitor(
        risk_threshold=risk_threshold,
        alert_webhook=alert_webhook,
        privacy_mode=privacy_mode,
        detectors=detectors,
        **monitor_config
    )
    
    # Setup context monitoring
    if enable_context_monitoring:
        print("🧠 Enabling context-aware monitoring...")
        context_monitor = ContextAwareMonitor(
            **kwargs.get('context_config', {})
        )
        monitor.add_component(context_monitor)
    
    # Setup compliance monitoring
    if enable_compliance_tracking:
        print("📋 Enabling compliance tracking...")
        compliance_monitor = ComplianceMonitor(
            **kwargs.get('compliance_config', {})
        )
        monitor.add_component(compliance_monitor)
    
    # Start monitoring
    monitor.start_monitoring()
    
    print("✅ Security monitoring initialized successfully!")
    print(f"   - Risk threshold: {risk_threshold}")
    print(f"   - Context monitoring: {'enabled' if enable_context_monitoring else 'disabled'}")
    print(f"   - Compliance tracking: {'enabled' if enable_compliance_tracking else 'disabled'}")
    print(f"   - Privacy mode: {'enabled' if privacy_mode else 'disabled'}")
    
    return monitor


def setup_security_monitoring(config: dict = None):
    """
    Advanced setup for security monitoring with detailed configuration.
    
    Args:
        config: Configuration dictionary with detailed settings
    """
    config = config or {}
    
    # Extract configuration sections
    monitor_config = config.get('monitor', {})
    context_config = config.get('context', {})
    compliance_config = config.get('compliance', {})
    
    # Initialize security monitor
    monitor = SecurityMonitor(**monitor_config)
    
    # Setup context monitoring
    if context_config.get('enabled', True):
        context_monitor = ContextAwareMonitor(**context_config.get('settings', {}))
        monitor.add_component(context_monitor)
    
    # Setup compliance monitoring
    if compliance_config.get('enabled', True):
        compliance_monitor = ComplianceMonitor(**compliance_config.get('settings', {}))
        monitor.add_component(compliance_monitor)
    
    # Start monitoring if configured
    if config.get('auto_start', True):
        monitor.start_monitoring()
    
    return monitor


# Convenience functions
def create_security_monitor(**kwargs):
    """Create a SecurityMonitor instance with given configuration."""
    return SecurityMonitor(**kwargs)

def start_security_monitoring(**kwargs):
    """Start security monitoring with quick setup."""
    return quick_setup_security(**kwargs)



