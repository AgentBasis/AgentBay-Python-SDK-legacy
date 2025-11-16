"""
AgentBay SDK - Complete AI Agent Monitoring Solution

A comprehensive SDK for monitoring AI agents and system components, providing:

LLM Tracking:
- OpenAI, Anthropic, Gemini, Grok, Watson X instrumentation
- Conversation tracking with privacy controls
- Token usage and streaming metrics
- Tool call monitoring

System Component Tracking:
- CPU, memory, disk, network monitoring
- Process resource tracking
- System health indicators
- Real-time metrics collection

Features:
- OpenTelemetry-based telemetry
- Privacy-first design with configurable data collection
- Seamless integration between LLM and system tracking
- Background monitoring with configurable intervals
- Compatible with existing observability stacks

Quick Start:
    from agentbay import quick_setup
    
    # Initialize both LLM and system tracking
    quick_setup(
        llm_providers=["openai", "anthropic"],
        system_monitoring=True,
        collection_interval=30
    )
"""

# LLM Tracking
from .llm_tracking import (
    instrument_openai,
    uninstrument_openai,
    instrument_anthropic, 
    uninstrument_anthropic,
    instrument_gemini,
    uninstrument_gemini,
    instrument_watsonx,
    uninstrument_watsonx,
    instrument_grok,
    uninstrument_grok,
    configure_privacy as configure_llm_privacy,
)

# System Component Tracking
from .comp_tracking import (
    instrument_system,
    uninstrument_system,
    start_system_monitoring,
    stop_system_monitoring,
    configure as configure_system,
    trace_system_operation,
    record_system_event,
    get_system_info,
    check_system_health,
    get_system_summary,
    record_system_snapshot,
    quick_start as system_quick_start,
)

# OpenTelemetry initialization
from .llm_tracking.otel import init_tracing

# Agentic Framework Instrumentation
from .bay_frameworks.instrumentation import (
    instrument_all,
    uninstrument_all,
    get_active_libraries,
)

# Security Monitoring
from .security_monitor import (
    SecurityMonitor,
    quick_setup_security,
    setup_security_monitoring,
)

# Version info (imported from _version.py - single source of truth)
from ._version import __version__
__author__ = "AgentBay Team"
__description__ = "Complete AI Agent and System Monitoring SDK"

# Export all public functions
__all__ = [
    # LLM tracking
    "instrument_openai",
    "uninstrument_openai", 
    "instrument_anthropic",
    "uninstrument_anthropic",
    "instrument_gemini",
    "uninstrument_gemini",
    "instrument_watsonx", 
    "uninstrument_watsonx",
    "instrument_grok",
    "uninstrument_grok",
    "configure_llm_privacy",
    
    # System tracking
    "instrument_system",
    "uninstrument_system",
    "start_system_monitoring",
    "stop_system_monitoring", 
    "configure_system",
    "trace_system_operation",
    "record_system_event",
    "get_system_info",
    "check_system_health",
    "get_system_summary",
    "record_system_snapshot",
    "system_quick_start",
    
    # Setup utilities
    "init_tracing",
    "quick_setup",
    "setup_agentbay",
    
    # Agentic framework instrumentation
    "instrument_all",
    "uninstrument_all",
    "get_active_libraries",
    
    # Security monitoring
    "SecurityMonitor",
    "quick_setup_security",
    "setup_security_monitoring",
]


def quick_setup(
    llm_providers: list = None,
    system_monitoring: bool = True,
    collection_interval: float = 30.0,
    privacy_mode: bool = False,
    enable_network_monitoring: bool = False,
    enable_framework_instrumentation: bool = False,
    enable_security_monitoring: bool = False,
    exporter: str = "otlp-http",
    endpoint: str = None,
    api_key: str = None,
    **kwargs
):
    """
    Quick setup for AgentBay SDK with LLM, system, framework, and security tracking.
    
    Args:
        llm_providers: List of LLM providers to instrument (e.g., ["openai", "anthropic"])
        system_monitoring: Enable system component monitoring
        collection_interval: System monitoring collection interval in seconds
        privacy_mode: Enable privacy mode for minimal data collection
        enable_network_monitoring: Enable network monitoring (disabled by default)
        enable_framework_instrumentation: Enable auto-instrumentation for agentic frameworks
        enable_security_monitoring: Enable security monitoring (prompt injection, jailbreak detection)
        exporter: OpenTelemetry exporter type ("console" or "otlp-http", default: "otlp-http")
        endpoint: OTLP HTTP endpoint (defaults to AgentBay backend or AGENTBAY_ENDPOINT env var)
        api_key: API key for authentication (defaults to AGENTBAY_API_KEY env var)
        **kwargs: Additional configuration options
    """
    import os
    
    print("Initializing AgentBay SDK...")
    
    # Get API key from parameter or environment variable
    resolved_api_key = api_key or os.getenv("AGENTBAY_API_KEY")
    
    # Validate API key is present (required for otlp-http exporter)
    if exporter == "otlp-http" and not resolved_api_key:
        raise ValueError(
            "AGENTBAY_API_KEY is required. "
            "Please set it as an environment variable: export AGENTBAY_API_KEY=your-key-here "
            "or pass it as a parameter: quick_setup(api_key='your-key-here')"
        )
    
    # Initialize OpenTelemetry
    init_tracing(exporter=exporter, endpoint=endpoint, api_key=resolved_api_key)
    
    # Setup LLM tracking
    if llm_providers:
        print("Instrumenting LLM providers...")
        
        # Configure LLM privacy
        configure_llm_privacy(
            capture_content=not privacy_mode,
            **kwargs.get('llm_config', {})
        )
        
        # Instrument requested providers
        for provider in llm_providers:
            if provider.lower() == "openai":
                instrument_openai()
            elif provider.lower() == "anthropic":
                instrument_anthropic()
            elif provider.lower() == "gemini":
                instrument_gemini()
            elif provider.lower() == "watsonx":
                instrument_watsonx()
            elif provider.lower() == "grok":
                instrument_grok()
            else:
                print(f"Unknown LLM provider: {provider}")
        
        print(f"LLM tracking enabled for: {', '.join(llm_providers)}")
    
    # Setup system monitoring
    if system_monitoring:
        print("Instrumenting system monitoring...")
        
        # Configure system tracking
        configure_system(
            privacy_mode=privacy_mode,
            enable_network_monitoring=enable_network_monitoring,
            default_collection_interval=collection_interval,
            **kwargs.get('system_config', {})
        )
        
        # Start system monitoring
        instrument_system(service_name=kwargs.get('service_name', 'agentbay'))
        start_system_monitoring(collection_interval)
        
        print(f"System monitoring enabled (interval: {collection_interval}s)")
    
    # Setup framework instrumentation
    if enable_framework_instrumentation:
        print("Enabling agentic framework instrumentation...")
        instrument_all()
        print("Framework instrumentation enabled (auto-detects: langgraph, crewai, ag2, agno, smolagents, etc.)")
    
    # Setup security monitoring
    security_monitor_instance = None
    if enable_security_monitoring:
        print("Enabling security monitoring...")
        security_monitor_instance = quick_setup_security(
            enable_context_monitoring=kwargs.get('enable_context_monitoring', True),
            enable_compliance_tracking=kwargs.get('enable_compliance_tracking', True),
            risk_threshold=kwargs.get('risk_threshold', 0.7),
            alert_webhook=kwargs.get('alert_webhook'),
            privacy_mode=privacy_mode,
            **kwargs.get('security_config', {})
        )
        print("Security monitoring enabled (prompt injection, jailbreak detection, compliance tracking)")
    
    print("AgentBay SDK initialization complete!")
    
    return {
        "llm_providers": llm_providers or [],
        "system_monitoring": system_monitoring,
        "collection_interval": collection_interval,
        "privacy_mode": privacy_mode,
        "framework_instrumentation": enable_framework_instrumentation,
        "security_monitoring": enable_security_monitoring,
        "security_monitor": security_monitor_instance,
    }


def setup_agentbay(config: dict = None):
    """
    Advanced setup for AgentBay SDK with detailed configuration.
    
    Args:
        config: Configuration dictionary with detailed settings for LLM, system, and telemetry
    """
    config = config or {}
    
    # Extract configuration sections
    llm_config = config.get('llm', {})
    system_config = config.get('system', {})
    telemetry_config = config.get('telemetry', {})
    
    # Initialize telemetry
    telemetry_options = telemetry_config.get('options', {})
    init_tracing(
        exporter=telemetry_config.get('exporter', 'otlp-http'),
        endpoint=telemetry_options.get('endpoint'),
        api_key=telemetry_options.get('api_key'),
    )
    
    # Setup LLM tracking
    if llm_config.get('enabled', True):
        configure_llm_privacy(**llm_config.get('privacy', {}))
        
        for provider, enabled in llm_config.get('providers', {}).items():
            if enabled:
                globals()[f'instrument_{provider}']()
    
    # Setup system tracking  
    if system_config.get('enabled', True):
        configure_system(**system_config.get('settings', {}))
        instrument_system(service_name=config.get('service_name', 'agentbay'))
        
        if system_config.get('start_monitoring', True):
            start_system_monitoring(
                system_config.get('collection_interval', 30.0)
            )
    
    return config


# Convenience aliases
monitor_llm = lambda *providers: [globals()[f'instrument_{p}']() for p in providers]
monitor_system = start_system_monitoring
stop_monitoring = stop_system_monitoring

# Legacy compatibility
agentbay_init = quick_setup
