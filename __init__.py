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

# Version info
__version__ = "1.0.0"
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
    
    # Setup utilities
    "init_tracing",
    "quick_setup",
    "setup_agentbay",
    
    # Agentic framework instrumentation
    "instrument_all",
    "uninstrument_all",
    "get_active_libraries",
]


def quick_setup(
    llm_providers: list = None,
    system_monitoring: bool = True,
    collection_interval: float = 30.0,
    privacy_mode: bool = False,
    enable_network_monitoring: bool = False,
    enable_framework_instrumentation: bool = False,
    exporter: str = "console",
    **kwargs
):
    """
    Quick setup for AgentBay SDK with LLM, system, and framework tracking.
    
    Args:
        llm_providers: List of LLM providers to instrument (e.g., ["openai", "anthropic"])
        system_monitoring: Enable system component monitoring
        collection_interval: System monitoring collection interval in seconds
        privacy_mode: Enable privacy mode for minimal data collection
        enable_network_monitoring: Enable network monitoring (disabled by default)
        enable_framework_instrumentation: Enable auto-instrumentation for agentic frameworks (langgraph, crewai, etc.)
        exporter: OpenTelemetry exporter type ("console", "otlp", etc.)
        **kwargs: Additional configuration options
    """
    print("Initializing AgentBay SDK...")
    
    # Initialize OpenTelemetry
    init_tracing(exporter=exporter)
    
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
    
    print("AgentBay SDK initialization complete!")
    
    return {
        "llm_providers": llm_providers or [],
        "system_monitoring": system_monitoring,
        "collection_interval": collection_interval,
        "privacy_mode": privacy_mode,
        "framework_instrumentation": enable_framework_instrumentation,
    }


def setup_agentbay(config: dict = None):
    """
    Advanced setup for AgentBay SDK with detailed configuration.
    
    Args:
        config: Configuration dictionary with detailed settings
    """
    config = config or {}
    
    # Extract configuration sections
    llm_config = config.get('llm', {})
    system_config = config.get('system', {})
    telemetry_config = config.get('telemetry', {})
    
    # Initialize telemetry
    init_tracing(
        exporter=telemetry_config.get('exporter', 'console'),
        **telemetry_config.get('options', {})
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
