"""
Integration example showing how to use both LLM tracking and system component tracking together.

This example demonstrates:
1. Setting up both LLM and system tracking
2. Configuring each tracker independently
3. Using them together in a real application
4. Ensuring no conflicts between the trackers
"""

import time
import asyncio
from typing import Dict, Any

# Import LLM tracking
from MYSDK.Tracker_llm import (
    instrument_openai, 
    instrument_anthropic,
    configure_privacy as configure_llm_privacy
)

# Import system component tracking
from MYSDK.comp_tracker import (
    configure as configure_system,
    instrument_system,
    start_system_monitoring,
    trace_system_operation,
    get_system_summary,
    check_system_health
)

# Import OpenTelemetry for custom setup
from MYSDK.Tracker_llm.otel import init_tracing


def setup_agentbay_tracking():
    """Setup both LLM and system tracking with AgentBay branding."""
    
    print("🚀 Initializing AgentBay SDK...")
    
    # 1. Initialize OpenTelemetry tracing (shared by both trackers)
    init_tracing(exporter="console")  # Use console exporter for demo
    
    # 2. Configure LLM tracking privacy settings
    configure_llm_privacy(
        capture_content=True,  # Set to False to disable content capture
        # redactor=lambda s: s.replace("secret", "[REDACTED]")  # Optional redaction
    )
    
    # 3. Configure system tracking
    configure_system(
        enable_cpu_monitoring=True,
        enable_memory_monitoring=True,
        enable_disk_monitoring=True,
        enable_network_monitoring=False,  # Disabled for privacy by default
        enable_process_monitoring=True,
        cpu_alert_threshold=75.0,
        memory_alert_threshold=80.0,
        disk_alert_threshold=90.0,
        default_collection_interval=30.0,  # Collect every 30 seconds
        privacy_mode=False,
    )
    
    # 4. Instrument LLM providers
    print("📡 Instrumenting LLM providers...")
    instrument_openai()
    instrument_anthropic()
    
    # 5. Instrument system monitoring
    print("⚙️  Instrumenting system monitoring...")
    instrument_system(service_name="agentbay-demo")
    start_system_monitoring(interval=30.0)
    
    print("✅ AgentBay SDK initialized successfully!")
    print("   - LLM tracking: OpenAI, Anthropic")
    print("   - System tracking: CPU, Memory, Disk, Process")
    print("   - Collection interval: 30 seconds")


def demonstrate_llm_tracking():
    """Demonstrate LLM tracking with system monitoring running in background."""
    
    print("\n🤖 Testing LLM tracking...")
    
    try:
        # OpenAI example
        from openai import OpenAI
        
        client = OpenAI(api_key="dummy-key")  # Will fail but show instrumentation
        
        with trace_system_operation("llm_request", provider="openai") as span:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello from AgentBay!"}]
                )
                span.set_attribute("request_successful", True)
            except Exception as e:
                span.set_attribute("request_successful", False)
                span.set_attribute("error", str(e))
                print(f"   OpenAI request failed (expected with dummy key): {e}")
                
    except ImportError:
        print("   OpenAI not installed, skipping OpenAI demo")
    
    try:
        # Anthropic example
        from anthropic import Anthropic
        
        client = Anthropic(api_key="dummy-key")  # Will fail but show instrumentation
        
        with trace_system_operation("llm_request", provider="anthropic") as span:
            try:
                response = client.messages.create(
                    model="claude-3-5-sonnet",
                    messages=[{"role": "user", "content": "Hello from AgentBay!"}]
                )
                span.set_attribute("request_successful", True)
            except Exception as e:
                span.set_attribute("request_successful", False)
                span.set_attribute("error", str(e))
                print(f"   Anthropic request failed (expected with dummy key): {e}")
                
    except ImportError:
        print("   Anthropic not installed, skipping Anthropic demo")


def demonstrate_system_tracking():
    """Demonstrate system tracking features."""
    
    print("\n📊 Testing system tracking...")
    
    # Get system summary
    summary = get_system_summary()
    print(f"   System Status: {summary.get('status', 'unknown')}")
    print(f"   CPU Usage: {summary.get('cpu_usage_percent', 'N/A')}%")
    print(f"   Memory Usage: {summary.get('memory_usage_percent', 'N/A')}%")
    print(f"   Available Memory: {summary.get('memory_available_gb', 'N/A'):.1f} GB")
    
    if 'disk_usage_percent' in summary:
        print(f"   Disk Usage: {summary['disk_usage_percent']:.1f}%")
    
    # Check system health
    health = check_system_health()
    print(f"   Health Status: {health['status']}")
    
    if health.get('alerts'):
        print("   🚨 Alerts:")
        for alert in health['alerts']:
            print(f"     - {alert['type'].upper()}: {alert['message']}")
    else:
        print("   ✅ No alerts")


def simulate_workload():
    """Simulate some workload to generate interesting metrics."""
    
    print("\n⚡ Simulating workload...")
    
    with trace_system_operation("data_processing", workload_type="simulation") as span:
        # Simulate CPU-intensive work
        start_time = time.time()
        result = sum(i * i for i in range(1000000))
        end_time = time.time()
        
        span.set_attribute("computation_result", result)
        span.set_attribute("duration_seconds", end_time - start_time)
        span.set_attribute("operations_count", 1000000)
        
        print(f"   Completed computation in {end_time - start_time:.3f} seconds")
        print(f"   Result: {result}")


async def async_workload_demo():
    """Demonstrate async operations with tracking."""
    
    print("\n🔄 Testing async operations...")
    
    async def async_task(task_id: int, duration: float):
        with trace_system_operation("async_task", task_id=task_id) as span:
            await asyncio.sleep(duration)
            span.set_attribute("duration_seconds", duration)
            return f"Task {task_id} completed"
    
    # Run multiple async tasks
    tasks = [
        async_task(1, 0.5),
        async_task(2, 1.0),
        async_task(3, 0.3),
    ]
    
    results = await asyncio.gather(*tasks)
    for result in results:
        print(f"   {result}")


def monitor_for_period(duration: int = 60):
    """Monitor system for a specific period and show periodic updates."""
    
    print(f"\n⏱️  Monitoring system for {duration} seconds...")
    print("   (Press Ctrl+C to stop early)")
    
    start_time = time.time()
    last_update = start_time
    
    try:
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # Update every 10 seconds
            if current_time - last_update >= 10:
                summary = get_system_summary()
                elapsed = int(current_time - start_time)
                
                print(f"   [{elapsed:2d}s] CPU: {summary.get('cpu_usage_percent', 'N/A'):>5}% | "
                      f"Memory: {summary.get('memory_usage_percent', 'N/A'):>5}% | "
                      f"Status: {summary.get('status', 'unknown')}")
                
                last_update = current_time
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n   Monitoring stopped by user")


def main():
    """Main demonstration function."""
    
    print("=" * 60)
    print("🌊 AgentBay SDK - Integrated LLM & System Tracking Demo")
    print("=" * 60)
    
    # Setup tracking
    setup_agentbay_tracking()
    
    # Wait a moment for initialization
    time.sleep(2)
    
    # Demonstrate LLM tracking
    demonstrate_llm_tracking()
    
    # Demonstrate system tracking
    demonstrate_system_tracking()
    
    # Simulate some workload
    simulate_workload()
    
    # Test async operations
    asyncio.run(async_workload_demo())
    
    # Monitor for a short period
    monitor_for_period(30)  # Monitor for 30 seconds
    
    print("\n🎯 Demo completed!")
    print("\nKey features demonstrated:")
    print("✅ LLM request instrumentation (OpenAI, Anthropic)")
    print("✅ System resource monitoring (CPU, Memory, Disk)")
    print("✅ Custom operation tracing")
    print("✅ System health monitoring")
    print("✅ Async operation support")
    print("✅ Real-time monitoring")
    
    print("\n📊 Check your telemetry backend for collected data!")


if __name__ == "__main__":
    main()
