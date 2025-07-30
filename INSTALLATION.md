# Installation Guide

## Standard Installation (Recommended)

Install with all features including OpenTelemetry tracing:

```bash
pip install -r requirements.txt
```

Or install with specific extras:

```bash
pip install -e .[tracing]  # Include OpenTelemetry tracing
pip install -e .[dev]      # Include development tools
pip install -e .[tracing,dev]  # Include both
```

## Minimal Installation

If you don't need OpenTelemetry tracing features:

```bash
pip install -r requirements-minimal.txt
```

Or install only core dependencies:

```bash
pip install -e .  # Core functionality only
```

## Dependency Overview

### Core Dependencies (Always Required)
- `requests>=2.28.0` - HTTP client for API calls
- `aiohttp>=3.8.0` - Async HTTP client
- `python-dateutil>=2.8.0` - Date/time utilities

### Optional Dependencies

#### OpenTelemetry Tracing (`[tracing]` extra)
- `opentelemetry-api>=1.36.0` - Core tracing API
- `opentelemetry-sdk>=1.36.0` - SDK implementation
- `opentelemetry-exporter-otlp>=1.36.0` - OTLP exporter

**Benefits:**
- ✅ Full observability and distributed tracing
- ✅ Performance monitoring and debugging
- ✅ Integration with observability platforms
- ✅ Eliminates import warnings in IDEs

**Note:** The security module works perfectly without OpenTelemetry - it uses mock classes for graceful fallback.

#### Development Tools (`[dev]` extra)
- `pytest>=7.0.0` - Testing framework
- `pytest-asyncio>=0.20.0` - Async testing support  
- `black>=22.0.0` - Code formatting
- `mypy>=1.0.0` - Type checking

## IDE Configuration

If you see import warnings for OpenTelemetry in your IDE:

1. **Option 1 (Recommended):** Install with tracing support:
   ```bash
   pip install -e .[tracing]
   ```

2. **Option 2:** Configure your IDE to ignore these specific import warnings (the code works fine without them)

3. **Option 3:** Use the minimal installation and accept that tracing features will use mock implementations

## Verification

Test your installation:

```python
from tracker.security import detect_pii, OTEL_AVAILABLE

print(f"OpenTelemetry available: {OTEL_AVAILABLE}")
print(f"PII detection works: {detect_pii('test@example.com')}")
```

Both should work regardless of whether OpenTelemetry is installed! 