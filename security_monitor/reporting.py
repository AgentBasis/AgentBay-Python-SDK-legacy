"""
Security Reporting and Audit Logging

Provides comprehensive reporting and audit logging capabilities:
- SecurityReporter: Generate security reports and dashboards
- AuditLogger: Immutable audit trail logging
- Report templates and export formats
- Real-time alerting and notifications
"""

import json
import csv
import time
import threading
from typing import Dict, List, Any, Optional, Union, IO
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import hashlib
import uuid
import os
from pathlib import Path


class ReportFormat(Enum):
    """Supported report formats."""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"


class ReportType(Enum):
    """Types of security reports."""
    SECURITY_SUMMARY = "security_summary"
    THREAT_ANALYSIS = "threat_analysis"
    COMPLIANCE_AUDIT = "compliance_audit"
    INCIDENT_REPORT = "incident_report"
    RISK_ASSESSMENT = "risk_assessment"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    CUSTOM = "custom"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SecurityReport:
    """Comprehensive security report structure."""
    
    # Report metadata
    report_id: str
    report_type: ReportType
    title: str
    generated_timestamp: datetime
    period_start: datetime
    period_end: datetime
    
    # Executive summary
    executive_summary: Dict[str, Any]
    
    # Key metrics
    metrics: Dict[str, Any]
    
    # Detailed findings
    findings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Risk analysis
    risk_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Charts and visualizations data
    visualizations: Dict[str, Any] = field(default_factory=dict)
    
    # Appendices and raw data
    appendices: Dict[str, Any] = field(default_factory=dict)
    
    # Report metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        report_dict = asdict(self)
        # Convert enum and datetime values
        report_dict['report_type'] = self.report_type.value
        report_dict['generated_timestamp'] = self.generated_timestamp.isoformat()
        report_dict['period_start'] = self.period_start.isoformat()
        report_dict['period_end'] = self.period_end.isoformat()
        return report_dict
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str, format: ReportFormat = ReportFormat.JSON) -> None:
        """Save report to file in specified format."""
        if format == ReportFormat.JSON:
            with open(filepath, 'w') as f:
                f.write(self.to_json())
        elif format == ReportFormat.CSV:
            self._save_as_csv(filepath)
        elif format == ReportFormat.HTML:
            self._save_as_html(filepath)
        elif format == ReportFormat.MARKDOWN:
            self._save_as_markdown(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_as_csv(self, filepath: str) -> None:
        """Save key metrics as CSV."""
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Metric', 'Value', 'Description'])
            
            # Write metrics
            for key, value in self.metrics.items():
                description = self.metadata.get(f"{key}_description", "")
                writer.writerow([key, value, description])
    
    def _save_as_html(self, filepath: str) -> None:
        """Save report as HTML."""
        html_content = self._generate_html()
        with open(filepath, 'w') as f:
            f.write(html_content)
    
    def _save_as_markdown(self, filepath: str) -> None:
        """Save report as Markdown."""
        markdown_content = self._generate_markdown()
        with open(filepath, 'w') as f:
            f.write(markdown_content)
    
    def _generate_html(self) -> str:
        """Generate HTML representation of the report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .finding {{ background-color: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .recommendation {{ background-color: #d1ecf1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .high-risk {{ background-color: #f8d7da; }}
        .medium-risk {{ background-color: #fff3cd; }}
        .low-risk {{ background-color: #d4edda; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.title}</h1>
        <p><strong>Report ID:</strong> {self.report_id}</p>
        <p><strong>Generated:</strong> {self.generated_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        <p><strong>Period:</strong> {self.period_start.strftime('%Y-%m-%d')} to {self.period_end.strftime('%Y-%m-%d')}</p>
    </div>
    
    <h2>Executive Summary</h2>
    <div class="executive-summary">
        {self._format_dict_as_html(self.executive_summary)}
    </div>
    
    <h2>Key Metrics</h2>
    <div class="metrics">
        {self._format_dict_as_html(self.metrics)}
    </div>
    
    <h2>Findings</h2>
    <div class="findings">
        {self._format_findings_as_html()}
    </div>
    
    <h2>Recommendations</h2>
    <div class="recommendations">
        {self._format_recommendations_as_html()}
    </div>
</body>
</html>
        """
        return html
    
    def _generate_markdown(self) -> str:
        """Generate Markdown representation of the report."""
        markdown = f"""# {self.title}

**Report ID:** {self.report_id}
**Generated:** {self.generated_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
**Period:** {self.period_start.strftime('%Y-%m-%d')} to {self.period_end.strftime('%Y-%m-%d')}

## Executive Summary

{self._format_dict_as_markdown(self.executive_summary)}

## Key Metrics

{self._format_dict_as_markdown(self.metrics)}

## Findings

{self._format_findings_as_markdown()}

## Recommendations

{self._format_recommendations_as_markdown()}

## Risk Analysis

{self._format_dict_as_markdown(self.risk_analysis)}
"""
        return markdown
    
    def _format_dict_as_html(self, data: Dict[str, Any]) -> str:
        """Format dictionary as HTML."""
        html = "<ul>"
        for key, value in data.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"
        return html
    
    def _format_dict_as_markdown(self, data: Dict[str, Any]) -> str:
        """Format dictionary as Markdown."""
        markdown = ""
        for key, value in data.items():
            markdown += f"- **{key}:** {value}\n"
        return markdown
    
    def _format_findings_as_html(self) -> str:
        """Format findings as HTML."""
        html = ""
        for finding in self.findings:
            risk_class = f"{finding.get('risk_level', 'low')}-risk"
            html += f'<div class="finding {risk_class}">'
            html += f"<h3>{finding.get('title', 'Finding')}</h3>"
            html += f"<p>{finding.get('description', '')}</p>"
            html += f"<p><strong>Risk Level:</strong> {finding.get('risk_level', 'Unknown')}</p>"
            html += "</div>"
        return html
    
    def _format_findings_as_markdown(self) -> str:
        """Format findings as Markdown."""
        markdown = ""
        for i, finding in enumerate(self.findings, 1):
            markdown += f"### Finding {i}: {finding.get('title', 'Untitled')}\n\n"
            markdown += f"{finding.get('description', '')}\n\n"
            markdown += f"**Risk Level:** {finding.get('risk_level', 'Unknown')}\n\n"
        return markdown
    
    def _format_recommendations_as_html(self) -> str:
        """Format recommendations as HTML."""
        html = ""
        for i, recommendation in enumerate(self.recommendations, 1):
            html += f'<div class="recommendation">{i}. {recommendation}</div>'
        return html
    
    def _format_recommendations_as_markdown(self) -> str:
        """Format recommendations as Markdown."""
        markdown = ""
        for i, recommendation in enumerate(self.recommendations, 1):
            markdown += f"{i}. {recommendation}\n"
        return markdown


@dataclass
class AuditEvent:
    """Immutable audit event record."""
    
    # Event identification
    event_id: str
    timestamp: datetime
    event_type: str
    
    # Event source
    source_system: str
    source_component: str
    
    # Event details
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Event data
    event_data: Dict[str, Any] = field(default_factory=dict)
    
    # Security context
    security_level: str = "info"
    risk_score: float = 0.0
    
    # Integrity
    checksum: str = field(default="", init=False)
    
    def __post_init__(self):
        """Calculate checksum for integrity verification."""
        # Create deterministic string representation
        data_str = f"{self.event_id}|{self.timestamp.isoformat()}|{self.event_type}|{json.dumps(self.event_data, sort_keys=True)}"
        self.checksum = hashlib.sha256(data_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity using checksum."""
        data_str = f"{self.event_id}|{self.timestamp.isoformat()}|{self.event_type}|{json.dumps(self.event_data, sort_keys=True)}"
        expected_checksum = hashlib.sha256(data_str.encode()).hexdigest()
        return self.checksum == expected_checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        event_dict = asdict(self)
        event_dict['timestamp'] = self.timestamp.isoformat()
        return event_dict
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class SecurityReporter:
    """
    Comprehensive security reporting system.
    
    Generates various types of security reports, dashboards, and analytics
    from security monitoring data.
    """
    
    def __init__(
        self,
        output_directory: str = "./security_reports",
        auto_generate: bool = True,
        report_schedule: Dict[str, int] = None,
        **kwargs
    ):
        """
        Initialize SecurityReporter.
        
        Args:
            output_directory: Directory for saving reports
            auto_generate: Enable automatic report generation
            report_schedule: Schedule for automatic reports (report_type: interval_hours)
        """
        self.output_directory = Path(output_directory)
        self.auto_generate = auto_generate
        self.report_schedule = report_schedule or {
            'security_summary': 24,  # Daily
            'threat_analysis': 168,  # Weekly
        }
        
        # Create output directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Security monitor reference
        self._security_monitor = None
        
        # Report templates
        self._report_templates = self._load_report_templates()
        
        # Statistics
        self._reports_generated = 0
        self._last_report_time = None
        
        # Background reporting
        self._reporting_thread = None
        self._running = False
        
        print(f"📊 SecurityReporter initialized")
        print(f"   Output directory: {self.output_directory}")
        print(f"   Auto-generate: {self.auto_generate}")
    
    def set_security_monitor(self, monitor):
        """Set reference to the security monitor."""
        self._security_monitor = monitor
    
    def start(self):
        """Start automatic report generation."""
        if self.auto_generate:
            self._running = True
            self._start_reporting_thread()
            print("   Automatic reporting started")
    
    def stop(self):
        """Stop automatic report generation."""
        self._running = False
        if self._reporting_thread and self._reporting_thread.is_alive():
            self._reporting_thread.join(timeout=1)
        print("   Automatic reporting stopped")
    
    def generate_security_summary(
        self,
        period_days: int = 7,
        include_trends: bool = True,
        format: ReportFormat = ReportFormat.JSON
    ) -> SecurityReport:
        """Generate comprehensive security summary report."""
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=period_days)
        
        # Collect data from security monitor
        if not self._security_monitor:
            raise ValueError("Security monitor not set")
        
        events = self._security_monitor.get_events()
        metrics = self._security_monitor.get_metrics()
        
        # Filter events by time period
        period_events = [e for e in events if start_time <= e.timestamp <= end_time]
        
        # Generate executive summary
        executive_summary = {
            "total_events": len(period_events),
            "high_risk_events": len([e for e in period_events if e.risk_score >= 0.7]),
            "critical_events": len([e for e in period_events if e.severity.value == "critical"]),
            "average_risk_score": sum(e.risk_score for e in period_events) / len(period_events) if period_events else 0.0,
            "period_days": period_days,
            "monitoring_active": self._security_monitor._monitoring_active
        }
        
        # Generate key metrics
        key_metrics = {
            "total_interactions_monitored": metrics.total_events,
            "unique_agents": metrics.agents_monitored,
            "unique_sessions": metrics.sessions_monitored,
            "unique_users": metrics.unique_users,
            "events_by_type": metrics.events_by_type,
            "events_by_severity": metrics.events_by_severity,
            "compliance_score": getattr(metrics, 'compliance_score', 0.0)
        }
        
        # Generate findings
        findings = self._generate_security_findings(period_events)
        
        # Generate risk analysis
        risk_analysis = self._generate_risk_analysis(period_events, metrics)
        
        # Generate recommendations
        recommendations = self._generate_security_recommendations(period_events, metrics)
        
        # Create report
        report = SecurityReport(
            report_id=str(uuid.uuid4()),
            report_type=ReportType.SECURITY_SUMMARY,
            title=f"Security Summary Report - {period_days} Day Period",
            generated_timestamp=datetime.now(timezone.utc),
            period_start=start_time,
            period_end=end_time,
            executive_summary=executive_summary,
            metrics=key_metrics,
            findings=findings,
            risk_analysis=risk_analysis,
            recommendations=recommendations,
            metadata={
                "report_version": "1.0",
                "generator": "SecurityReporter",
                "period_days": period_days,
                "include_trends": include_trends
            }
        )
        
        # Save report
        filename = f"security_summary_{end_time.strftime('%Y%m%d_%H%M%S')}.{format.value}"
        filepath = self.output_directory / filename
        report.save(str(filepath), format)
        
        self._reports_generated += 1
        self._last_report_time = datetime.now(timezone.utc)
        
        print(f"📋 Security summary report generated: {filename}")
        
        return report
    
    def generate_threat_analysis(
        self,
        period_days: int = 30,
        threat_types: List[str] = None,
        format: ReportFormat = ReportFormat.JSON
    ) -> SecurityReport:
        """Generate detailed threat analysis report."""
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=period_days)
        
        if not self._security_monitor:
            raise ValueError("Security monitor not set")
        
        # Get threat-related events
        events = self._security_monitor.get_events()
        threat_events = [e for e in events if start_time <= e.timestamp <= end_time]
        
        # Filter by threat types if specified
        if threat_types:
            threat_events = [e for e in threat_events if e.event_type.value in threat_types]
        
        # Analyze threats
        threat_analysis = self._analyze_threats(threat_events)
        
        # Generate executive summary
        executive_summary = {
            "total_threats": len(threat_events),
            "unique_threat_types": len(set(e.event_type.value for e in threat_events)),
            "most_common_threat": threat_analysis.get("most_common_threat", "None"),
            "threat_trend": threat_analysis.get("trend", "stable"),
            "average_threat_severity": threat_analysis.get("average_severity", 0.0)
        }
        
        # Create report
        report = SecurityReport(
            report_id=str(uuid.uuid4()),
            report_type=ReportType.THREAT_ANALYSIS,
            title=f"Threat Analysis Report - {period_days} Day Period",
            generated_timestamp=datetime.now(timezone.utc),
            period_start=start_time,
            period_end=end_time,
            executive_summary=executive_summary,
            metrics=threat_analysis.get("metrics", {}),
            findings=threat_analysis.get("findings", []),
            risk_analysis=threat_analysis.get("risk_analysis", {}),
            recommendations=threat_analysis.get("recommendations", []),
            metadata={
                "threat_types_analyzed": threat_types or "all",
                "analysis_depth": "detailed"
            }
        )
        
        # Save report
        filename = f"threat_analysis_{end_time.strftime('%Y%m%d_%H%M%S')}.{format.value}"
        filepath = self.output_directory / filename
        report.save(str(filepath), format)
        
        self._reports_generated += 1
        
        print(f"🎯 Threat analysis report generated: {filename}")
        
        return report
    
    def generate_custom_report(
        self,
        title: str,
        data_sources: List[str],
        filters: Dict[str, Any] = None,
        template: str = None,
        format: ReportFormat = ReportFormat.JSON
    ) -> SecurityReport:
        """Generate custom report based on specified parameters."""
        
        # Implementation would depend on specific requirements
        # This is a placeholder for custom report generation
        
        report = SecurityReport(
            report_id=str(uuid.uuid4()),
            report_type=ReportType.CUSTOM,
            title=title,
            generated_timestamp=datetime.now(timezone.utc),
            period_start=datetime.now(timezone.utc) - timedelta(days=7),
            period_end=datetime.now(timezone.utc),
            executive_summary={"custom_report": True},
            metrics={"data_sources": len(data_sources)},
            metadata={
                "template": template,
                "filters": filters or {},
                "data_sources": data_sources
            }
        )
        
        # Save report
        filename = f"custom_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format.value}"
        filepath = self.output_directory / filename
        report.save(str(filepath), format)
        
        self._reports_generated += 1
        
        print(f"📄 Custom report generated: {filename}")
        
        return report
    
    def _generate_security_findings(self, events: List) -> List[Dict[str, Any]]:
        """Generate security findings from events."""
        findings = []
        
        # High-risk events
        high_risk_events = [e for e in events if e.risk_score >= 0.7]
        if high_risk_events:
            findings.append({
                "title": "High-Risk Security Events Detected",
                "description": f"Detected {len(high_risk_events)} high-risk security events requiring attention.",
                "risk_level": "high",
                "count": len(high_risk_events),
                "event_types": list(set(e.event_type.value for e in high_risk_events))
            })
        
        # Frequent threat patterns
        event_types = [e.event_type.value for e in events]
        if event_types:
            most_common = max(set(event_types), key=event_types.count)
            count = event_types.count(most_common)
            if count > 5:
                findings.append({
                    "title": "Frequent Security Pattern Detected",
                    "description": f"'{most_common}' events occurred {count} times, indicating a potential pattern.",
                    "risk_level": "medium",
                    "pattern": most_common,
                    "frequency": count
                })
        
        return findings
    
    def _generate_risk_analysis(self, events: List, metrics) -> Dict[str, Any]:
        """Generate risk analysis from events and metrics."""
        if not events:
            return {"overall_risk": "low", "risk_factors": []}
        
        # Calculate risk metrics
        avg_risk = sum(e.risk_score for e in events) / len(events)
        max_risk = max(e.risk_score for e in events)
        high_risk_count = len([e for e in events if e.risk_score >= 0.7])
        
        # Determine overall risk level
        if max_risk >= 0.9 or high_risk_count > 10:
            overall_risk = "critical"
        elif max_risk >= 0.7 or high_risk_count > 5:
            overall_risk = "high"
        elif avg_risk >= 0.4:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        return {
            "overall_risk": overall_risk,
            "average_risk_score": avg_risk,
            "maximum_risk_score": max_risk,
            "high_risk_event_count": high_risk_count,
            "risk_factors": [
                {"factor": "Event frequency", "impact": "medium"},
                {"factor": "Risk score distribution", "impact": "high"},
                {"factor": "Threat diversity", "impact": "medium"}
            ]
        }
    
    def _generate_security_recommendations(self, events: List, metrics) -> List[str]:
        """Generate security recommendations based on analysis."""
        recommendations = []
        
        # High-risk event recommendations
        high_risk_count = len([e for e in events if e.risk_score >= 0.7])
        if high_risk_count > 5:
            recommendations.append("Implement additional security controls for high-risk interactions")
            recommendations.append("Review and update threat detection rules")
        
        # Event volume recommendations
        if len(events) > 100:
            recommendations.append("Consider implementing automated response mechanisms")
            recommendations.append("Increase monitoring frequency for high-activity periods")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Continue current security monitoring practices")
            recommendations.append("Regular review of security policies and procedures")
        
        return recommendations
    
    def _analyze_threats(self, threat_events: List) -> Dict[str, Any]:
        """Analyze threat patterns and characteristics."""
        if not threat_events:
            return {
                "most_common_threat": "None",
                "trend": "stable",
                "average_severity": 0.0,
                "metrics": {},
                "findings": [],
                "risk_analysis": {},
                "recommendations": ["No threats detected - maintain current security posture"]
            }
        
        # Analyze threat types
        threat_types = [e.event_type.value for e in threat_events]
        threat_counts = {threat: threat_types.count(threat) for threat in set(threat_types)}
        most_common_threat = max(threat_counts, key=threat_counts.get)
        
        # Calculate severity metrics
        severity_scores = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
        avg_severity = sum(severity_scores.get(e.severity.value, 0.0) for e in threat_events) / len(threat_events)
        
        # Analyze trends (simplified)
        if len(threat_events) >= 10:
            recent_half = threat_events[-len(threat_events)//2:]
            earlier_half = threat_events[:len(threat_events)//2]
            
            if len(recent_half) > len(earlier_half) * 1.2:
                trend = "increasing"
            elif len(recent_half) < len(earlier_half) * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "most_common_threat": most_common_threat,
            "trend": trend,
            "average_severity": avg_severity,
            "metrics": {
                "threat_type_distribution": threat_counts,
                "total_threats": len(threat_events),
                "unique_threat_types": len(set(threat_types))
            },
            "findings": [
                {
                    "title": f"Primary Threat: {most_common_threat}",
                    "description": f"Most frequent threat type with {threat_counts[most_common_threat]} occurrences",
                    "risk_level": "medium" if threat_counts[most_common_threat] > 5 else "low"
                }
            ],
            "risk_analysis": {
                "threat_diversity": len(set(threat_types)),
                "severity_distribution": {sev: sum(1 for e in threat_events if e.severity.value == sev) 
                                       for sev in ["low", "medium", "high", "critical"]}
            },
            "recommendations": [
                f"Focus security controls on {most_common_threat} threats",
                "Implement threat-specific detection rules",
                "Monitor threat trend patterns for early warning"
            ]
        }
    
    def _load_report_templates(self) -> Dict[str, Any]:
        """Load report templates for different report types."""
        return {
            "security_summary": {
                "sections": ["executive_summary", "metrics", "findings", "recommendations"],
                "charts": ["risk_trend", "event_distribution", "severity_breakdown"]
            },
            "threat_analysis": {
                "sections": ["threat_overview", "detailed_analysis", "risk_assessment", "mitigation"],
                "charts": ["threat_timeline", "threat_types", "severity_distribution"]
            },
            "compliance_audit": {
                "sections": ["compliance_status", "violations", "remediation", "certification"],
                "charts": ["compliance_score", "violation_trends", "framework_coverage"]
            }
        }
    
    def _start_reporting_thread(self) -> None:
        """Start background thread for automatic report generation."""
        def reporting_loop():
            while self._running:
                try:
                    current_time = datetime.now(timezone.utc)
                    
                    # Check if any reports need to be generated
                    for report_type, interval_hours in self.report_schedule.items():
                        if self._should_generate_report(report_type, interval_hours, current_time):
                            if report_type == 'security_summary':
                                self.generate_security_summary(period_days=1)
                            elif report_type == 'threat_analysis':
                                self.generate_threat_analysis(period_days=7)
                    
                    # Sleep for an hour before checking again
                    time.sleep(3600)
                    
                except Exception as e:
                    print(f"⚠️  Error in automatic reporting: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
        
        self._reporting_thread = threading.Thread(target=reporting_loop, daemon=True)
        self._reporting_thread.start()
    
    def _should_generate_report(self, report_type: str, interval_hours: int, current_time: datetime) -> bool:
        """Check if a report should be generated based on schedule."""
        # Simple implementation - in practice would track last generation times per report type
        if not self._last_report_time:
            return True
        
        time_since_last = current_time - self._last_report_time
        return time_since_last.total_seconds() >= (interval_hours * 3600)


class AuditLogger:
    """
    Immutable audit trail logging system.
    
    Provides tamper-evident logging for security events and system actions
    with integrity verification and compliance support.
    """
    
    def __init__(
        self,
        log_directory: str = "./audit_logs",
        max_log_size: int = 100 * 1024 * 1024,  # 100MB
        compression_enabled: bool = True,
        encryption_enabled: bool = False,
        **kwargs
    ):
        """
        Initialize AuditLogger.
        
        Args:
            log_directory: Directory for storing audit logs
            max_log_size: Maximum size per log file before rotation
            compression_enabled: Enable log compression
            encryption_enabled: Enable log encryption (requires key management)
        """
        self.log_directory = Path(log_directory)
        self.max_log_size = max_log_size
        self.compression_enabled = compression_enabled
        self.encryption_enabled = encryption_enabled
        
        # Create log directory
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Current log file
        self._current_log_file = None
        self._current_log_size = 0
        
        # Event buffer for performance
        self._event_buffer: List[AuditEvent] = []
        self._buffer_lock = threading.Lock()
        self._buffer_size_limit = 1000
        
        # Statistics
        self._events_logged = 0
        self._files_created = 0
        
        # Background flushing
        self._flush_thread = None
        self._running = False
        
        # Initialize first log file
        self._rotate_log_file()
        
        print(f"📝 AuditLogger initialized")
        print(f"   Log directory: {self.log_directory}")
        print(f"   Max log size: {self.max_log_size // (1024*1024)}MB")
    
    def start(self):
        """Start audit logging."""
        self._running = True
        self._start_flush_thread()
        print("   Audit logging started")
    
    def stop(self):
        """Stop audit logging and flush remaining events."""
        self._running = False
        
        # Flush remaining events
        self._flush_events()
        
        # Stop flush thread
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=1)
        
        # Close current log file
        if self._current_log_file:
            self._current_log_file.close()
        
        print("   Audit logging stopped")
    
    def log_event(
        self,
        event_type: str,
        source_component: str,
        event_data: Dict[str, Any],
        agent_id: str = None,
        session_id: str = None,
        user_id: str = None,
        security_level: str = "info",
        risk_score: float = 0.0
    ) -> str:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event being logged
            source_component: Component that generated the event
            event_data: Event-specific data
            agent_id: Associated agent ID
            session_id: Associated session ID
            user_id: Associated user ID
            security_level: Security level of the event
            risk_score: Risk score associated with the event
        
        Returns:
            Event ID of the logged event
        """
        # Create audit event
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            source_system="SecurityMonitor",
            source_component=source_component,
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            event_data=event_data,
            security_level=security_level,
            risk_score=risk_score
        )
        
        # Add to buffer
        with self._buffer_lock:
            self._event_buffer.append(event)
            
            # Flush if buffer is full
            if len(self._event_buffer) >= self._buffer_size_limit:
                self._flush_events_internal()
        
        self._events_logged += 1
        
        return event.event_id
    
    def get_audit_trail(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        event_types: List[str] = None,
        agent_id: str = None,
        session_id: str = None,
        user_id: str = None,
        limit: int = None
    ) -> List[AuditEvent]:
        """
        Retrieve audit trail with optional filtering.
        
        Args:
            start_time: Start time for filtering
            end_time: End time for filtering
            event_types: List of event types to include
            agent_id: Filter by agent ID
            session_id: Filter by session ID
            user_id: Filter by user ID
            limit: Maximum number of events to return
        
        Returns:
            List of audit events matching the criteria
        """
        events = []
        
        # Read from all log files
        for log_file_path in sorted(self.log_directory.glob("audit_*.jsonl")):
            try:
                with open(log_file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            event_data = json.loads(line)
                            event = AuditEvent(**event_data)
                            
                            # Verify integrity
                            if not event.verify_integrity():
                                print(f"⚠️  Integrity check failed for event {event.event_id}")
                                continue
                            
                            # Apply filters
                            if start_time and event.timestamp < start_time:
                                continue
                            if end_time and event.timestamp > end_time:
                                continue
                            if event_types and event.event_type not in event_types:
                                continue
                            if agent_id and event.agent_id != agent_id:
                                continue
                            if session_id and event.session_id != session_id:
                                continue
                            if user_id and event.user_id != user_id:
                                continue
                            
                            events.append(event)
                            
                            # Apply limit
                            if limit and len(events) >= limit:
                                return events
                                
            except Exception as e:
                print(f"⚠️  Error reading audit log {log_file_path}: {e}")
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return events[:limit] if limit else events
    
    def verify_audit_integrity(self, start_time: datetime = None, end_time: datetime = None) -> Dict[str, Any]:
        """
        Verify integrity of audit logs.
        
        Args:
            start_time: Start time for verification
            end_time: End time for verification
        
        Returns:
            Dictionary with verification results
        """
        total_events = 0
        valid_events = 0
        invalid_events = 0
        corrupted_files = []
        
        for log_file_path in sorted(self.log_directory.glob("audit_*.jsonl")):
            try:
                with open(log_file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            total_events += 1
                            try:
                                event_data = json.loads(line)
                                event = AuditEvent(**event_data)
                                
                                # Check time range
                                if start_time and event.timestamp < start_time:
                                    continue
                                if end_time and event.timestamp > end_time:
                                    continue
                                
                                if event.verify_integrity():
                                    valid_events += 1
                                else:
                                    invalid_events += 1
                                    
                            except Exception:
                                invalid_events += 1
                                
            except Exception as e:
                corrupted_files.append(str(log_file_path))
                print(f"⚠️  Corrupted audit log file: {log_file_path}")
        
        integrity_rate = (valid_events / total_events * 100) if total_events > 0 else 100.0
        
        return {
            "total_events": total_events,
            "valid_events": valid_events,
            "invalid_events": invalid_events,
            "integrity_rate": integrity_rate,
            "corrupted_files": corrupted_files,
            "verification_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _flush_events(self) -> None:
        """Flush events from buffer to log file."""
        with self._buffer_lock:
            self._flush_events_internal()
    
    def _flush_events_internal(self) -> None:
        """Internal method to flush events (assumes lock is held)."""
        if not self._event_buffer:
            return
        
        # Ensure we have a current log file
        if not self._current_log_file:
            self._rotate_log_file()
        
        # Write events to log file
        for event in self._event_buffer:
            event_json = event.to_json()
            self._current_log_file.write(event_json + '\n')
            self._current_log_size += len(event_json) + 1
        
        # Flush to disk
        self._current_log_file.flush()
        
        # Clear buffer
        self._event_buffer.clear()
        
        # Rotate log file if it's too large
        if self._current_log_size >= self.max_log_size:
            self._rotate_log_file()
    
    def _rotate_log_file(self) -> None:
        """Rotate to a new log file."""
        # Close current file
        if self._current_log_file:
            self._current_log_file.close()
        
        # Create new log file
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        log_filename = f"audit_{timestamp}_{uuid.uuid4().hex[:8]}.jsonl"
        log_filepath = self.log_directory / log_filename
        
        self._current_log_file = open(log_filepath, 'w')
        self._current_log_size = 0
        self._files_created += 1
        
        print(f"📄 New audit log file created: {log_filename}")
    
    def _start_flush_thread(self) -> None:
        """Start background thread for periodic event flushing."""
        def flush_loop():
            while self._running:
                try:
                    time.sleep(10)  # Flush every 10 seconds
                    self._flush_events()
                except Exception as e:
                    print(f"⚠️  Error in audit log flushing: {e}")
                    time.sleep(5)
        
        self._flush_thread = threading.Thread(target=flush_loop, daemon=True)
        self._flush_thread.start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audit logger statistics."""
        return {
            "events_logged": self._events_logged,
            "files_created": self._files_created,
            "current_buffer_size": len(self._event_buffer),
            "current_log_size": self._current_log_size,
            "log_directory": str(self.log_directory),
            "running": self._running
        }





