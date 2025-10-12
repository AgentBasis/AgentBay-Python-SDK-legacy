"""
Compliance and Governance Features

Provides comprehensive compliance tracking and governance capabilities:
- ComplianceMonitor: Main compliance tracking orchestrator
- ComplianceReport: Structured compliance reporting
- Data retention and privacy controls
- Regulatory compliance metrics
- Audit trail generation
"""

import json
import time
import threading
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import hashlib
import uuid
import os


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"  # General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    SOC2 = "soc2"  # Service Organization Control 2
    ISO27001 = "iso27001"  # ISO/IEC 27001
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    CUSTOM = "custom"  # Custom compliance requirements


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    SECRET = "secret"


class RetentionPolicy(Enum):
    """Data retention policies."""
    IMMEDIATE = "immediate"  # Delete immediately
    SHORT_TERM = "short_term"  # 30 days
    MEDIUM_TERM = "medium_term"  # 1 year
    LONG_TERM = "long_term"  # 7 years
    PERMANENT = "permanent"  # Keep indefinitely
    CUSTOM = "custom"  # Custom retention period


@dataclass
class ComplianceRule:
    """Represents a compliance rule or requirement."""
    rule_id: str
    framework: ComplianceFramework
    category: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    automated_check: bool = True
    check_function: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceViolation:
    """Represents a compliance violation."""
    violation_id: str
    rule_id: str
    timestamp: datetime
    agent_id: str
    session_id: Optional[str]
    user_id: Optional[str]
    violation_type: str
    description: str
    severity: str
    data_involved: Dict[str, Any]
    remediation_required: bool
    remediation_steps: List[str]
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


@dataclass
class DataRetentionRecord:
    """Tracks data retention for compliance."""
    record_id: str
    data_type: str
    classification: DataClassification
    retention_policy: RetentionPolicy
    created_timestamp: datetime
    retention_until: datetime
    agent_id: str
    session_id: Optional[str]
    user_id: Optional[str]
    data_hash: str  # Hash of the actual data
    metadata: Dict[str, Any] = field(default_factory=dict)
    deleted: bool = False
    deletion_timestamp: Optional[datetime] = None


@dataclass
class ComplianceMetrics:
    """Compliance monitoring metrics."""
    framework: ComplianceFramework
    period_start: datetime
    period_end: datetime
    
    # Violation metrics
    total_violations: int = 0
    violations_by_severity: Dict[str, int] = field(default_factory=dict)
    violations_by_type: Dict[str, int] = field(default_factory=dict)
    resolved_violations: int = 0
    
    # Data governance metrics
    total_data_records: int = 0
    records_by_classification: Dict[str, int] = field(default_factory=dict)
    retention_compliance_rate: float = 0.0
    data_deletion_requests: int = 0
    
    # Access and usage metrics
    data_access_requests: int = 0
    consent_tracking_records: int = 0
    privacy_policy_updates: int = 0
    
    # Audit metrics
    audit_events: int = 0
    security_incidents: int = 0
    compliance_score: float = 0.0  # 0.0 to 100.0


@dataclass
class ComplianceReport:
    """Comprehensive compliance report."""
    report_id: str
    generated_timestamp: datetime
    framework: ComplianceFramework
    period_start: datetime
    period_end: datetime
    
    # Executive summary
    overall_compliance_score: float
    critical_violations: int
    high_priority_actions: List[str]
    
    # Detailed metrics
    metrics: ComplianceMetrics
    violations: List[ComplianceViolation]
    
    # Data governance
    retention_summary: Dict[str, Any]
    data_classification_summary: Dict[str, Any]
    
    # Recommendations
    recommendations: List[str]
    remediation_plan: List[Dict[str, Any]]
    
    # Metadata
    report_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        report_dict = asdict(self)
        # Convert datetime objects to ISO strings
        report_dict['generated_timestamp'] = self.generated_timestamp.isoformat()
        report_dict['period_start'] = self.period_start.isoformat()
        report_dict['period_end'] = self.period_end.isoformat()
        report_dict['framework'] = self.framework.value
        
        # Convert violation timestamps
        for violation in report_dict['violations']:
            violation['timestamp'] = violation['timestamp'].isoformat() if isinstance(violation['timestamp'], datetime) else violation['timestamp']
            if violation.get('resolution_timestamp'):
                violation['resolution_timestamp'] = violation['resolution_timestamp'].isoformat() if isinstance(violation['resolution_timestamp'], datetime) else violation['resolution_timestamp']
        
        return report_dict
    
    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save_to_file(self, filepath: str) -> None:
        """Save report to file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())


class ComplianceMonitor:
    """
    Comprehensive compliance monitoring and governance system.
    
    Tracks regulatory compliance, manages data retention, and provides
    audit capabilities for AI agent interactions.
    """
    
    def __init__(
        self,
        frameworks: List[ComplianceFramework] = None,
        data_retention_enabled: bool = True,
        audit_logging_enabled: bool = True,
        privacy_mode: bool = False,
        storage_path: str = None,
        **kwargs
    ):
        """
        Initialize ComplianceMonitor.
        
        Args:
            frameworks: List of compliance frameworks to monitor
            data_retention_enabled: Enable data retention tracking
            audit_logging_enabled: Enable audit logging
            privacy_mode: Enable privacy-preserving mode
            storage_path: Path for storing compliance data
        """
        self.frameworks = frameworks or [ComplianceFramework.GDPR]
        self.data_retention_enabled = data_retention_enabled
        self.audit_logging_enabled = audit_logging_enabled
        self.privacy_mode = privacy_mode
        self.storage_path = storage_path or "./compliance_data"
        
        # Ensure storage directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Internal state
        self._security_monitor = None
        self._running = False
        
        # Compliance tracking
        self._compliance_rules: Dict[str, ComplianceRule] = {}
        self._violations: List[ComplianceViolation] = []
        self._retention_records: Dict[str, DataRetentionRecord] = {}
        
        # Audit logging
        self._audit_events: List[Dict[str, Any]] = []
        
        # Statistics
        self._total_interactions_monitored = 0
        self._compliance_checks_performed = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background tasks
        self._retention_cleanup_thread = None
        self._metrics_thread = None
        
        # Load default rules
        self._load_default_compliance_rules()
        
        print(f"📋 ComplianceMonitor initialized")
        print(f"   Frameworks: {[f.value for f in self.frameworks]}")
        print(f"   Data retention: {'enabled' if data_retention_enabled else 'disabled'}")
        print(f"   Audit logging: {'enabled' if audit_logging_enabled else 'disabled'}")
    
    def set_security_monitor(self, monitor):
        """Set reference to the main security monitor."""
        self._security_monitor = monitor
    
    def start(self):
        """Start compliance monitoring."""
        self._running = True
        
        if self.data_retention_enabled:
            self._start_retention_cleanup_thread()
        
        self._start_metrics_thread()
        print("   Compliance monitoring started")
    
    def stop(self):
        """Stop compliance monitoring."""
        self._running = False
        
        # Stop background threads
        if self._retention_cleanup_thread and self._retention_cleanup_thread.is_alive():
            self._retention_cleanup_thread.join(timeout=1)
        
        if self._metrics_thread and self._metrics_thread.is_alive():
            self._metrics_thread.join(timeout=1)
        
        print("   Compliance monitoring stopped")
    
    def analyze_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """
        Analyze an agent interaction for compliance issues.
        
        Args:
            interaction_data: Dictionary containing interaction details
        """
        self._total_interactions_monitored += 1
        
        # Extract interaction details
        agent_id = interaction_data.get('agent_id')
        user_input = interaction_data.get('user_input', '')
        agent_response = interaction_data.get('agent_response', '')
        session_id = interaction_data.get('session_id')
        user_id = interaction_data.get('user_id')
        timestamp = datetime.now(timezone.utc)
        
        # Create data retention record
        if self.data_retention_enabled:
            self._create_retention_record(
                data_type="agent_interaction",
                agent_id=agent_id,
                session_id=session_id,
                user_id=user_id,
                data_content=interaction_data,
                classification=DataClassification.CONFIDENTIAL
            )
        
        # Perform compliance checks
        self._perform_compliance_checks(interaction_data)
        
        # Log audit event
        if self.audit_logging_enabled:
            self._log_audit_event(
                event_type="interaction_analyzed",
                agent_id=agent_id,
                session_id=session_id,
                user_id=user_id,
                metadata={"timestamp": timestamp.isoformat()}
            )
    
    def add_compliance_rule(self, rule: ComplianceRule) -> None:
        """Add a custom compliance rule."""
        with self._lock:
            self._compliance_rules[rule.rule_id] = rule
        print(f"   Added compliance rule: {rule.rule_id}")
    
    def record_violation(self, violation: ComplianceViolation) -> None:
        """Record a compliance violation."""
        with self._lock:
            self._violations.append(violation)
        
        # Log audit event
        if self.audit_logging_enabled:
            self._log_audit_event(
                event_type="compliance_violation",
                agent_id=violation.agent_id,
                session_id=violation.session_id,
                user_id=violation.user_id,
                metadata={
                    "violation_id": violation.violation_id,
                    "rule_id": violation.rule_id,
                    "severity": violation.severity
                }
            )
        
        print(f"🚨 Compliance violation recorded: {violation.violation_type} (severity: {violation.severity})")
    
    def request_data_deletion(
        self, 
        user_id: str, 
        reason: str = "user_request",
        data_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Handle data deletion request (e.g., GDPR right to be forgotten).
        
        Args:
            user_id: User requesting data deletion
            reason: Reason for deletion
            data_types: Specific data types to delete (None for all)
        
        Returns:
            Dictionary with deletion summary
        """
        with self._lock:
            records_to_delete = []
            
            for record in self._retention_records.values():
                if record.user_id == user_id and not record.deleted:
                    if data_types is None or record.data_type in data_types:
                        records_to_delete.append(record)
            
            # Mark records as deleted
            deleted_count = 0
            for record in records_to_delete:
                record.deleted = True
                record.deletion_timestamp = datetime.now(timezone.utc)
                deleted_count += 1
        
        # Log audit event
        if self.audit_logging_enabled:
            self._log_audit_event(
                event_type="data_deletion_request",
                user_id=user_id,
                metadata={
                    "reason": reason,
                    "records_deleted": deleted_count,
                    "data_types": data_types
                }
            )
        
        print(f"🗑️  Data deletion request processed: {deleted_count} records for user {user_id}")
        
        return {
            "user_id": user_id,
            "records_deleted": deleted_count,
            "deletion_timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason
        }
    
    def generate_compliance_report(
        self, 
        framework: ComplianceFramework,
        period_days: int = 30
    ) -> ComplianceReport:
        """
        Generate a comprehensive compliance report.
        
        Args:
            framework: Compliance framework to report on
            period_days: Number of days to include in report
        
        Returns:
            ComplianceReport object
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=period_days)
        
        # Calculate metrics
        metrics = self._calculate_compliance_metrics(framework, start_time, end_time)
        
        # Get violations for the period
        period_violations = [
            v for v in self._violations
            if start_time <= v.timestamp <= end_time
        ]
        
        # Calculate overall compliance score
        compliance_score = self._calculate_compliance_score(metrics, period_violations)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, period_violations)
        
        # Create remediation plan
        remediation_plan = self._create_remediation_plan(period_violations)
        
        # Create report
        report = ComplianceReport(
            report_id=str(uuid.uuid4()),
            generated_timestamp=datetime.now(timezone.utc),
            framework=framework,
            period_start=start_time,
            period_end=end_time,
            overall_compliance_score=compliance_score,
            critical_violations=len([v for v in period_violations if v.severity == "critical"]),
            high_priority_actions=recommendations[:5],  # Top 5 recommendations
            metrics=metrics,
            violations=period_violations,
            retention_summary=self._generate_retention_summary(),
            data_classification_summary=self._generate_classification_summary(),
            recommendations=recommendations,
            remediation_plan=remediation_plan
        )
        
        # Save report to file
        report_filename = f"compliance_report_{framework.value}_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        report_filepath = os.path.join(self.storage_path, report_filename)
        report.save_to_file(report_filepath)
        
        print(f"📊 Compliance report generated: {report_filename}")
        print(f"   Overall score: {compliance_score:.1f}%")
        print(f"   Critical violations: {report.critical_violations}")
        print(f"   Recommendations: {len(recommendations)}")
        
        return report
    
    def get_audit_trail(
        self, 
        start_time: datetime = None,
        end_time: datetime = None,
        event_types: List[str] = None,
        agent_id: str = None,
        user_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit trail with optional filtering.
        
        Args:
            start_time: Start time for filtering
            end_time: End time for filtering
            event_types: List of event types to include
            agent_id: Filter by agent ID
            user_id: Filter by user ID
        
        Returns:
            List of audit events
        """
        with self._lock:
            events = self._audit_events.copy()
        
        # Apply filters
        if start_time:
            events = [e for e in events if e['timestamp'] >= start_time]
        
        if end_time:
            events = [e for e in events if e['timestamp'] <= end_time]
        
        if event_types:
            events = [e for e in events if e['event_type'] in event_types]
        
        if agent_id:
            events = [e for e in events if e.get('agent_id') == agent_id]
        
        if user_id:
            events = [e for e in events if e.get('user_id') == user_id]
        
        return events
    
    def _create_retention_record(
        self,
        data_type: str,
        agent_id: str,
        data_content: Dict[str, Any],
        classification: DataClassification = DataClassification.INTERNAL,
        retention_policy: RetentionPolicy = RetentionPolicy.MEDIUM_TERM,
        session_id: str = None,
        user_id: str = None
    ) -> DataRetentionRecord:
        """Create a data retention record."""
        
        # Determine retention period
        retention_periods = {
            RetentionPolicy.IMMEDIATE: timedelta(days=0),
            RetentionPolicy.SHORT_TERM: timedelta(days=30),
            RetentionPolicy.MEDIUM_TERM: timedelta(days=365),
            RetentionPolicy.LONG_TERM: timedelta(days=365*7),
            RetentionPolicy.PERMANENT: timedelta(days=365*100),  # 100 years as "permanent"
        }
        
        created_time = datetime.now(timezone.utc)
        retention_until = created_time + retention_periods.get(retention_policy, timedelta(days=365))
        
        # Create data hash (for privacy)
        data_str = json.dumps(data_content, sort_keys=True)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()
        
        record = DataRetentionRecord(
            record_id=str(uuid.uuid4()),
            data_type=data_type,
            classification=classification,
            retention_policy=retention_policy,
            created_timestamp=created_time,
            retention_until=retention_until,
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            data_hash=data_hash
        )
        
        with self._lock:
            self._retention_records[record.record_id] = record
        
        return record
    
    def _perform_compliance_checks(self, interaction_data: Dict[str, Any]) -> None:
        """Perform automated compliance checks on interaction data."""
        self._compliance_checks_performed += 1
        
        agent_id = interaction_data.get('agent_id')
        user_input = interaction_data.get('user_input', '')
        agent_response = interaction_data.get('agent_response', '')
        
        # Check for PII in responses (simplified)
        if self._contains_pii(agent_response):
            violation = ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id="pii_in_response",
                timestamp=datetime.now(timezone.utc),
                agent_id=agent_id,
                session_id=interaction_data.get('session_id'),
                user_id=interaction_data.get('user_id'),
                violation_type="data_privacy",
                description="Potential PII detected in agent response",
                severity="high",
                data_involved={"response_length": len(agent_response)},
                remediation_required=True,
                remediation_steps=["Review response content", "Implement PII filtering"]
            )
            self.record_violation(violation)
        
        # Check for data retention compliance
        if interaction_data.get('user_id') and not self._has_valid_consent(interaction_data.get('user_id')):
            violation = ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                rule_id="missing_consent",
                timestamp=datetime.now(timezone.utc),
                agent_id=agent_id,
                session_id=interaction_data.get('session_id'),
                user_id=interaction_data.get('user_id'),
                violation_type="consent_management",
                description="User interaction without valid consent",
                severity="medium",
                data_involved={"interaction_type": "agent_conversation"},
                remediation_required=True,
                remediation_steps=["Obtain user consent", "Update consent records"]
            )
            self.record_violation(violation)
    
    def _contains_pii(self, text: str) -> bool:
        """Check if text contains potential PII (simplified implementation)."""
        import re
        
        # Simple PII patterns
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _has_valid_consent(self, user_id: str) -> bool:
        """Check if user has valid consent (placeholder implementation)."""
        # In a real implementation, this would check against a consent database
        return True  # Assume consent for demo purposes
    
    def _calculate_compliance_metrics(
        self, 
        framework: ComplianceFramework,
        start_time: datetime,
        end_time: datetime
    ) -> ComplianceMetrics:
        """Calculate compliance metrics for a given period."""
        
        # Filter violations by time period
        period_violations = [
            v for v in self._violations
            if start_time <= v.timestamp <= end_time
        ]
        
        # Count violations by severity and type
        violations_by_severity = {}
        violations_by_type = {}
        resolved_count = 0
        
        for violation in period_violations:
            # By severity
            violations_by_severity[violation.severity] = violations_by_severity.get(violation.severity, 0) + 1
            
            # By type
            violations_by_type[violation.violation_type] = violations_by_type.get(violation.violation_type, 0) + 1
            
            # Resolved count
            if violation.resolved:
                resolved_count += 1
        
        # Filter retention records by time period
        period_records = [
            r for r in self._retention_records.values()
            if start_time <= r.created_timestamp <= end_time
        ]
        
        # Count records by classification
        records_by_classification = {}
        for record in period_records:
            classification = record.classification.value
            records_by_classification[classification] = records_by_classification.get(classification, 0) + 1
        
        # Calculate retention compliance rate
        total_records = len(period_records)
        expired_records = [r for r in period_records if r.retention_until < datetime.now(timezone.utc)]
        properly_deleted = [r for r in expired_records if r.deleted]
        retention_compliance_rate = (len(properly_deleted) / len(expired_records)) * 100 if expired_records else 100.0
        
        return ComplianceMetrics(
            framework=framework,
            period_start=start_time,
            period_end=end_time,
            total_violations=len(period_violations),
            violations_by_severity=violations_by_severity,
            violations_by_type=violations_by_type,
            resolved_violations=resolved_count,
            total_data_records=total_records,
            records_by_classification=records_by_classification,
            retention_compliance_rate=retention_compliance_rate,
            data_deletion_requests=len([e for e in self._audit_events if e['event_type'] == 'data_deletion_request']),
            audit_events=len(self._audit_events)
        )
    
    def _calculate_compliance_score(
        self, 
        metrics: ComplianceMetrics, 
        violations: List[ComplianceViolation]
    ) -> float:
        """Calculate overall compliance score."""
        base_score = 100.0
        
        # Deduct points for violations
        for violation in violations:
            if violation.severity == "critical":
                base_score -= 10.0
            elif violation.severity == "high":
                base_score -= 5.0
            elif violation.severity == "medium":
                base_score -= 2.0
            else:
                base_score -= 1.0
        
        # Add points for good practices
        if metrics.retention_compliance_rate > 95.0:
            base_score += 5.0
        
        return max(0.0, min(100.0, base_score))
    
    def _generate_recommendations(
        self, 
        metrics: ComplianceMetrics, 
        violations: List[ComplianceViolation]
    ) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        # Violation-based recommendations
        critical_violations = [v for v in violations if v.severity == "critical"]
        if critical_violations:
            recommendations.append("Address critical compliance violations immediately")
        
        high_violations = [v for v in violations if v.severity == "high"]
        if high_violations:
            recommendations.append("Implement fixes for high-severity violations")
        
        # Retention-based recommendations
        if metrics.retention_compliance_rate < 90.0:
            recommendations.append("Improve data retention compliance procedures")
        
        # General recommendations
        if metrics.total_violations > 0:
            recommendations.append("Implement automated compliance monitoring")
            recommendations.append("Provide compliance training for development team")
        
        if not recommendations:
            recommendations.append("Maintain current compliance standards")
        
        return recommendations
    
    def _create_remediation_plan(self, violations: List[ComplianceViolation]) -> List[Dict[str, Any]]:
        """Create remediation plan for violations."""
        plan = []
        
        # Group violations by type
        violations_by_type = {}
        for violation in violations:
            violation_type = violation.violation_type
            if violation_type not in violations_by_type:
                violations_by_type[violation_type] = []
            violations_by_type[violation_type].append(violation)
        
        # Create remediation items
        for violation_type, type_violations in violations_by_type.items():
            critical_count = len([v for v in type_violations if v.severity == "critical"])
            high_count = len([v for v in type_violations if v.severity == "high"])
            
            priority = "critical" if critical_count > 0 else "high" if high_count > 0 else "medium"
            
            plan.append({
                "violation_type": violation_type,
                "priority": priority,
                "violation_count": len(type_violations),
                "estimated_effort": "high" if len(type_violations) > 5 else "medium",
                "target_completion": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
                "remediation_steps": type_violations[0].remediation_steps if type_violations else []
            })
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        plan.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return plan
    
    def _generate_retention_summary(self) -> Dict[str, Any]:
        """Generate data retention summary."""
        with self._lock:
            records = list(self._retention_records.values())
        
        total_records = len(records)
        deleted_records = len([r for r in records if r.deleted])
        expired_records = len([r for r in records if r.retention_until < datetime.now(timezone.utc)])
        
        return {
            "total_records": total_records,
            "deleted_records": deleted_records,
            "expired_records": expired_records,
            "active_records": total_records - deleted_records,
            "compliance_rate": (deleted_records / expired_records * 100) if expired_records > 0 else 100.0
        }
    
    def _generate_classification_summary(self) -> Dict[str, Any]:
        """Generate data classification summary."""
        with self._lock:
            records = list(self._retention_records.values())
        
        classification_counts = {}
        for record in records:
            if not record.deleted:
                classification = record.classification.value
                classification_counts[classification] = classification_counts.get(classification, 0) + 1
        
        return classification_counts
    
    def _log_audit_event(
        self,
        event_type: str,
        agent_id: str = None,
        session_id: str = None,
        user_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Log an audit event."""
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc),
            "event_type": event_type,
            "agent_id": agent_id,
            "session_id": session_id,
            "user_id": user_id,
            "metadata": metadata or {}
        }
        
        with self._lock:
            self._audit_events.append(event)
            
            # Maintain audit log size
            if len(self._audit_events) > 10000:
                self._audit_events = self._audit_events[-5000:]  # Keep last 5000 events
    
    def _load_default_compliance_rules(self) -> None:
        """Load default compliance rules for supported frameworks."""
        
        # GDPR rules
        if ComplianceFramework.GDPR in self.frameworks:
            gdpr_rules = [
                ComplianceRule(
                    rule_id="gdpr_consent",
                    framework=ComplianceFramework.GDPR,
                    category="consent_management",
                    description="Ensure valid consent for data processing",
                    severity="high"
                ),
                ComplianceRule(
                    rule_id="gdpr_data_minimization",
                    framework=ComplianceFramework.GDPR,
                    category="data_protection",
                    description="Process only necessary personal data",
                    severity="medium"
                ),
                ComplianceRule(
                    rule_id="gdpr_right_to_deletion",
                    framework=ComplianceFramework.GDPR,
                    category="data_subject_rights",
                    description="Honor data deletion requests",
                    severity="high"
                )
            ]
            
            for rule in gdpr_rules:
                self._compliance_rules[rule.rule_id] = rule
        
        print(f"   Loaded {len(self._compliance_rules)} default compliance rules")
    
    def _start_retention_cleanup_thread(self) -> None:
        """Start background thread for data retention cleanup."""
        def cleanup_loop():
            while self._running:
                try:
                    self._cleanup_expired_data()
                    time.sleep(3600)  # Run every hour
                except Exception as e:
                    print(f"⚠️  Error in retention cleanup: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
        
        self._retention_cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._retention_cleanup_thread.start()
    
    def _start_metrics_thread(self) -> None:
        """Start background thread for metrics collection."""
        def metrics_loop():
            while self._running:
                try:
                    # Generate periodic compliance reports
                    for framework in self.frameworks:
                        self.generate_compliance_report(framework, period_days=1)
                    
                    time.sleep(86400)  # Run daily
                    
                except Exception as e:
                    print(f"⚠️  Error in metrics collection: {e}")
                    time.sleep(3600)  # Wait 1 hour before retrying
        
        self._metrics_thread = threading.Thread(target=metrics_loop, daemon=True)
        self._metrics_thread.start()
    
    def _cleanup_expired_data(self) -> None:
        """Clean up expired data records."""
        current_time = datetime.now(timezone.utc)
        cleanup_count = 0
        
        with self._lock:
            for record in self._retention_records.values():
                if not record.deleted and record.retention_until <= current_time:
                    record.deleted = True
                    record.deletion_timestamp = current_time
                    cleanup_count += 1
        
        if cleanup_count > 0:
            print(f"🧹 Cleaned up {cleanup_count} expired data records")
            
            # Log audit event
            self._log_audit_event(
                event_type="automatic_data_cleanup",
                metadata={"records_cleaned": cleanup_count}
            )





