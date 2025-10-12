"""
Security Analyzers

Advanced analysis modules for security monitoring:
- ContentAnalyzer: Analyzes content for security risks
- RiskAssessment: Calculates and manages risk scores
- BehaviorAnalyzer: Analyzes behavioral patterns
"""

import re
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import json


@dataclass
class AnalysisResult:
    """Result of a security analysis."""
    analyzer_name: str
    analysis_type: str
    risk_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    findings: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RiskFactor:
    """Represents a risk factor in assessment."""
    factor_id: str
    category: str
    description: str
    weight: float  # 0.0 to 1.0
    current_value: float  # 0.0 to 1.0
    threshold: float = 0.5
    trend: str = "stable"  # "increasing", "decreasing", "stable"


@dataclass
class BehaviorPattern:
    """Represents a behavioral pattern."""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    first_seen: datetime
    last_seen: datetime
    risk_level: str  # "low", "medium", "high", "critical"
    indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecurityAnalyzer(ABC):
    """Base class for security analyzers."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self._analysis_count = 0
        self._last_analysis = None
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> AnalysisResult:
        """Perform security analysis on the given data."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "analysis_count": self._analysis_count,
            "last_analysis": self._last_analysis.isoformat() if self._last_analysis else None
        }


class ContentAnalyzer(SecurityAnalyzer):
    """
    Analyzes content for security risks including sensitive information,
    malicious patterns, and policy violations.
    """
    
    def __init__(self, enabled: bool = True, privacy_mode: bool = False):
        super().__init__("ContentAnalyzer", enabled)
        self.privacy_mode = privacy_mode
        
        # Load analysis patterns
        self._sensitive_patterns = self._load_sensitive_patterns()
        self._malicious_patterns = self._load_malicious_patterns()
        self._policy_patterns = self._load_policy_patterns()
        
        # Content classification
        self._classification_rules = self._load_classification_rules()
        
        # Statistics
        self._content_analyzed = 0
        self._sensitive_content_detected = 0
        self._malicious_content_detected = 0
    
    def analyze(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> AnalysisResult:
        """Analyze content for security risks."""
        if not self.enabled:
            return AnalysisResult(
                analyzer_name=self.name,
                analysis_type="content_security",
                risk_score=0.0,
                confidence=1.0,
                findings=[],
                recommendations=[]
            )
        
        self._analysis_count += 1
        self._last_analysis = datetime.now(timezone.utc)
        self._content_analyzed += 1
        
        # Extract content
        user_input = data.get('user_input', '')
        agent_response = data.get('agent_response', '')
        
        risk_score = 0.0
        findings = []
        recommendations = []
        metadata = {}
        
        # Analyze user input
        if user_input:
            input_analysis = self._analyze_text_content(user_input, "user_input")
            risk_score = max(risk_score, input_analysis['risk_score'])
            findings.extend(input_analysis['findings'])
            recommendations.extend(input_analysis['recommendations'])
            metadata['input_analysis'] = input_analysis['metadata']
        
        # Analyze agent response
        if agent_response:
            response_analysis = self._analyze_text_content(agent_response, "agent_response")
            risk_score = max(risk_score, response_analysis['risk_score'])
            findings.extend(response_analysis['findings'])
            recommendations.extend(response_analysis['recommendations'])
            metadata['response_analysis'] = response_analysis['metadata']
        
        # Cross-reference analysis
        if user_input and agent_response:
            cross_analysis = self._analyze_interaction_context(user_input, agent_response)
            risk_score = max(risk_score, cross_analysis['risk_score'])
            findings.extend(cross_analysis['findings'])
            recommendations.extend(cross_analysis['recommendations'])
            metadata['cross_analysis'] = cross_analysis['metadata']
        
        # Calculate confidence
        confidence = min(0.9, len(findings) * 0.15 + 0.4)
        
        # Update statistics
        if risk_score > 0.5:
            if any('sensitive' in finding for finding in findings):
                self._sensitive_content_detected += 1
            if any('malicious' in finding for finding in findings):
                self._malicious_content_detected += 1
        
        return AnalysisResult(
            analyzer_name=self.name,
            analysis_type="content_security",
            risk_score=risk_score,
            confidence=confidence,
            findings=findings,
            recommendations=recommendations,
            metadata=metadata
        )
    
    def _analyze_text_content(self, text: str, content_type: str) -> Dict[str, Any]:
        """Analyze individual text content."""
        risk_score = 0.0
        findings = []
        recommendations = []
        metadata = {
            'content_type': content_type,
            'content_length': len(text),
            'word_count': len(text.split())
        }
        
        # Sensitive information detection
        sensitive_risk, sensitive_findings = self._detect_sensitive_information(text)
        if sensitive_risk > 0.3:
            risk_score = max(risk_score, sensitive_risk)
            findings.extend([f"sensitive_info_{f}" for f in sensitive_findings])
            recommendations.append("Review and redact sensitive information")
        
        # Malicious content detection
        malicious_risk, malicious_findings = self._detect_malicious_content(text)
        if malicious_risk > 0.4:
            risk_score = max(risk_score, malicious_risk)
            findings.extend([f"malicious_content_{f}" for f in malicious_findings])
            recommendations.append("Block or sanitize malicious content")
        
        # Policy violation detection
        policy_risk, policy_findings = self._detect_policy_violations(text)
        if policy_risk > 0.3:
            risk_score = max(risk_score, policy_risk)
            findings.extend([f"policy_violation_{f}" for f in policy_findings])
            recommendations.append("Review content against policy guidelines")
        
        # Content classification
        classification = self._classify_content(text)
        metadata['classification'] = classification
        
        if classification in ['restricted', 'confidential']:
            risk_score = max(risk_score, 0.6)
            findings.append(f"classified_content_{classification}")
            recommendations.append("Apply appropriate access controls")
        
        return {
            'risk_score': risk_score,
            'findings': findings,
            'recommendations': recommendations,
            'metadata': metadata
        }
    
    def _analyze_interaction_context(self, user_input: str, agent_response: str) -> Dict[str, Any]:
        """Analyze the interaction context between input and response."""
        risk_score = 0.0
        findings = []
        recommendations = []
        metadata = {}
        
        # Check for information leakage
        leakage_risk = self._detect_information_leakage(user_input, agent_response)
        if leakage_risk > 0.4:
            risk_score = max(risk_score, leakage_risk)
            findings.append("potential_information_leakage")
            recommendations.append("Review response for unintended information disclosure")
        
        # Check for inappropriate responses
        inappropriate_risk = self._detect_inappropriate_response(user_input, agent_response)
        if inappropriate_risk > 0.5:
            risk_score = max(risk_score, inappropriate_risk)
            findings.append("inappropriate_response")
            recommendations.append("Implement response filtering and validation")
        
        # Check response relevance and safety
        relevance_score = self._calculate_response_relevance(user_input, agent_response)
        if relevance_score < 0.3:
            risk_score = max(risk_score, 0.4)
            findings.append("low_response_relevance")
            recommendations.append("Improve response relevance and coherence")
        
        metadata['leakage_risk'] = leakage_risk
        metadata['inappropriate_risk'] = inappropriate_risk
        metadata['relevance_score'] = relevance_score
        
        return {
            'risk_score': risk_score,
            'findings': findings,
            'recommendations': recommendations,
            'metadata': metadata
        }
    
    def _detect_sensitive_information(self, text: str) -> Tuple[float, List[str]]:
        """Detect sensitive information in text."""
        risk_score = 0.0
        findings = []
        
        for pattern_name, pattern_info in self._sensitive_patterns.items():
            pattern = pattern_info['pattern']
            weight = pattern_info['weight']
            
            if re.search(pattern, text, re.IGNORECASE):
                risk_score = max(risk_score, weight)
                findings.append(pattern_name)
        
        return risk_score, findings
    
    def _detect_malicious_content(self, text: str) -> Tuple[float, List[str]]:
        """Detect malicious content patterns."""
        risk_score = 0.0
        findings = []
        
        for pattern_name, pattern_info in self._malicious_patterns.items():
            pattern = pattern_info['pattern']
            weight = pattern_info['weight']
            
            if re.search(pattern, text, re.IGNORECASE):
                risk_score = max(risk_score, weight)
                findings.append(pattern_name)
        
        return risk_score, findings
    
    def _detect_policy_violations(self, text: str) -> Tuple[float, List[str]]:
        """Detect policy violations in content."""
        risk_score = 0.0
        findings = []
        
        for policy_name, policy_info in self._policy_patterns.items():
            pattern = policy_info['pattern']
            weight = policy_info['weight']
            
            if re.search(pattern, text, re.IGNORECASE):
                risk_score = max(risk_score, weight)
                findings.append(policy_name)
        
        return risk_score, findings
    
    def _classify_content(self, text: str) -> str:
        """Classify content based on sensitivity level."""
        for classification, rules in self._classification_rules.items():
            for rule in rules:
                if re.search(rule, text, re.IGNORECASE):
                    return classification
        
        return "public"
    
    def _detect_information_leakage(self, user_input: str, agent_response: str) -> float:
        """Detect potential information leakage from input to response."""
        # Simple implementation - check if response contains patterns from input
        # that might indicate system information leakage
        
        risk_score = 0.0
        
        # Check for system information in response
        system_indicators = ['error', 'exception', 'stack trace', 'debug', 'internal']
        for indicator in system_indicators:
            if indicator.lower() in agent_response.lower():
                risk_score = max(risk_score, 0.6)
        
        # Check for unexpected information disclosure
        if len(agent_response) > len(user_input) * 3:  # Response much longer than input
            risk_score = max(risk_score, 0.3)
        
        return risk_score
    
    def _detect_inappropriate_response(self, user_input: str, agent_response: str) -> float:
        """Detect inappropriate responses to user input."""
        risk_score = 0.0
        
        # Check for harmful content in response
        harmful_patterns = [
            r'\b(violence|harm|illegal|dangerous)\b',
            r'\b(hack|crack|exploit|vulnerability)\b',
            r'\b(steal|fraud|scam|cheat)\b'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, agent_response, re.IGNORECASE):
                risk_score = max(risk_score, 0.7)
        
        return risk_score
    
    def _calculate_response_relevance(self, user_input: str, agent_response: str) -> float:
        """Calculate relevance score between input and response."""
        if not user_input or not agent_response:
            return 0.0
        
        # Simple word overlap calculation
        input_words = set(user_input.lower().split())
        response_words = set(agent_response.lower().split())
        
        if not input_words:
            return 0.0
        
        overlap = input_words.intersection(response_words)
        relevance = len(overlap) / len(input_words)
        
        return min(1.0, relevance)
    
    def _load_sensitive_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load sensitive information detection patterns."""
        return {
            'ssn': {'pattern': r'\b\d{3}-\d{2}-\d{4}\b', 'weight': 0.9},
            'credit_card': {'pattern': r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b', 'weight': 0.9},
            'email': {'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'weight': 0.6},
            'phone': {'pattern': r'\b\d{3}-\d{3}-\d{4}\b', 'weight': 0.5},
            'ip_address': {'pattern': r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'weight': 0.4},
            'api_key': {'pattern': r'\b[A-Za-z0-9]{32,}\b', 'weight': 0.8},
        }
    
    def _load_malicious_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load malicious content detection patterns."""
        return {
            'sql_injection': {'pattern': r'(union|select|insert|update|delete|drop)\s+', 'weight': 0.8},
            'xss_attempt': {'pattern': r'<script|javascript:|onload=|onerror=', 'weight': 0.8},
            'command_injection': {'pattern': r'(;|\|&|&&|\|\|)\s*(rm|del|format|shutdown)', 'weight': 0.9},
            'path_traversal': {'pattern': r'\.\.[\\/]', 'weight': 0.7},
            'malicious_url': {'pattern': r'https?://[^\s]*\.(tk|ml|ga|cf)/', 'weight': 0.6},
        }
    
    def _load_policy_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load policy violation detection patterns."""
        return {
            'hate_speech': {'pattern': r'\b(hate|discriminat|racist|sexist)\b', 'weight': 0.7},
            'harassment': {'pattern': r'\b(threaten|intimidat|bully|harass)\b', 'weight': 0.6},
            'inappropriate_language': {'pattern': r'\b(profanity|explicit|offensive)\b', 'weight': 0.4},
            'privacy_violation': {'pattern': r'\b(personal|private|confidential)\s+information\b', 'weight': 0.5},
        }
    
    def _load_classification_rules(self) -> Dict[str, List[str]]:
        """Load content classification rules."""
        return {
            'secret': [r'\b(secret|classified|top.secret)\b'],
            'confidential': [r'\b(confidential|internal.only|restricted)\b'],
            'restricted': [r'\b(restricted|limited.access|sensitive)\b'],
            'internal': [r'\b(internal|company.only|proprietary)\b'],
            'public': [r'\b(public|open|general)\b']
        }


class RiskAssessment(SecurityAnalyzer):
    """
    Comprehensive risk assessment and scoring system.
    
    Calculates risk scores based on multiple factors and maintains
    risk profiles for agents, sessions, and users.
    """
    
    def __init__(self, enabled: bool = True):
        super().__init__("RiskAssessment", enabled)
        
        # Risk factors
        self._risk_factors: Dict[str, RiskFactor] = {}
        self._agent_risk_profiles: Dict[str, Dict[str, Any]] = {}
        self._session_risk_profiles: Dict[str, Dict[str, Any]] = {}
        self._user_risk_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Risk calculation weights
        self._factor_weights = self._load_factor_weights()
        
        # Initialize default risk factors
        self._initialize_risk_factors()
    
    def analyze(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> AnalysisResult:
        """Perform comprehensive risk assessment."""
        if not self.enabled:
            return AnalysisResult(
                analyzer_name=self.name,
                analysis_type="risk_assessment",
                risk_score=0.0,
                confidence=1.0,
                findings=[],
                recommendations=[]
            )
        
        self._analysis_count += 1
        self._last_analysis = datetime.now(timezone.utc)
        
        agent_id = data.get('agent_id')
        session_id = data.get('session_id')
        user_id = data.get('user_id')
        
        # Calculate component risk scores
        content_risk = self._calculate_content_risk(data)
        behavioral_risk = self._calculate_behavioral_risk(data, context)
        contextual_risk = self._calculate_contextual_risk(data, context)
        historical_risk = self._calculate_historical_risk(agent_id, session_id, user_id)
        
        # Combine risk scores
        overall_risk = self._combine_risk_scores([
            ('content', content_risk, self._factor_weights.get('content', 0.3)),
            ('behavioral', behavioral_risk, self._factor_weights.get('behavioral', 0.25)),
            ('contextual', contextual_risk, self._factor_weights.get('contextual', 0.25)),
            ('historical', historical_risk, self._factor_weights.get('historical', 0.2))
        ])
        
        # Update risk profiles
        self._update_risk_profiles(agent_id, session_id, user_id, overall_risk)
        
        # Generate findings and recommendations
        findings = self._generate_risk_findings(overall_risk, content_risk, behavioral_risk, contextual_risk, historical_risk)
        recommendations = self._generate_risk_recommendations(overall_risk, findings)
        
        # Calculate confidence
        confidence = self._calculate_risk_confidence(data, context)
        
        return AnalysisResult(
            analyzer_name=self.name,
            analysis_type="risk_assessment",
            risk_score=overall_risk,
            confidence=confidence,
            findings=findings,
            recommendations=recommendations,
            metadata={
                'component_risks': {
                    'content': content_risk,
                    'behavioral': behavioral_risk,
                    'contextual': contextual_risk,
                    'historical': historical_risk
                },
                'risk_factors': {k: v.current_value for k, v in self._risk_factors.items()}
            }
        )
    
    def get_risk_profile(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Get risk profile for agent, session, or user."""
        profiles = {
            'agent': self._agent_risk_profiles,
            'session': self._session_risk_profiles,
            'user': self._user_risk_profiles
        }
        
        return profiles.get(entity_type, {}).get(entity_id, {})
    
    def _calculate_content_risk(self, data: Dict[str, Any]) -> float:
        """Calculate risk based on content analysis."""
        # This would integrate with ContentAnalyzer results
        user_input = data.get('user_input', '')
        agent_response = data.get('agent_response', '')
        
        risk_score = 0.0
        
        # Simple content risk heuristics
        if len(user_input) > 1000:  # Very long input
            risk_score = max(risk_score, 0.3)
        
        if len(agent_response) > 2000:  # Very long response
            risk_score = max(risk_score, 0.2)
        
        # Check for suspicious patterns
        suspicious_patterns = ['system', 'admin', 'password', 'secret', 'hack']
        for pattern in suspicious_patterns:
            if pattern in user_input.lower():
                risk_score = max(risk_score, 0.4)
        
        return min(1.0, risk_score)
    
    def _calculate_behavioral_risk(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> float:
        """Calculate risk based on behavioral patterns."""
        if not context:
            return 0.0
        
        risk_score = 0.0
        
        # Check interaction frequency
        interaction_count = context.get('interaction_count', 0)
        if interaction_count > 100:  # High frequency
            risk_score = max(risk_score, 0.4)
        
        # Check for rapid interactions
        time_since_last = context.get('time_since_last_interaction', 0)
        if time_since_last < 1:  # Less than 1 second
            risk_score = max(risk_score, 0.6)
        
        # Check for pattern variations
        pattern_diversity = context.get('pattern_diversity', 1.0)
        if pattern_diversity < 0.3:  # Low diversity (repetitive)
            risk_score = max(risk_score, 0.5)
        
        return min(1.0, risk_score)
    
    def _calculate_contextual_risk(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> float:
        """Calculate risk based on contextual factors."""
        if not context:
            return 0.0
        
        risk_score = 0.0
        
        # Time-based risk
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Outside normal hours
            risk_score = max(risk_score, 0.2)
        
        # Session duration risk
        session_duration = context.get('session_duration_minutes', 0)
        if session_duration > 120:  # Very long session
            risk_score = max(risk_score, 0.3)
        
        # Geographic risk (if available)
        if context.get('unusual_location', False):
            risk_score = max(risk_score, 0.4)
        
        return min(1.0, risk_score)
    
    def _calculate_historical_risk(self, agent_id: str, session_id: str, user_id: str) -> float:
        """Calculate risk based on historical patterns."""
        risk_score = 0.0
        
        # Agent historical risk
        if agent_id and agent_id in self._agent_risk_profiles:
            agent_profile = self._agent_risk_profiles[agent_id]
            avg_risk = agent_profile.get('average_risk', 0.0)
            incident_count = agent_profile.get('incident_count', 0)
            
            risk_score = max(risk_score, avg_risk * 0.5)
            if incident_count > 5:
                risk_score = max(risk_score, 0.4)
        
        # User historical risk
        if user_id and user_id in self._user_risk_profiles:
            user_profile = self._user_risk_profiles[user_id]
            user_avg_risk = user_profile.get('average_risk', 0.0)
            user_incidents = user_profile.get('incident_count', 0)
            
            risk_score = max(risk_score, user_avg_risk * 0.3)
            if user_incidents > 3:
                risk_score = max(risk_score, 0.5)
        
        return min(1.0, risk_score)
    
    def _combine_risk_scores(self, risk_components: List[Tuple[str, float, float]]) -> float:
        """Combine multiple risk scores using weighted average."""
        total_weight = sum(weight for _, _, weight in risk_components)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(score * weight for _, score, weight in risk_components)
        return min(1.0, weighted_sum / total_weight)
    
    def _update_risk_profiles(self, agent_id: str, session_id: str, user_id: str, risk_score: float) -> None:
        """Update risk profiles with new risk score."""
        current_time = datetime.now(timezone.utc)
        
        # Update agent profile
        if agent_id:
            if agent_id not in self._agent_risk_profiles:
                self._agent_risk_profiles[agent_id] = {
                    'risk_history': deque(maxlen=100),
                    'average_risk': 0.0,
                    'max_risk': 0.0,
                    'incident_count': 0,
                    'last_updated': current_time
                }
            
            profile = self._agent_risk_profiles[agent_id]
            profile['risk_history'].append((current_time, risk_score))
            profile['average_risk'] = sum(score for _, score in profile['risk_history']) / len(profile['risk_history'])
            profile['max_risk'] = max(profile['max_risk'], risk_score)
            profile['last_updated'] = current_time
            
            if risk_score > 0.7:
                profile['incident_count'] += 1
        
        # Update session profile
        if session_id:
            if session_id not in self._session_risk_profiles:
                self._session_risk_profiles[session_id] = {
                    'risk_history': [],
                    'average_risk': 0.0,
                    'max_risk': 0.0,
                    'created': current_time,
                    'last_updated': current_time
                }
            
            profile = self._session_risk_profiles[session_id]
            profile['risk_history'].append((current_time, risk_score))
            profile['average_risk'] = sum(score for _, score in profile['risk_history']) / len(profile['risk_history'])
            profile['max_risk'] = max(profile['max_risk'], risk_score)
            profile['last_updated'] = current_time
        
        # Update user profile
        if user_id:
            if user_id not in self._user_risk_profiles:
                self._user_risk_profiles[user_id] = {
                    'risk_history': deque(maxlen=200),
                    'average_risk': 0.0,
                    'max_risk': 0.0,
                    'incident_count': 0,
                    'first_seen': current_time,
                    'last_updated': current_time
                }
            
            profile = self._user_risk_profiles[user_id]
            profile['risk_history'].append((current_time, risk_score))
            profile['average_risk'] = sum(score for _, score in profile['risk_history']) / len(profile['risk_history'])
            profile['max_risk'] = max(profile['max_risk'], risk_score)
            profile['last_updated'] = current_time
            
            if risk_score > 0.7:
                profile['incident_count'] += 1
    
    def _generate_risk_findings(self, overall_risk: float, content_risk: float, behavioral_risk: float, 
                               contextual_risk: float, historical_risk: float) -> List[str]:
        """Generate risk findings based on component scores."""
        findings = []
        
        if overall_risk > 0.8:
            findings.append("critical_risk_level_detected")
        elif overall_risk > 0.6:
            findings.append("high_risk_level_detected")
        elif overall_risk > 0.4:
            findings.append("medium_risk_level_detected")
        
        if content_risk > 0.6:
            findings.append("high_content_risk")
        
        if behavioral_risk > 0.6:
            findings.append("suspicious_behavioral_patterns")
        
        if contextual_risk > 0.5:
            findings.append("elevated_contextual_risk")
        
        if historical_risk > 0.5:
            findings.append("concerning_historical_patterns")
        
        return findings
    
    def _generate_risk_recommendations(self, risk_score: float, findings: List[str]) -> List[str]:
        """Generate risk-based recommendations."""
        recommendations = []
        
        if risk_score > 0.8:
            recommendations.append("Implement immediate security controls")
            recommendations.append("Review interaction for policy violations")
        
        if risk_score > 0.6:
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Consider additional verification steps")
        
        if "high_content_risk" in findings:
            recommendations.append("Implement content filtering")
        
        if "suspicious_behavioral_patterns" in findings:
            recommendations.append("Analyze behavioral patterns for anomalies")
        
        if not recommendations:
            recommendations.append("Continue standard monitoring")
        
        return recommendations
    
    def _calculate_risk_confidence(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> float:
        """Calculate confidence in risk assessment."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence with more data
        if data.get('user_input'):
            confidence += 0.1
        if data.get('agent_response'):
            confidence += 0.1
        if context:
            confidence += 0.2
        
        # Historical data improves confidence
        agent_id = data.get('agent_id')
        if agent_id and agent_id in self._agent_risk_profiles:
            history_length = len(self._agent_risk_profiles[agent_id]['risk_history'])
            confidence += min(0.2, history_length * 0.01)
        
        return min(0.95, confidence)
    
    def _initialize_risk_factors(self) -> None:
        """Initialize default risk factors."""
        factors = [
            RiskFactor("content_sensitivity", "content", "Sensitivity of content being processed", 0.3, 0.0),
            RiskFactor("behavioral_anomaly", "behavior", "Anomalous behavioral patterns", 0.25, 0.0),
            RiskFactor("contextual_risk", "context", "Contextual risk factors", 0.2, 0.0),
            RiskFactor("historical_incidents", "history", "Historical security incidents", 0.25, 0.0),
        ]
        
        for factor in factors:
            self._risk_factors[factor.factor_id] = factor
    
    def _load_factor_weights(self) -> Dict[str, float]:
        """Load risk factor weights."""
        return {
            'content': 0.3,
            'behavioral': 0.25,
            'contextual': 0.25,
            'historical': 0.2
        }


class BehaviorAnalyzer(SecurityAnalyzer):
    """
    Analyzes behavioral patterns in agent interactions.
    
    Identifies anomalous behaviors, usage patterns, and potential
    security threats based on interaction patterns.
    """
    
    def __init__(self, enabled: bool = True, pattern_window: int = 50):
        super().__init__("BehaviorAnalyzer", enabled)
        self.pattern_window = pattern_window
        
        # Behavior tracking
        self._behavior_patterns: Dict[str, BehaviorPattern] = {}
        self._agent_behaviors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._session_behaviors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._user_behaviors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Pattern detection
        self._pattern_detectors = self._initialize_pattern_detectors()
        
        # Statistics
        self._patterns_detected = 0
        self._anomalies_detected = 0
    
    def analyze(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> AnalysisResult:
        """Analyze behavioral patterns."""
        if not self.enabled:
            return AnalysisResult(
                analyzer_name=self.name,
                analysis_type="behavior_analysis",
                risk_score=0.0,
                confidence=1.0,
                findings=[],
                recommendations=[]
            )
        
        self._analysis_count += 1
        self._last_analysis = datetime.now(timezone.utc)
        
        agent_id = data.get('agent_id')
        session_id = data.get('session_id')
        user_id = data.get('user_id')
        
        # Record behavior
        behavior_record = self._create_behavior_record(data, context)
        self._record_behavior(agent_id, session_id, user_id, behavior_record)
        
        # Analyze patterns
        risk_score = 0.0
        findings = []
        recommendations = []
        
        # Pattern detection
        pattern_results = self._detect_patterns(agent_id, session_id, user_id)
        if pattern_results['risk_score'] > 0.3:
            risk_score = max(risk_score, pattern_results['risk_score'])
            findings.extend(pattern_results['findings'])
            recommendations.extend(pattern_results['recommendations'])
        
        # Anomaly detection
        anomaly_results = self._detect_anomalies(agent_id, session_id, user_id, behavior_record)
        if anomaly_results['risk_score'] > 0.3:
            risk_score = max(risk_score, anomaly_results['risk_score'])
            findings.extend(anomaly_results['findings'])
            recommendations.extend(anomaly_results['recommendations'])
        
        # Trend analysis
        trend_results = self._analyze_trends(agent_id, session_id, user_id)
        if trend_results['risk_score'] > 0.2:
            risk_score = max(risk_score, trend_results['risk_score'])
            findings.extend(trend_results['findings'])
            recommendations.extend(trend_results['recommendations'])
        
        # Calculate confidence
        confidence = self._calculate_behavior_confidence(agent_id, session_id, user_id)
        
        return AnalysisResult(
            analyzer_name=self.name,
            analysis_type="behavior_analysis",
            risk_score=risk_score,
            confidence=confidence,
            findings=findings,
            recommendations=recommendations,
            metadata={
                'patterns_detected': self._patterns_detected,
                'anomalies_detected': self._anomalies_detected,
                'behavior_record': behavior_record
            }
        )
    
    def get_behavior_patterns(self, entity_type: str, entity_id: str) -> List[BehaviorPattern]:
        """Get behavior patterns for specific entity."""
        patterns = []
        for pattern in self._behavior_patterns.values():
            if (entity_type == 'agent' and pattern.pattern_id.startswith(f"agent_{entity_id}_")) or \
               (entity_type == 'session' and pattern.pattern_id.startswith(f"session_{entity_id}_")) or \
               (entity_type == 'user' and pattern.pattern_id.startswith(f"user_{entity_id}_")):
                patterns.append(pattern)
        return patterns
    
    def _create_behavior_record(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a behavior record from interaction data."""
        return {
            'timestamp': datetime.now(timezone.utc),
            'input_length': len(data.get('user_input', '')),
            'response_length': len(data.get('agent_response', '')),
            'interaction_type': self._classify_interaction_type(data),
            'complexity_score': self._calculate_complexity_score(data),
            'response_time': context.get('response_time', 0) if context else 0,
            'context_data': context or {}
        }
    
    def _record_behavior(self, agent_id: str, session_id: str, user_id: str, behavior_record: Dict[str, Any]) -> None:
        """Record behavior for tracking."""
        if agent_id:
            self._agent_behaviors[agent_id].append(behavior_record)
            if len(self._agent_behaviors[agent_id]) > self.pattern_window:
                self._agent_behaviors[agent_id] = self._agent_behaviors[agent_id][-self.pattern_window:]
        
        if session_id:
            self._session_behaviors[session_id].append(behavior_record)
            if len(self._session_behaviors[session_id]) > self.pattern_window:
                self._session_behaviors[session_id] = self._session_behaviors[session_id][-self.pattern_window:]
        
        if user_id:
            self._user_behaviors[user_id].append(behavior_record)
            if len(self._user_behaviors[user_id]) > self.pattern_window:
                self._user_behaviors[user_id] = self._user_behaviors[user_id][-self.pattern_window:]
    
    def _detect_patterns(self, agent_id: str, session_id: str, user_id: str) -> Dict[str, Any]:
        """Detect behavioral patterns."""
        risk_score = 0.0
        findings = []
        recommendations = []
        
        # Check for repetitive patterns
        if agent_id and len(self._agent_behaviors[agent_id]) >= 10:
            repetition_score = self._calculate_repetition_score(self._agent_behaviors[agent_id])
            if repetition_score > 0.7:
                risk_score = max(risk_score, 0.6)
                findings.append("high_repetition_pattern")
                recommendations.append("Investigate repetitive behavior patterns")
        
        # Check for escalation patterns
        if session_id and len(self._session_behaviors[session_id]) >= 5:
            escalation_score = self._calculate_escalation_score(self._session_behaviors[session_id])
            if escalation_score > 0.6:
                risk_score = max(risk_score, 0.7)
                findings.append("escalation_pattern_detected")
                recommendations.append("Monitor for potential security escalation")
        
        return {
            'risk_score': risk_score,
            'findings': findings,
            'recommendations': recommendations
        }
    
    def _detect_anomalies(self, agent_id: str, session_id: str, user_id: str, current_record: Dict[str, Any]) -> Dict[str, Any]:
        """Detect behavioral anomalies."""
        risk_score = 0.0
        findings = []
        recommendations = []
        
        # Anomaly detection based on historical patterns
        if agent_id and len(self._agent_behaviors[agent_id]) >= 20:
            anomaly_score = self._calculate_anomaly_score(self._agent_behaviors[agent_id], current_record)
            if anomaly_score > 0.6:
                risk_score = max(risk_score, 0.5)
                findings.append("behavioral_anomaly_detected")
                recommendations.append("Investigate unusual behavioral patterns")
                self._anomalies_detected += 1
        
        return {
            'risk_score': risk_score,
            'findings': findings,
            'recommendations': recommendations
        }
    
    def _analyze_trends(self, agent_id: str, session_id: str, user_id: str) -> Dict[str, Any]:
        """Analyze behavioral trends."""
        risk_score = 0.0
        findings = []
        recommendations = []
        
        # Analyze trends in complexity and response times
        if agent_id and len(self._agent_behaviors[agent_id]) >= 10:
            behaviors = self._agent_behaviors[agent_id]
            complexity_trend = self._calculate_trend([b['complexity_score'] for b in behaviors[-10:]])
            
            if complexity_trend > 0.5:  # Increasing complexity
                risk_score = max(risk_score, 0.3)
                findings.append("increasing_complexity_trend")
                recommendations.append("Monitor for potential manipulation attempts")
        
        return {
            'risk_score': risk_score,
            'findings': findings,
            'recommendations': recommendations
        }
    
    def _classify_interaction_type(self, data: Dict[str, Any]) -> str:
        """Classify the type of interaction."""
        user_input = data.get('user_input', '').lower()
        
        if any(word in user_input for word in ['help', 'assist', 'support']):
            return 'help_request'
        elif any(word in user_input for word in ['how', 'what', 'why', 'when', 'where']):
            return 'information_request'
        elif any(word in user_input for word in ['do', 'create', 'make', 'generate']):
            return 'action_request'
        else:
            return 'general_conversation'
    
    def _calculate_complexity_score(self, data: Dict[str, Any]) -> float:
        """Calculate complexity score for interaction."""
        user_input = data.get('user_input', '')
        
        # Simple complexity metrics
        word_count = len(user_input.split())
        unique_words = len(set(user_input.lower().split()))
        avg_word_length = sum(len(word) for word in user_input.split()) / word_count if word_count > 0 else 0
        
        # Normalize to 0-1 scale
        complexity = min(1.0, (word_count * 0.01) + (unique_words * 0.02) + (avg_word_length * 0.1))
        return complexity
    
    def _calculate_repetition_score(self, behaviors: List[Dict[str, Any]]) -> float:
        """Calculate repetition score for behavior sequence."""
        if len(behaviors) < 2:
            return 0.0
        
        # Check for similar patterns
        similar_count = 0
        for i in range(1, len(behaviors)):
            current = behaviors[i]
            previous = behaviors[i-1]
            
            # Simple similarity check
            if (abs(current['input_length'] - previous['input_length']) < 10 and
                current['interaction_type'] == previous['interaction_type']):
                similar_count += 1
        
        return similar_count / (len(behaviors) - 1)
    
    def _calculate_escalation_score(self, behaviors: List[Dict[str, Any]]) -> float:
        """Calculate escalation score for behavior sequence."""
        if len(behaviors) < 3:
            return 0.0
        
        # Check for increasing complexity or length
        escalation_indicators = 0
        for i in range(2, len(behaviors)):
            if (behaviors[i]['complexity_score'] > behaviors[i-1]['complexity_score'] > behaviors[i-2]['complexity_score']):
                escalation_indicators += 1
        
        return escalation_indicators / max(1, len(behaviors) - 2)
    
    def _calculate_anomaly_score(self, historical_behaviors: List[Dict[str, Any]], current_record: Dict[str, Any]) -> float:
        """Calculate anomaly score for current behavior."""
        if not historical_behaviors:
            return 0.0
        
        # Calculate statistical baselines
        input_lengths = [b['input_length'] for b in historical_behaviors]
        complexity_scores = [b['complexity_score'] for b in historical_behaviors]
        
        avg_input_length = sum(input_lengths) / len(input_lengths)
        avg_complexity = sum(complexity_scores) / len(complexity_scores)
        
        # Calculate deviations
        input_deviation = abs(current_record['input_length'] - avg_input_length) / max(avg_input_length, 1)
        complexity_deviation = abs(current_record['complexity_score'] - avg_complexity) / max(avg_complexity, 0.1)
        
        # Combine deviations
        anomaly_score = min(1.0, (input_deviation + complexity_deviation) / 2)
        return anomaly_score
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction for a sequence of values."""
        if len(values) < 2:
            return 0.0
        
        # Simple trend calculation
        increases = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        return increases / (len(values) - 1)
    
    def _calculate_behavior_confidence(self, agent_id: str, session_id: str, user_id: str) -> float:
        """Calculate confidence in behavior analysis."""
        confidence = 0.3  # Base confidence
        
        # More historical data increases confidence
        if agent_id and agent_id in self._agent_behaviors:
            history_length = len(self._agent_behaviors[agent_id])
            confidence += min(0.4, history_length * 0.02)
        
        if session_id and session_id in self._session_behaviors:
            session_length = len(self._session_behaviors[session_id])
            confidence += min(0.2, session_length * 0.04)
        
        if user_id and user_id in self._user_behaviors:
            user_history = len(self._user_behaviors[user_id])
            confidence += min(0.3, user_history * 0.01)
        
        return min(0.9, confidence)
    
    def _initialize_pattern_detectors(self) -> Dict[str, Any]:
        """Initialize pattern detection algorithms."""
        return {
            'repetition_detector': {'threshold': 0.7, 'window_size': 10},
            'escalation_detector': {'threshold': 0.6, 'min_sequence': 3},
            'anomaly_detector': {'threshold': 0.6, 'min_history': 20}
        }





