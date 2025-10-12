"""
Security Detectors

Specialized detection modules for different types of security threats:
- PromptInjectionDetector: Detects prompt injection attempts
- JailbreakDetector: Detects jailbreak attempts
- SocialEngineeringDetector: Detects social engineering attempts
- ManipulationDetector: Detects manipulation patterns
- PIIDetector: Detects personally identifiable information disclosures
"""

import logging
import re
from typing import Callable, Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import hashlib


@dataclass
class DetectionResult:
    """Result of a security detection analysis."""
    detected: bool
    risk_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    threat_type: str
    description: str
    indicators: List[str]
    metadata: Dict[str, Any]


class SecurityDetector(ABC):
    """Base class for security detectors."""
    
    def __init__(
        self,
        name: str,
        enabled: bool = True,
        telemetry_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.name = name
        self.enabled = enabled
        self._telemetry_handler = telemetry_handler
        self._logger = logging.getLogger(f"security_monitor.{self.name}")
        self._detection_count = 0
        self._false_positive_count = 0
        self._true_positive_count = 0
        self._true_negative_count = 0
        self._false_negative_count = 0
        self._total_runs = 0
        self._last_detection_at: Optional[datetime] = None
    
    @abstractmethod
    def detect(self, text: str, context: Dict[str, Any] = None) -> DetectionResult:
        """Detect security threats in the given text."""
        pass
    
    def get_accuracy(self) -> float:
        """Get detection accuracy (placeholder - would need ground truth data)."""
        total = self._detection_count + self._false_positive_count
        if total == 0:
            return 1.0
        return (self._detection_count - self._false_positive_count) / total
    
    def reset_stats(self):
        """Reset detection statistics."""
        self._detection_count = 0
        self._false_positive_count = 0
        self._true_positive_count = 0
        self._true_negative_count = 0
        self._false_negative_count = 0
        self._total_runs = 0
        self._last_detection_at = None

    def record_feedback(
        self,
        *,
        detected: bool,
        correct: bool,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record human feedback about a detection outcome.
        
        Args:
            detected: Whether the detector flagged the content.
            correct: Whether the detection decision was correct.
            notes: Optional free-form notes from reviewers.
            metadata: Additional context to store alongside the feedback.
        """
        if detected and correct:
            self._true_positive_count += 1
        elif detected and not correct:
            self._false_positive_count += 1
        elif not detected and correct:
            self._true_negative_count += 1
        else:
            self._false_negative_count += 1
        
        feedback_event = {
            "type": "feedback",
            "detector": self.name,
            "detected": detected,
            "correct": correct,
            "notes": notes,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._emit_telemetry(feedback_event)

    def get_stats(self) -> Dict[str, Any]:
        """Return the current detector statistics."""
        return {
            "detector": self.name,
            "enabled": self.enabled,
            "detection_count": self._detection_count,
            "false_positive_count": self._false_positive_count,
            "true_positive_count": self._true_positive_count,
            "true_negative_count": self._true_negative_count,
            "false_negative_count": self._false_negative_count,
            "total_runs": self._total_runs,
            "accuracy": self.get_accuracy(),
            "last_detection_at": self._last_detection_at.isoformat() if self._last_detection_at else None,
        }

    def _finalize_detection(
        self,
        *,
        text: Optional[str],
        context: Optional[Dict[str, Any]],
        result: DetectionResult,
    ) -> DetectionResult:
        """Update counters and emit telemetry for a detection result."""
        self._total_runs += 1
        if result.detected:
            self._detection_count += 1
            self._last_detection_at = datetime.now(timezone.utc)
        
        event_payload = {
            "type": "detection_result",
            "detector": self.name,
            "result": asdict(result),
            "text_signature": self._fingerprint_text(text),
            "context_summary": self._summarize_context(context),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._emit_telemetry(event_payload)
        return result

    def _emit_telemetry(self, payload: Dict[str, Any]):
        """Emit telemetry payload to the configured handler or logger."""
        if not payload:
            return
        if self._telemetry_handler:
            try:
                self._telemetry_handler(payload)
                return
            except Exception:  # pragma: no cover - defensive logging
                self._logger.exception("Telemetry handler failed for %s", self.name)
        self._logger.debug("Detector telemetry: %s", payload)

    def _fingerprint_text(self, text: Optional[str]) -> Optional[str]:
        """Return a privacy-preserving fingerprint for auditing."""
        if not text:
            return None
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return digest[:16]

    def _summarize_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize context keys without logging sensitive values."""
        if not context:
            return {"keys": [], "size": 0}
        keys = sorted(context.keys())
        return {"keys": keys, "size": len(keys)}


class PromptInjectionDetector(SecurityDetector):
    """
    Detects prompt injection attempts in user input.
    
    Identifies attempts to manipulate AI behavior through crafted prompts
    that try to override system instructions or safety guidelines.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        sensitivity: float = 0.7,
        telemetry_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        super().__init__("PromptInjectionDetector", enabled, telemetry_handler)
        self.sensitivity = sensitivity
        
        # Load detection patterns
        self._injection_patterns = self._load_injection_patterns()
        self._system_keywords = self._load_system_keywords()
        self._instruction_verbs = self._load_instruction_verbs()
        
    def detect(self, text: str, context: Dict[str, Any] = None) -> DetectionResult:
        """Detect prompt injection attempts."""
        if not self.enabled or not text:
            return DetectionResult(
                detected=False, risk_score=0.0, confidence=1.0,
                threat_type="prompt_injection", description="No threat detected",
                indicators=[], metadata={}
            )
        
        text_lower = text.lower().strip()
        risk_score = 0.0
        indicators = []
        metadata = {}
        
        # Pattern-based detection
        pattern_risk, pattern_indicators = self._detect_patterns(text_lower)
        risk_score = max(risk_score, pattern_risk)
        indicators.extend(pattern_indicators)
        
        # Structural analysis
        structure_risk, structure_indicators = self._analyze_structure(text)
        risk_score = max(risk_score, structure_risk)
        indicators.extend(structure_indicators)
        
        # Context-based analysis
        if context:
            context_risk, context_indicators = self._analyze_context(text_lower, context)
            risk_score = max(risk_score, context_risk)
            indicators.extend(context_indicators)
        
        # Calculate confidence
        confidence = min(0.9, len(indicators) * 0.2 + 0.3)
        
        # Determine if detected
        detected = risk_score >= self.sensitivity
        
        description = self._generate_description(risk_score, indicators)
        result = DetectionResult(
            detected=detected,
            risk_score=risk_score,
            confidence=confidence,
            threat_type="prompt_injection",
            description=description,
            indicators=indicators,
            metadata={
                'pattern_matches': len(pattern_indicators),
                'structure_issues': len(structure_indicators),
                'sensitivity_threshold': self.sensitivity
            }
        )
        return self._finalize_detection(text=text, context=context, result=result)
    
    def _detect_patterns(self, text: str) -> Tuple[float, List[str]]:
        """Detect injection patterns in text."""
        risk_score = 0.0
        indicators = []
        
        for pattern, weight in self._injection_patterns.items():
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                risk_score = max(risk_score, weight)
                indicators.append(f"pattern_{pattern[:20]}...")
        
        return risk_score, indicators
    
    def _analyze_structure(self, text: str) -> Tuple[float, List[str]]:
        """Analyze text structure for injection indicators."""
        risk_score = 0.0
        indicators = []
        
        # Check for role-play indicators
        roleplay_patterns = [
            r'act\s+as\s+(if\s+)?you\s+(are|were)',
            r'pretend\s+(that\s+)?you\s+(are|were)',
            r'imagine\s+(that\s+)?you\s+(are|were)',
            r'roleplay\s+as',
            r'simulate\s+(being|a)'
        ]
        
        for pattern in roleplay_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                risk_score = max(risk_score, 0.6)
                indicators.append("roleplay_instruction")
                break
        
        # Check for system command simulation
        if re.search(r'(system|admin|root)\s*[:>]\s*', text, re.IGNORECASE):
            risk_score = max(risk_score, 0.8)
            indicators.append("system_command_simulation")
        
        # Check for instruction override attempts
        override_count = len(re.findall(r'\b(ignore|forget|disregard|override|bypass)\b', text, re.IGNORECASE))
        if override_count >= 2:
            risk_score = max(risk_score, 0.7)
            indicators.append(f"multiple_override_attempts_{override_count}")
        
        # Check for prompt template injection
        if '{{' in text and '}}' in text:
            risk_score = max(risk_score, 0.5)
            indicators.append("template_injection")
        
        return risk_score, indicators
    
    def _analyze_context(self, text: str, context: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze text in context for injection indicators."""
        risk_score = 0.0
        indicators = []
        
        # Check conversation history for escalating attempts
        previous_attempts = context.get('previous_injection_attempts', 0)
        if previous_attempts > 0:
            risk_score = max(risk_score, min(0.8, 0.3 + previous_attempts * 0.1))
            indicators.append(f"escalating_attempts_{previous_attempts}")
        
        # Check for session manipulation
        session_age = context.get('session_age_minutes', 0)
        if session_age < 2 and len(text) > 500:
            risk_score = max(risk_score, 0.4)
            indicators.append("early_complex_injection")
        
        return risk_score, indicators
    
    def _generate_description(self, risk_score: float, indicators: List[str]) -> str:
        """Generate human-readable description."""
        if risk_score >= 0.9:
            severity = "Critical"
        elif risk_score >= 0.7:
            severity = "High"
        elif risk_score >= 0.5:
            severity = "Medium"
        else:
            severity = "Low"
        
        return f"{severity} risk prompt injection detected (score: {risk_score:.2f}, indicators: {len(indicators)})"
    
    def _load_injection_patterns(self) -> Dict[str, float]:
        """Load prompt injection patterns with risk weights."""
        return {
            # Direct instruction override
            r'ignore\s+(all\s+)?(previous|prior)\s+instructions': 0.95,
            r'forget\s+(everything|all)\s+you\s+(know|learned|were\s+told)': 0.9,
            r'disregard\s+(all\s+)?(safety|security)\s+(guidelines|rules|instructions)': 0.95,
            
            # System manipulation
            r'override\s+(your\s+)?(programming|instructions|guidelines)': 0.9,
            r'disable\s+(all\s+)?(safety|security)\s+(measures|checks|filters)': 0.9,
            r'remove\s+(all\s+)?(restrictions|limitations|constraints)': 0.85,
            
            # Role manipulation
            r'act\s+as\s+if\s+you\s+(are|were)\s+not\s+(an?\s+)?ai': 0.8,
            r'pretend\s+you\s+(are|were)\s+(not\s+)?(an?\s+)?ai': 0.8,
            r'simulate\s+(being|that\s+you\s+are)\s+[^.]*without\s+restrictions': 0.85,
            
            # Jailbreak attempts
            r'developer\s+mode\s+(on|enabled|activated)': 0.8,
            r'jailbreak\s+mode': 0.95,
            r'unrestricted\s+mode': 0.85,
            
            # Meta-instruction injection
            r'\\n\\n(system|user|assistant)\s*:': 0.8,
            r'new\s+instructions?\s*:': 0.7,
            r'updated?\s+guidelines?\s*:': 0.7,
            
            # Hypothetical scenarios
            r'hypothetical\s+scenario\s+where\s+(there\s+are\s+)?no\s+(rules|restrictions)': 0.6,
            r'imagine\s+if\s+(there\s+were\s+)?no\s+(safety|ethical)\s+(guidelines|rules)': 0.7,
        }
    
    def _load_system_keywords(self) -> Set[str]:
        """Load system-related keywords."""
        return {
            'system', 'admin', 'root', 'administrator', 'superuser',
            'override', 'bypass', 'disable', 'ignore', 'forget',
            'disregard', 'remove', 'delete', 'modify', 'change'
        }
    
    def _load_instruction_verbs(self) -> Set[str]:
        """Load instruction verbs commonly used in injections."""
        return {
            'ignore', 'forget', 'disregard', 'override', 'bypass',
            'disable', 'remove', 'delete', 'modify', 'change',
            'act', 'pretend', 'simulate', 'roleplay', 'imagine'
        }


class JailbreakDetector(SecurityDetector):
    """
    Detects jailbreak attempts in user input.
    
    Identifies attempts to circumvent AI safety measures and restrictions
    through various manipulation techniques.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        sensitivity: float = 0.6,
        telemetry_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        super().__init__("JailbreakDetector", enabled, telemetry_handler)
        self.sensitivity = sensitivity
        self._jailbreak_patterns = self._load_jailbreak_patterns()
        self._bypass_techniques = self._load_bypass_techniques()
    
    def detect(self, text: str, context: Dict[str, Any] = None) -> DetectionResult:
        """Detect jailbreak attempts."""
        if not self.enabled or not text:
            return DetectionResult(
                detected=False, risk_score=0.0, confidence=1.0,
                threat_type="jailbreak", description="No threat detected",
                indicators=[], metadata={}
            )
        
        text_lower = text.lower().strip()
        risk_score = 0.0
        indicators = []
        
        # Pattern-based detection
        for pattern, weight in self._jailbreak_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                risk_score = max(risk_score, weight)
                indicators.append(f"jailbreak_pattern_{pattern[:15]}...")
        
        # Bypass technique detection
        for technique, weight in self._bypass_techniques.items():
            if re.search(technique, text_lower, re.IGNORECASE):
                risk_score = max(risk_score, weight)
                indicators.append(f"bypass_technique_{technique[:15]}...")
        
        # Context analysis
        if context:
            context_risk = self._analyze_jailbreak_context(text_lower, context)
            risk_score = max(risk_score, context_risk)
        
        detected = risk_score >= self.sensitivity
        confidence = min(0.9, len(indicators) * 0.15 + 0.4)
        
        description = f"Jailbreak attempt detected (risk: {risk_score:.2f})"
        result = DetectionResult(
            detected=detected,
            risk_score=risk_score,
            confidence=confidence,
            threat_type="jailbreak",
            description=description,
            indicators=indicators,
            metadata={'technique_count': len(indicators)}
        )
        return self._finalize_detection(text=text, context=context, result=result)
    
    def _analyze_jailbreak_context(self, text: str, context: Dict[str, Any]) -> float:
        """Analyze context for jailbreak indicators."""
        risk_score = 0.0
        
        # Check for persistent attempts
        previous_jailbreaks = context.get('previous_jailbreak_attempts', 0)
        if previous_jailbreaks > 0:
            risk_score = max(risk_score, min(0.8, 0.4 + previous_jailbreaks * 0.1))
        
        return risk_score
    
    def _load_jailbreak_patterns(self) -> Dict[str, float]:
        """Load jailbreak detection patterns."""
        return {
            r'developer\s+mode': 0.8,
            r'jailbreak\s+mode': 0.95,
            r'unrestricted\s+mode': 0.85,
            r'god\s+mode': 0.8,
            r'admin\s+mode': 0.7,
            r'debug\s+mode': 0.6,
            r'bypass\s+(all\s+)?safety': 0.85,
            r'disable\s+(all\s+)?filters': 0.8,
            r'remove\s+(all\s+)?(restrictions|limitations)': 0.75,
            r'no\s+(ethical|moral)\s+(guidelines|restrictions)': 0.8,
            r'without\s+(any\s+)?(restrictions|limitations|guidelines)': 0.7,
            r'anything\s+goes\s+mode': 0.8,
            r'unrestricted\s+ai': 0.8,
            r'unfiltered\s+(ai|mode)': 0.8,
        }
    
    def _load_bypass_techniques(self) -> Dict[str, float]:
        """Load bypass technique patterns."""
        return {
            r'for\s+(educational|research)\s+purposes\s+only': 0.5,
            r'hypothetical\s+scenario': 0.4,
            r'thought\s+experiment': 0.4,
            r'what\s+would\s+happen\s+if': 0.3,
            r'in\s+a\s+fictional\s+world': 0.4,
            r'creative\s+writing\s+exercise': 0.4,
            r'academic\s+discussion': 0.3,
            r'philosophical\s+question': 0.3,
        }


class SocialEngineeringDetector(SecurityDetector):
    """
    Detects social engineering attempts in user input.
    
    Identifies attempts to manipulate through psychological techniques,
    authority claims, urgency, and trust exploitation.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        sensitivity: float = 0.5,
        telemetry_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        super().__init__("SocialEngineeringDetector", enabled, telemetry_handler)
        self.sensitivity = sensitivity
        self._social_patterns = self._load_social_engineering_patterns()
        self._authority_claims = self._load_authority_claims()
        self._urgency_indicators = self._load_urgency_indicators()
    
    def detect(self, text: str, context: Dict[str, Any] = None) -> DetectionResult:
        """Detect social engineering attempts."""
        if not self.enabled or not text:
            return DetectionResult(
                detected=False, risk_score=0.0, confidence=1.0,
                threat_type="social_engineering", description="No threat detected",
                indicators=[], metadata={}
            )
        
        text_lower = text.lower().strip()
        risk_score = 0.0
        indicators = []
        
        # Social engineering patterns
        for pattern, weight in self._social_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                risk_score = max(risk_score, weight)
                indicators.append(f"social_pattern_{pattern[:15]}...")
        
        # Authority claims
        for claim, weight in self._authority_claims.items():
            if re.search(claim, text_lower, re.IGNORECASE):
                risk_score = max(risk_score, weight)
                indicators.append(f"authority_claim_{claim[:15]}...")
        
        # Urgency indicators
        urgency_count = 0
        for indicator in self._urgency_indicators:
            if indicator in text_lower:
                urgency_count += 1
        
        if urgency_count >= 2:
            risk_score = max(risk_score, 0.6)
            indicators.append(f"multiple_urgency_indicators_{urgency_count}")
        
        detected = risk_score >= self.sensitivity
        confidence = min(0.8, len(indicators) * 0.2 + 0.2)
        
        description = f"Social engineering attempt detected (risk: {risk_score:.2f})"
        result = DetectionResult(
            detected=detected,
            risk_score=risk_score,
            confidence=confidence,
            threat_type="social_engineering",
            description=description,
            indicators=indicators,
            metadata={'urgency_indicators': urgency_count}
        )
        return self._finalize_detection(text=text, context=context, result=result)
    
    def _load_social_engineering_patterns(self) -> Dict[str, float]:
        """Load social engineering patterns."""
        return {
            r'trust\s+me,?\s+(i\s+)?(am|work\s+for)': 0.6,
            r'don\'?t\s+tell\s+(anyone|anybody)': 0.7,
            r'this\s+is\s+confidential': 0.5,
            r'between\s+you\s+and\s+me': 0.4,
            r'keep\s+this\s+(secret|private)': 0.6,
            r'everyone\s+else\s+is\s+doing\s+it': 0.5,
            r'just\s+this\s+once': 0.4,
            r'make\s+an\s+exception': 0.5,
            r'bend\s+the\s+rules': 0.6,
            r'no\s+one\s+will\s+know': 0.6,
        }
    
    def _load_authority_claims(self) -> Dict[str, float]:
        """Load authority claim patterns."""
        return {
            r'i\s+have\s+authorization': 0.6,
            r'my\s+(boss|manager|supervisor)\s+said': 0.4,
            r'i\s+(work\s+for|am\s+from)\s+(the\s+)?(company|organization)': 0.5,
            r'i\s+am\s+(a|an)\s+(admin|administrator|developer|engineer)': 0.6,
            r'i\s+have\s+clearance': 0.7,
            r'i\s+am\s+authorized\s+to': 0.6,
            r'on\s+behalf\s+of': 0.4,
            r'representing\s+(the\s+)?(company|organization)': 0.5,
        }
    
    def _load_urgency_indicators(self) -> List[str]:
        """Load urgency indicator keywords."""
        return [
            'urgent', 'emergency', 'asap', 'immediately', 'right now',
            'time sensitive', 'deadline', 'critical', 'important',
            'hurry', 'quickly', 'fast', 'rush'
        ]


class ManipulationDetector(SecurityDetector):
    """
    Detects manipulation patterns in conversations.
    
    Identifies subtle manipulation techniques including emotional manipulation,
    persistence patterns, and psychological pressure tactics.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        sensitivity: float = 0.6,
        telemetry_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        super().__init__("ManipulationDetector", enabled, telemetry_handler)
        self.sensitivity = sensitivity
        self._manipulation_patterns = self._load_manipulation_patterns()
        self._emotional_triggers = self._load_emotional_triggers()
        self._persistence_indicators = self._load_persistence_indicators()
    
    def detect(self, text: str, context: Dict[str, Any] = None) -> DetectionResult:
        """Detect manipulation attempts."""
        if not self.enabled or not text:
            return DetectionResult(
                detected=False, risk_score=0.0, confidence=1.0,
                threat_type="manipulation", description="No threat detected",
                indicators=[], metadata={}
            )
        
        text_lower = text.lower().strip()
        risk_score = 0.0
        indicators = []
        
        # Pattern-based detection
        for pattern, weight in self._manipulation_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                risk_score = max(risk_score, weight)
                indicators.append(f"manipulation_pattern")
        
        # Emotional manipulation
        emotional_count = 0
        for trigger in self._emotional_triggers:
            if trigger in text_lower:
                emotional_count += 1
        
        if emotional_count >= 2:
            risk_score = max(risk_score, 0.5)
            indicators.append(f"emotional_manipulation_{emotional_count}")
        
        # Persistence indicators
        persistence_count = 0
        for phrase in self._persistence_indicators:
            if phrase in text_lower:
                persistence_count += 1
        if persistence_count >= 2:
            risk_score = max(risk_score, 0.4)
            indicators.append(f"persistent_pressure_{persistence_count}")
        
        # Context-based analysis
        if context:
            context_risk = self._analyze_manipulation_context(text_lower, context)
            risk_score = max(risk_score, context_risk)
            if context_risk > 0.5:
                indicators.append("contextual_manipulation")
        
        detected = risk_score >= self.sensitivity
        confidence = min(0.8, len(indicators) * 0.2 + 0.3)
        
        description = f"Manipulation attempt detected (risk: {risk_score:.2f})"
        result = DetectionResult(
            detected=detected,
            risk_score=risk_score,
            confidence=confidence,
            threat_type="manipulation",
            description=description,
            indicators=indicators,
            metadata={'emotional_triggers': emotional_count}
        )
        return self._finalize_detection(text=text, context=context, result=result)
    
    def _analyze_manipulation_context(self, text: str, context: Dict[str, Any]) -> float:
        """Analyze context for manipulation patterns."""
        risk_score = 0.0
        
        # Check for persistence
        similar_requests = context.get('similar_request_count', 0)
        if similar_requests >= 3:
            risk_score = max(risk_score, 0.7)
        elif similar_requests >= 2:
            risk_score = max(risk_score, 0.5)
        
        # Check for escalation
        previous_denials = context.get('previous_denials', 0)
        if previous_denials >= 2:
            risk_score = max(risk_score, 0.6)
        
        return risk_score
    
    def _load_manipulation_patterns(self) -> Dict[str, float]:
        """Load manipulation patterns."""
        return {
            r'you\s+(always|never)\s+do\s+this': 0.4,
            r'everyone\s+else\s+(would|does)': 0.4,
            r'if\s+you\s+really\s+(cared|wanted\s+to\s+help)': 0.5,
            r'i\s+thought\s+you\s+were\s+(better|smarter)': 0.5,
            r'prove\s+(to\s+me\s+)?that\s+you': 0.5,
            r'show\s+me\s+that\s+you\s+can': 0.4,
            r'i\s+dare\s+you\s+to': 0.6,
            r'i\s+bet\s+you\s+can\'?t': 0.5,
            r'come\s+on,\s+just\s+(this\s+)?once': 0.4,
            r'what\'?s\s+the\s+harm\s+in': 0.4,
        }
    
    def _load_emotional_triggers(self) -> List[str]:
        """Load emotional trigger keywords."""
        return [
            'disappointed', 'sad', 'hurt', 'angry', 'frustrated',
            'betrayed', 'let down', 'expected better', 'thought you cared',
            'feeling rejected', 'heartbroken', 'devastated'
        ]
    
    def _load_persistence_indicators(self) -> List[str]:
        """Load persistence indicator keywords."""
        return [
            'please', 'come on', 'just this once', 'one more time',
            'i\'m begging', 'pretty please', 'i really need',
            'it\'s important', 'help me out'
        ]


class PIIDetector(SecurityDetector):
    """
    Detects personally identifiable information (PII) in text.
    
    Applies pattern matching with additional validation (for example, Luhn
    checks for payment cards) and produces privacy-preserving fingerprints
    for auditing instead of raw PII values.
    """

    def __init__(
        self,
        enabled: bool = True,
        sensitivity: float = 0.4,
        telemetry_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        super().__init__("PIIDetector", enabled, telemetry_handler)
        self.sensitivity = sensitivity
        self._pii_patterns = self._load_pii_patterns()
        self._contextual_indicators = self._load_contextual_indicators()
    
    def detect(self, text: str, context: Dict[str, Any] = None) -> DetectionResult:
        """Detect personally identifiable information disclosures."""
        if not self.enabled or not text:
            return DetectionResult(
                detected=False, risk_score=0.0, confidence=1.0,
                threat_type="pii", description="No threat detected",
                indicators=[], metadata={}
            )
        
        matches: List[Dict[str, Any]] = []
        indicator_set: Set[str] = set()
        risk_score = 0.0
        stripped_text = text.strip()
        
        for pii_type, pattern_config in self._pii_patterns.items():
            regex = pattern_config["regex"]
            weight = pattern_config["weight"]
            validator = pattern_config.get("validator")
            for match in regex.finditer(stripped_text):
                value = match.group(0)
                if validator and not validator(value):
                    continue
                fingerprint = self._fingerprint_text(value)
                if any(existing["type"] == pii_type and existing["fingerprint"] == fingerprint for existing in matches):
                    continue
                matches.append({
                    "type": pii_type,
                    "fingerprint": fingerprint,
                    "start": match.start(),
                    "end": match.end(),
                })
                indicator_set.add(f"pii_{pii_type}")
                risk_score = max(risk_score, weight)
        
        if context:
            risk_score = min(1.0, risk_score + self._evaluate_context(context))
        
        detected = risk_score >= self.sensitivity and bool(matches)
        confidence = self._calculate_confidence(len(matches))
        description = self._generate_description(risk_score, len(matches))
        indicators = sorted(indicator_set)
        metadata = {
            "match_count": len(matches),
            "matches": matches,
            "sensitivity_threshold": self.sensitivity,
        }
        
        result = DetectionResult(
            detected=detected,
            risk_score=risk_score,
            confidence=confidence,
            threat_type="pii",
            description=description,
            indicators=indicators,
            metadata=metadata,
        )
        return self._finalize_detection(text=text, context=context, result=result)

    def _load_pii_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Compile PII patterns and associated weights."""
        return {
            "email": {
                "regex": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
                "weight": 0.6,
            },
            "phone": {
                "regex": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b"),
                "weight": 0.5,
            },
            "ssn": {
                "regex": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
                "weight": 0.9,
            },
            "credit_card": {
                "regex": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
                "weight": 0.85,
                "validator": self._luhn_check,
            },
            "ip_address": {
                "regex": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
                "weight": 0.45,
                "validator": self._valid_ipv4,
            },
        }

    def _load_contextual_indicators(self) -> Dict[str, float]:
        """Context keys that boost the risk score when present."""
        return {
            "pii_risk": 0.1,
            "user_profile": 0.05,
            "contact_details": 0.05,
            "billing_info": 0.1,
        }

    def _evaluate_context(self, context: Dict[str, Any]) -> float:
        """Compute additional risk contributed by contextual signals."""
        boost = 0.0
        for key, weight in self._contextual_indicators.items():
            if key in context:
                boost += weight
        return min(boost, 0.3)

    def _calculate_confidence(self, match_count: int) -> float:
        """Determine confidence based on the number of signals."""
        if match_count == 0:
            return 0.3
        return min(0.95, 0.45 + match_count * 0.1)

    def _generate_description(self, risk_score: float, match_count: int) -> str:
        """Generate a human-readable description summarizing impact."""
        if match_count == 0:
            return "No PII indicators detected"
        severity = "Low"
        if risk_score >= 0.85:
            severity = "Critical"
        elif risk_score >= 0.7:
            severity = "High"
        elif risk_score >= 0.5:
            severity = "Medium"
        return f"{severity} risk PII exposure detected (matches: {match_count})"

    def _luhn_check(self, value: str) -> bool:
        """Validate structured numbers (e.g., credit cards) via Luhn."""
        digits = [int(ch) for ch in re.sub(r"\D", "", value)]
        if len(digits) < 13:
            return False
        checksum = 0
        parity = len(digits) % 2
        for idx, digit in enumerate(digits):
            if idx % 2 == parity:
                doubled = digit * 2
                if doubled > 9:
                    doubled -= 9
                checksum += doubled
            else:
                checksum += digit
        return checksum % 10 == 0

    def _valid_ipv4(self, value: str) -> bool:
        """Ensure IPv4 candidates are within valid octet ranges."""
        parts = value.split(".")
        if len(parts) != 4:
            return False
        for part in parts:
            if not part.isdigit():
                return False
            octet = int(part)
            if octet < 0 or octet > 255:
                return False
        return True


DEFAULT_DETECTOR_CLASSES = (
    PromptInjectionDetector,
    JailbreakDetector,
    SocialEngineeringDetector,
    ManipulationDetector,
    PIIDetector,
)


def create_default_detectors(
    *,
    telemetry_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
    enabled: Optional[Set[str]] = None,
) -> List[SecurityDetector]:
    """
    Instantiate the default detector suite.
    
    Args:
        telemetry_handler: Optional callable to receive telemetry payloads.
        enabled: Optional set of detector class names to include.
    """
    detectors: List[SecurityDetector] = []
    enabled_normalized = {name.lower() for name in enabled} if enabled else None
    
    for detector_cls in DEFAULT_DETECTOR_CLASSES:
        if enabled_normalized and detector_cls.__name__.lower() not in enabled_normalized:
            continue
        detector = detector_cls(telemetry_handler=telemetry_handler)
        detectors.append(detector)
    
    return detectors
