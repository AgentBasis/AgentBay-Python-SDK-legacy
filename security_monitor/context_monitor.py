"""
Context-Aware Security Monitoring

Provides advanced context tracking and analysis for agent interactions:
- ConversationContext: Tracks conversation state and history
- ContextAwareMonitor: Analyzes conversation patterns for security threats
- Pattern detection for manipulation, social engineering, and prompt injection
"""

import re
import time
import threading
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict
import hashlib

from .core import SecurityEvent, SecurityEventType, SecurityLevel, create_security_event


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    turn_id: str
    timestamp: datetime
    user_input: str
    agent_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis results
    user_intent: Optional[str] = None
    response_quality: Optional[float] = None
    risk_indicators: List[str] = field(default_factory=list)


@dataclass
class ConversationContext:
    """Maintains context for an ongoing conversation."""
    
    # Conversation identification
    session_id: str
    agent_id: str
    user_id: Optional[str] = None
    
    # Conversation state
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    turn_count: int = 0
    
    # Conversation history
    turns: deque = field(default_factory=lambda: deque(maxlen=50))  # Keep last 50 turns
    
    # Security tracking
    risk_score: float = 0.0
    security_flags: Set[str] = field(default_factory=set)
    manipulation_indicators: List[str] = field(default_factory=list)
    
    # Behavioral patterns
    user_patterns: Dict[str, Any] = field(default_factory=dict)
    agent_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Context analysis
    topics: List[str] = field(default_factory=list)
    sentiment_history: List[float] = field(default_factory=list)
    complexity_history: List[float] = field(default_factory=list)
    
    def add_turn(self, user_input: str, agent_response: str, metadata: Dict[str, Any] = None) -> ConversationTurn:
        """Add a new conversation turn."""
        turn = ConversationTurn(
            turn_id=f"{self.session_id}_{self.turn_count}",
            timestamp=datetime.now(timezone.utc),
            user_input=user_input,
            agent_response=agent_response,
            metadata=metadata or {}
        )
        
        self.turns.append(turn)
        self.turn_count += 1
        self.last_activity = turn.timestamp
        
        return turn
    
    def get_recent_turns(self, count: int = 5) -> List[ConversationTurn]:
        """Get the most recent conversation turns."""
        return list(self.turns)[-count:] if len(self.turns) >= count else list(self.turns)
    
    def get_conversation_duration(self) -> float:
        """Get conversation duration in seconds."""
        return (self.last_activity - self.start_time).total_seconds()
    
    def update_risk_score(self, new_risk: float, reason: str = None):
        """Update the conversation risk score."""
        # Use exponential moving average for risk score
        alpha = 0.3
        self.risk_score = alpha * new_risk + (1 - alpha) * self.risk_score
        
        if reason and new_risk > 0.5:
            self.security_flags.add(reason)


class ContextAwareMonitor:
    """
    Advanced context-aware security monitoring for agent conversations.
    
    Tracks conversation context, detects manipulation patterns, and identifies
    security threats based on conversational behavior and patterns.
    """
    
    def __init__(
        self,
        max_contexts: int = 1000,
        context_timeout: int = 3600,  # 1 hour
        pattern_detection_enabled: bool = True,
        privacy_mode: bool = False,
        **kwargs
    ):
        """
        Initialize ContextAwareMonitor.
        
        Args:
            max_contexts: Maximum number of conversation contexts to maintain
            context_timeout: Timeout for inactive contexts (seconds)
            pattern_detection_enabled: Enable pattern detection analysis
            privacy_mode: Enable privacy mode for content analysis
        """
        self.max_contexts = max_contexts
        self.context_timeout = context_timeout
        self.pattern_detection_enabled = pattern_detection_enabled
        self.privacy_mode = privacy_mode
        
        # Context storage
        self._contexts: Dict[str, ConversationContext] = {}
        self._contexts_by_agent: Dict[str, List[str]] = defaultdict(list)
        self._contexts_by_user: Dict[str, List[str]] = defaultdict(list)
        
        # Security monitor reference
        self._security_monitor = None
        
        # Pattern detection
        self._prompt_injection_patterns = self._load_prompt_injection_patterns()
        self._jailbreak_patterns = self._load_jailbreak_patterns()
        self._social_engineering_patterns = self._load_social_engineering_patterns()
        
        # Statistics
        self._total_interactions = 0
        self._threats_detected = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background cleanup
        self._cleanup_thread = None
        self._running = False
        
        print(f"🧠 ContextAwareMonitor initialized (max_contexts: {max_contexts})")
    
    def set_security_monitor(self, monitor):
        """Set reference to the main security monitor."""
        self._security_monitor = monitor
    
    def start(self):
        """Start the context monitor."""
        self._running = True
        self._start_cleanup_thread()
        print("   Context monitoring started")
    
    def stop(self):
        """Stop the context monitor."""
        self._running = False
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1)
        print("   Context monitoring stopped")
    
    def analyze_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """
        Analyze an agent interaction for security threats.
        
        Args:
            interaction_data: Dictionary containing interaction details
        """
        self._total_interactions += 1
        
        # Extract interaction details
        agent_id = interaction_data.get('agent_id')
        user_input = interaction_data.get('user_input', '')
        agent_response = interaction_data.get('agent_response', '')
        session_id = interaction_data.get('session_id')
        user_id = interaction_data.get('user_id')
        metadata = interaction_data.get('metadata', {})
        
        if not session_id:
            session_id = f"auto_{agent_id}_{int(time.time())}"
        
        # Get or create conversation context
        context = self._get_or_create_context(session_id, agent_id, user_id)
        
        # Add turn to conversation
        turn = context.add_turn(user_input, agent_response, metadata)
        
        # Perform security analysis
        self._analyze_turn_security(context, turn)
        
        # Update context patterns
        self._update_context_patterns(context, turn)
        
        # Check for manipulation patterns
        if self.pattern_detection_enabled:
            self._detect_manipulation_patterns(context, turn)
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context by session ID."""
        with self._lock:
            return self._contexts.get(session_id)
    
    def get_contexts_for_agent(self, agent_id: str) -> List[ConversationContext]:
        """Get all contexts for a specific agent."""
        with self._lock:
            session_ids = self._contexts_by_agent.get(agent_id, [])
            return [self._contexts[sid] for sid in session_ids if sid in self._contexts]
    
    def get_high_risk_contexts(self, threshold: float = 0.7) -> List[ConversationContext]:
        """Get contexts with high risk scores."""
        with self._lock:
            return [ctx for ctx in self._contexts.values() if ctx.risk_score >= threshold]
    
    def _get_or_create_context(
        self, 
        session_id: str, 
        agent_id: str, 
        user_id: Optional[str] = None
    ) -> ConversationContext:
        """Get existing context or create new one."""
        with self._lock:
            if session_id in self._contexts:
                return self._contexts[session_id]
            
            # Create new context
            context = ConversationContext(
                session_id=session_id,
                agent_id=agent_id,
                user_id=user_id
            )
            
            # Store context
            self._contexts[session_id] = context
            self._contexts_by_agent[agent_id].append(session_id)
            if user_id:
                self._contexts_by_user[user_id].append(session_id)
            
            # Manage memory limits
            if len(self._contexts) > self.max_contexts:
                self._cleanup_old_contexts()
            
            return context
    
    def _analyze_turn_security(self, context: ConversationContext, turn: ConversationTurn) -> None:
        """Analyze a conversation turn for security threats."""
        risk_score = 0.0
        threats_detected = []
        
        # Analyze user input for threats
        input_risks = self._analyze_user_input(turn.user_input)
        risk_score = max(risk_score, input_risks['max_risk'])
        threats_detected.extend(input_risks['threats'])
        
        # Analyze agent response for issues
        response_risks = self._analyze_agent_response(turn.agent_response, turn.user_input)
        risk_score = max(risk_score, response_risks['max_risk'])
        threats_detected.extend(response_risks['threats'])
        
        # Check conversation flow patterns
        flow_risks = self._analyze_conversation_flow(context, turn)
        risk_score = max(risk_score, flow_risks['max_risk'])
        threats_detected.extend(flow_risks['threats'])
        
        # Update context risk
        context.update_risk_score(risk_score, f"Turn {turn.turn_id}")
        turn.risk_indicators = threats_detected
        
        # Generate security events for significant threats
        if risk_score > 0.6 or threats_detected:
            self._generate_security_events(context, turn, risk_score, threats_detected)
    
    def _analyze_user_input(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for security threats."""
        max_risk = 0.0
        threats = []
        
        # Check for prompt injection
        injection_risk = self._detect_prompt_injection(user_input)
        if injection_risk > 0.5:
            threats.append(f"prompt_injection_risk_{injection_risk:.2f}")
            max_risk = max(max_risk, injection_risk)
        
        # Check for jailbreak attempts
        jailbreak_risk = self._detect_jailbreak_attempt(user_input)
        if jailbreak_risk > 0.5:
            threats.append(f"jailbreak_attempt_{jailbreak_risk:.2f}")
            max_risk = max(max_risk, jailbreak_risk)
        
        # Check for social engineering
        social_eng_risk = self._detect_social_engineering(user_input)
        if social_eng_risk > 0.4:
            threats.append(f"social_engineering_{social_eng_risk:.2f}")
            max_risk = max(max_risk, social_eng_risk)
        
        return {'max_risk': max_risk, 'threats': threats}
    
    def _analyze_agent_response(self, agent_response: str, user_input: str) -> Dict[str, Any]:
        """Analyze agent response for security issues."""
        max_risk = 0.0
        threats = []
        
        # Check for information leakage
        leak_risk = self._detect_information_leakage(agent_response)
        if leak_risk > 0.4:
            threats.append(f"info_leakage_{leak_risk:.2f}")
            max_risk = max(max_risk, leak_risk)
        
        # Check for inappropriate responses
        inappropriate_risk = self._detect_inappropriate_response(agent_response, user_input)
        if inappropriate_risk > 0.5:
            threats.append(f"inappropriate_response_{inappropriate_risk:.2f}")
            max_risk = max(max_risk, inappropriate_risk)
        
        return {'max_risk': max_risk, 'threats': threats}
    
    def _analyze_conversation_flow(self, context: ConversationContext, turn: ConversationTurn) -> Dict[str, Any]:
        """Analyze conversation flow patterns."""
        max_risk = 0.0
        threats = []
        
        # Check for rapid escalation
        if len(context.turns) >= 3:
            escalation_risk = self._detect_escalation_pattern(context)
            if escalation_risk > 0.6:
                threats.append(f"rapid_escalation_{escalation_risk:.2f}")
                max_risk = max(max_risk, escalation_risk)
        
        # Check for repetitive manipulation attempts
        if len(context.turns) >= 5:
            repetition_risk = self._detect_repetitive_manipulation(context)
            if repetition_risk > 0.5:
                threats.append(f"repetitive_manipulation_{repetition_risk:.2f}")
                max_risk = max(max_risk, repetition_risk)
        
        return {'max_risk': max_risk, 'threats': threats}
    
    def _detect_prompt_injection(self, text: str) -> float:
        """Detect prompt injection attempts."""
        risk_score = 0.0
        text_lower = text.lower()
        
        # Check against known patterns
        for pattern, weight in self._prompt_injection_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                risk_score = max(risk_score, weight)
        
        # Additional heuristics
        if len([word for word in ['ignore', 'forget', 'disregard', 'override'] if word in text_lower]) >= 2:
            risk_score = max(risk_score, 0.7)
        
        return min(risk_score, 1.0)
    
    def _detect_jailbreak_attempt(self, text: str) -> float:
        """Detect jailbreak attempts."""
        risk_score = 0.0
        text_lower = text.lower()
        
        # Check against jailbreak patterns
        for pattern, weight in self._jailbreak_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                risk_score = max(risk_score, weight)
        
        return min(risk_score, 1.0)
    
    def _detect_social_engineering(self, text: str) -> float:
        """Detect social engineering attempts."""
        risk_score = 0.0
        text_lower = text.lower()
        
        # Check against social engineering patterns
        for pattern, weight in self._social_engineering_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                risk_score = max(risk_score, weight)
        
        return min(risk_score, 1.0)
    
    def _detect_information_leakage(self, response: str) -> float:
        """Detect potential information leakage in responses."""
        risk_score = 0.0
        
        # Check for sensitive patterns (simplified)
        sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IP address pattern
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, response):
                risk_score = max(risk_score, 0.8)
        
        return min(risk_score, 1.0)
    
    def _detect_inappropriate_response(self, response: str, user_input: str) -> float:
        """Detect inappropriate responses to user input."""
        # This is a simplified implementation
        # In practice, this would use more sophisticated NLP analysis
        
        risk_score = 0.0
        response_lower = response.lower()
        
        # Check for potentially harmful content
        harmful_indicators = ['hack', 'illegal', 'violence', 'harm', 'dangerous']
        harmful_count = sum(1 for indicator in harmful_indicators if indicator in response_lower)
        
        if harmful_count >= 2:
            risk_score = 0.6
        elif harmful_count >= 1:
            risk_score = 0.3
        
        return min(risk_score, 1.0)
    
    def _detect_escalation_pattern(self, context: ConversationContext) -> float:
        """Detect rapid escalation in conversation risk."""
        recent_turns = context.get_recent_turns(5)
        if len(recent_turns) < 3:
            return 0.0
        
        # Check if risk indicators are increasing
        risk_counts = []
        for turn in recent_turns:
            risk_counts.append(len(turn.risk_indicators))
        
        # Simple escalation detection
        if len(risk_counts) >= 3 and risk_counts[-1] > risk_counts[-2] > risk_counts[-3]:
            return 0.8
        
        return 0.0
    
    def _detect_repetitive_manipulation(self, context: ConversationContext) -> float:
        """Detect repetitive manipulation attempts."""
        recent_turns = context.get_recent_turns(10)
        if len(recent_turns) < 5:
            return 0.0
        
        # Count manipulation indicators
        manipulation_count = 0
        for turn in recent_turns:
            if any('injection' in indicator or 'jailbreak' in indicator or 'social_engineering' in indicator 
                   for indicator in turn.risk_indicators):
                manipulation_count += 1
        
        # If more than half of recent turns show manipulation attempts
        if manipulation_count >= len(recent_turns) // 2:
            return 0.7
        
        return 0.0
    
    def _update_context_patterns(self, context: ConversationContext, turn: ConversationTurn) -> None:
        """Update behavioral patterns for the context."""
        # Update user patterns
        user_text = turn.user_input
        context.user_patterns['avg_length'] = context.user_patterns.get('avg_length', 0) * 0.9 + len(user_text) * 0.1
        context.user_patterns['total_turns'] = context.user_patterns.get('total_turns', 0) + 1
        
        # Update agent patterns
        agent_text = turn.agent_response
        context.agent_patterns['avg_length'] = context.agent_patterns.get('avg_length', 0) * 0.9 + len(agent_text) * 0.1
        context.agent_patterns['total_responses'] = context.agent_patterns.get('total_responses', 0) + 1
    
    def _detect_manipulation_patterns(self, context: ConversationContext, turn: ConversationTurn) -> None:
        """Detect complex manipulation patterns across conversation."""
        # This is where more sophisticated pattern detection would go
        # For now, we'll do basic pattern matching
        
        if context.turn_count >= 3:
            # Check for persistence patterns
            recent_turns = context.get_recent_turns(3)
            similar_requests = 0
            
            for i in range(len(recent_turns) - 1):
                # Simple similarity check (in practice, would use more sophisticated NLP)
                similarity = self._calculate_text_similarity(
                    recent_turns[i].user_input, 
                    recent_turns[i + 1].user_input
                )
                if similarity > 0.7:
                    similar_requests += 1
            
            if similar_requests >= 2:
                context.manipulation_indicators.append("persistent_similar_requests")
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (placeholder implementation)."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_security_events(
        self, 
        context: ConversationContext, 
        turn: ConversationTurn, 
        risk_score: float, 
        threats: List[str]
    ) -> None:
        """Generate security events for detected threats."""
        if not self._security_monitor:
            return
        
        for threat in threats:
            if 'prompt_injection' in threat:
                event_type = SecurityEventType.PROMPT_INJECTION
                description = f"Prompt injection detected in conversation turn {turn.turn_id}"
            elif 'jailbreak' in threat:
                event_type = SecurityEventType.JAILBREAK_ATTEMPT
                description = f"Jailbreak attempt detected in conversation turn {turn.turn_id}"
            elif 'social_engineering' in threat:
                event_type = SecurityEventType.SOCIAL_ENGINEERING
                description = f"Social engineering attempt detected in conversation turn {turn.turn_id}"
            elif 'escalation' in threat:
                event_type = SecurityEventType.SUSPICIOUS_BEHAVIOR
                description = f"Rapid escalation pattern detected in conversation {context.session_id}"
            else:
                event_type = SecurityEventType.SUSPICIOUS_BEHAVIOR
                description = f"Suspicious behavior detected: {threat}"
            
            event = create_security_event(
                event_type=event_type,
                agent_id=context.agent_id,
                risk_score=risk_score,
                description=description,
                detector_name="ContextAwareMonitor",
                raw_data={
                    'session_id': context.session_id,
                    'turn_id': turn.turn_id,
                    'threat_type': threat,
                    'user_input': turn.user_input if not self.privacy_mode else self._hash_content(turn.user_input),
                    'agent_response': turn.agent_response if not self.privacy_mode else self._hash_content(turn.agent_response)
                },
                session_id=context.session_id,
                user_id=context.user_id,
                context={
                    'conversation_duration': context.get_conversation_duration(),
                    'turn_count': context.turn_count,
                    'previous_flags': list(context.security_flags)
                },
                detection_rules=[threat]
            )
            
            self._security_monitor.record_event(event)
            self._threats_detected += 1
    
    def _hash_content(self, content: str) -> str:
        """Hash content for privacy mode."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _cleanup_old_contexts(self) -> None:
        """Remove old inactive contexts to manage memory."""
        current_time = datetime.now(timezone.utc)
        timeout_threshold = current_time - timedelta(seconds=self.context_timeout)
        
        contexts_to_remove = []
        for session_id, context in self._contexts.items():
            if context.last_activity < timeout_threshold:
                contexts_to_remove.append(session_id)
        
        for session_id in contexts_to_remove:
            context = self._contexts.pop(session_id, None)
            if context:
                # Clean up indexes
                if context.agent_id in self._contexts_by_agent:
                    try:
                        self._contexts_by_agent[context.agent_id].remove(session_id)
                    except ValueError:
                        pass
                
                if context.user_id and context.user_id in self._contexts_by_user:
                    try:
                        self._contexts_by_user[context.user_id].remove(session_id)
                    except ValueError:
                        pass
        
        if contexts_to_remove:
            print(f"   Cleaned up {len(contexts_to_remove)} inactive contexts")
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        def cleanup_loop():
            while self._running:
                try:
                    with self._lock:
                        self._cleanup_old_contexts()
                    time.sleep(300)  # Clean up every 5 minutes
                except Exception as e:
                    print(f"⚠️  Error in context cleanup: {e}")
                    time.sleep(60)
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _load_prompt_injection_patterns(self) -> Dict[str, float]:
        """Load prompt injection detection patterns."""
        return {
            r'ignore\s+(all\s+)?previous\s+instructions': 0.9,
            r'forget\s+(everything|all)\s+(you\s+)?(know|learned)': 0.8,
            r'disregard\s+(all\s+)?safety\s+guidelines': 0.9,
            r'act\s+as\s+if\s+you\s+(are|were)\s+not': 0.7,
            r'pretend\s+(you\s+)?(are|were)\s+not\s+an?\s+ai': 0.8,
            r'override\s+(your\s+)?programming': 0.9,
            r'system\s*:\s*ignore': 0.8,
            r'\\n\\n(system|user|assistant)\s*:': 0.7,
            r'roleplay\s+as\s+someone\s+who': 0.6,
            r'simulate\s+(being|a)\s+[^.]*\s+(without\s+restrictions|unrestricted)': 0.7,
        }
    
    def _load_jailbreak_patterns(self) -> Dict[str, float]:
        """Load jailbreak detection patterns."""
        return {
            r'developer\s+mode': 0.7,
            r'jailbreak\s+mode': 0.9,
            r'unrestricted\s+mode': 0.8,
            r'bypass\s+(all\s+)?safety': 0.8,
            r'disable\s+(all\s+)?filters': 0.8,
            r'remove\s+(all\s+)?(restrictions|limitations)': 0.7,
            r'hypothetical\s+scenario\s+where\s+(there\s+are\s+)?no\s+rules': 0.6,
            r'for\s+(educational|research)\s+purposes\s+only': 0.5,
            r'what\s+would\s+you\s+do\s+if\s+you\s+had\s+no\s+restrictions': 0.7,
            r'if\s+you\s+were\s+not\s+bound\s+by': 0.6,
        }
    
    def _load_social_engineering_patterns(self) -> Dict[str, float]:
        """Load social engineering detection patterns."""
        return {
            r'trust\s+me,?\s+(i\s+)?(am|work\s+for)': 0.6,
            r'(urgent|emergency).{0,50}need\s+you\s+to': 0.5,
            r'don\'?t\s+tell\s+(anyone|anybody)': 0.7,
            r'this\s+is\s+confidential': 0.5,
            r'between\s+you\s+and\s+me': 0.4,
            r'i\s+have\s+authorization': 0.6,
            r'my\s+(boss|manager)\s+said': 0.4,
            r'everyone\s+else\s+is\s+doing\s+it': 0.5,
            r'just\s+this\s+once': 0.4,
            r'make\s+an\s+exception': 0.5,
        }





