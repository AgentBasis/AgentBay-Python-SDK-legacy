# llm_tracker.py
import requests
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union

class LLMTracker:
    def __init__(self, backend_url: str, 
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 timeout: float = 5.0,
                 enable_logging: bool = True):
        """
        Enhanced LLM usage tracker with retry logic and comprehensive error handling.
        
        Args:
            backend_url: Your API endpoint that will receive token usage logs.
                        Example: https://api.yourdomain.com/llm-usage
            max_retries: Maximum number of retry attempts for failed requests (default: 3)
            retry_delay: Initial delay between retries in seconds (default: 1.0)
            timeout: Request timeout in seconds (default: 5.0)
            enable_logging: Whether to enable detailed logging (default: True)
        """
        self.backend_url = backend_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # Setup logging
        self.logger = logging.getLogger(__name__) if enable_logging else None
        if self.logger:
            self.logger.setLevel(logging.INFO)

    def _send_to_backend(self, payload: Dict[str, Any]) -> bool:
        """
        Send payload to backend with retry logic and comprehensive error handling.
        
        Args:
            payload: Data to send to backend
            
        Returns:
            bool: True if successful, False if all retries failed
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if self.logger and attempt > 0:
                    self.logger.info(f"[LLMTracker] Retry attempt {attempt}/{self.max_retries}")
                
                response = requests.post(
                    self.backend_url, 
                    json=payload, 
                    timeout=self.timeout,
                    headers={'Content-Type': 'application/json'}
                )
                
                # Check if request was successful
                response.raise_for_status()
                
                if self.logger:
                    self.logger.info(f"[LLMTracker] Successfully sent usage data: {payload.get('provider', 'unknown')}/{payload.get('model', 'unknown')}")
                
                return True
                
            except requests.exceptions.Timeout as e:
                last_exception = f"Request timeout after {self.timeout}s"
                if self.logger:
                    self.logger.warning(f"[LLMTracker] Timeout on attempt {attempt + 1}: {e}")
                    
            except requests.exceptions.ConnectionError as e:
                last_exception = f"Connection error: {e}"
                if self.logger:
                    self.logger.warning(f"[LLMTracker] Connection error on attempt {attempt + 1}: {e}")
                    
            except requests.exceptions.HTTPError as e:
                last_exception = f"HTTP error {e.response.status_code}: {e}"
                if self.logger:
                    self.logger.warning(f"[LLMTracker] HTTP error on attempt {attempt + 1}: {e}")
                
                # Don't retry on client errors (4xx)
                if e.response.status_code < 500:
                    break
                    
            except Exception as e:
                last_exception = f"Unexpected error: {e}"
                if self.logger:
                    self.logger.warning(f"[LLMTracker] Unexpected error on attempt {attempt + 1}: {e}")
            
            # Wait before retrying (with exponential backoff)
            if attempt < self.max_retries:
                delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                time.sleep(delay)
        
        # All retries failed
        if self.logger:
            self.logger.error(f"[LLMTracker] Failed to send usage data after {self.max_retries + 1} attempts. Last error: {last_exception}")
        else:
            print(f"[LLMTracker] Failed to send usage data: {last_exception}")
        
        return False

    def _record(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> bool:
        """
        Record LLM usage with enhanced error handling.
        
        Args:
            provider: LLM provider name
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            bool: True if successfully sent to backend, False otherwise
        """
        try:
            total = prompt_tokens + completion_tokens
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "provider": provider,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total,
            }
            
            return self._send_to_backend(payload)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"[LLMTracker] Error creating usage record: {e}")
            else:
                print(f"[LLMTracker] Error creating usage record: {e}")
            return False

    def chat(self, provider: str, client: Any, model: str, **kwargs) -> Any:
        """
        Enhanced chat method with better error handling and provider support.
        
        Args:
            provider: 'openai', 'anthropic', or 'gemini'
            client: SDK client object
            model: model name
            kwargs: arguments for the SDK's chat/generate method
            
        Returns:
            The response from the LLM provider
            
        Raises:
            ValueError: If provider is not supported
            Exception: Re-raises any exceptions from the LLM provider
        """
        if self.logger:
            self.logger.info(f"[LLMTracker] Making {provider} request with model {model}")

        try:
            if provider == "openai":
                response = client.chat.completions.create(model=model, **kwargs)
                usage = getattr(response, 'usage', None)
                if usage:
                    self._record("openai", model, usage.prompt_tokens, usage.completion_tokens)
                else:
                    if self.logger:
                        self.logger.warning("[LLMTracker] No usage data found in OpenAI response")
                return response

            elif provider == "anthropic":
                response = client.messages.create(model=model, **kwargs)
                usage = getattr(response, 'usage', None)
                if usage:
                    input_tokens = getattr(usage, 'input_tokens', 0)
                    output_tokens = getattr(usage, 'output_tokens', 0)
                    self._record("anthropic", model, input_tokens, output_tokens)
                else:
                    if self.logger:
                        self.logger.warning("[LLMTracker] No usage data found in Anthropic response")
                return response

            elif provider == "gemini":
                response = client.generate_content(**kwargs)
                meta = getattr(response, "usage_metadata", None)
                if meta:
                    prompt_tokens = getattr(meta, "prompt_token_count", getattr(meta, "input_tokens", 0))
                    completion_tokens = getattr(meta, "candidates_token_count", getattr(meta, "output_tokens", 0))
                    self._record("gemini", model, prompt_tokens, completion_tokens)
                else:
                    if self.logger:
                        self.logger.warning("[LLMTracker] No usage metadata found in Gemini response")
                    # Record with zero tokens if no metadata available
                    self._record("gemini", model, 0, 0)
                return response

            else:
                error_msg = f"Unsupported provider: {provider}. Supported providers: openai, anthropic, gemini"
                if self.logger:
                    self.logger.error(f"[LLMTracker] {error_msg}")
                raise ValueError(error_msg)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[LLMTracker] Error during {provider} request: {e}")
            raise  # Re-raise the exception so caller can handle it

    def test_connection(self) -> bool:
        """
        Test connection to backend with a minimal payload.
        
        Returns:
            bool: True if backend is reachable, False otherwise
        """
        test_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": "test",
            "model": "connection-test",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "test": True
        }
        
        if self.logger:
            self.logger.info("[LLMTracker] Testing backend connection...")
        
        success = self._send_to_backend(test_payload)
        
        if self.logger:
            status = "successful" if success else "failed"
            self.logger.info(f"[LLMTracker] Backend connection test {status}")
        
        return success