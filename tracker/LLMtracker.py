# llm_tracker.py
import requests
from datetime import datetime

class LLMTracker:
    def __init__(self, backend_url: str):
        """
        backend_url: Your API endpoint that will receive token usage logs.
                     Example: https://api.yourdomain.com/llm-usage
        """
        self.backend_url = backend_url

    def _send_to_backend(self, payload: dict):
        try:
            # Send JSON payload to backend
            requests.post(self.backend_url, json=payload, timeout=3)
        except Exception as e:
            print(f"[LLMTracker] Failed to send usage data: {e}")

    def _record(self, provider, model, prompt_tokens, completion_tokens):
        total = prompt_tokens + completion_tokens
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
        }
        self._send_to_backend(payload)

    def chat(self, provider, client, model, **kwargs):
        """
        provider: 'openai', 'anthropic', or 'gemini'
        client: SDK client object
        model: model name
        kwargs: arguments for the SDK's chat/generate method
        """

        if provider == "openai":
            response = client.chat.completions.create(model=model, **kwargs)
            u = response.usage
            self._record("openai", model, u.prompt_tokens, u.completion_tokens)
            return response

        elif provider == "anthropic":
            response = client.messages.create(model=model, **kwargs)
            u = response.usage
            self._record("anthropic", model, u.input_tokens, u.output_tokens)
            return response

        elif provider == "gemini":
            response = client.generate_content(**kwargs)
            meta = getattr(response, "usage_metadata", None)
            if meta:
                prompt_tokens = getattr(meta, "prompt_token_count", getattr(meta, "input_tokens", 0))
                completion_tokens = getattr(meta, "candidates_token_count", getattr(meta, "output_tokens", 0))
            else:
                prompt_tokens = completion_tokens = 0
            self._record("gemini", model, prompt_tokens, completion_tokens)
            return response

        else:
            raise ValueError(f"Unsupported provider: {provider}")