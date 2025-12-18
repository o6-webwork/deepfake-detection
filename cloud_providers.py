"""
Cloud VLM Provider Adapters for Gemini, OpenAI, and Anthropic Claude.

This module provides unified interfaces to cloud vision-language models,
normalizing different API formats to work with the OSINT detection pipeline.
"""

import base64
from typing import Dict, List, Any, Optional
from openai import OpenAI


class CloudProviderAdapter:
    """Base adapter for cloud VLM providers."""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def create_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 500,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None
    ) -> Any:
        """Create a chat completion. Must be implemented by subclasses."""
        raise NotImplementedError


class OpenAIAdapter(CloudProviderAdapter):
    """Adapter for OpenAI GPT-4V models (uses native OpenAI SDK)."""

    def __init__(self, model_name: str, api_key: str, base_url: str = "https://api.openai.com/v1/"):
        super().__init__(model_name, api_key)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=180.0,
            max_retries=0
        )

    def create_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 500,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None
    ) -> Any:
        """Create chat completion using OpenAI API."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if logprobs:
            kwargs["logprobs"] = True
            if top_logprobs:
                kwargs["top_logprobs"] = top_logprobs

        return self.client.chat.completions.create(**kwargs)


class AnthropicAdapter(CloudProviderAdapter):
    """Adapter for Anthropic Claude models."""

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key, timeout=180.0)
        except ImportError:
            raise ImportError(
                "Anthropic SDK not installed. Run: pip install anthropic"
            )

    def _convert_messages(self, messages: List[Dict]) -> tuple:
        """
        Convert OpenAI-style messages to Anthropic format.

        Returns:
            (system_prompt, anthropic_messages)
        """
        system_prompt = ""
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                # Convert content format
                if isinstance(msg["content"], str):
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                elif isinstance(msg["content"], list):
                    # Convert multi-part content
                    converted_content = []
                    for part in msg["content"]:
                        if part["type"] == "text":
                            converted_content.append({
                                "type": "text",
                                "text": part["text"]
                            })
                        elif part["type"] == "image_url":
                            # Extract base64 from data URI
                            image_url = part["image_url"]["url"]
                            if image_url.startswith("data:"):
                                # Format: data:image/png;base64,<base64_data>
                                media_type = image_url.split(";")[0].split(":")[1]
                                base64_data = image_url.split(",")[1]
                                converted_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_data
                                    }
                                })

                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": converted_content
                    })

        return system_prompt, anthropic_messages

    def create_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 500,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None
    ) -> Any:
        """Create chat completion using Anthropic API."""
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Note: Claude doesn't support logprobs natively
        # We'll return a mock response object with the text
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=anthropic_messages
        )

        # Convert to OpenAI-like format
        class MockChoice:
            def __init__(self, content):
                self.message = type('obj', (object,), {'content': content})
                self.logprobs = None

        class MockResponse:
            def __init__(self, response):
                self.choices = [MockChoice(response.content[0].text)]

        return MockResponse(response)


class GeminiAdapter(CloudProviderAdapter):
    """Adapter for Google Gemini models."""

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Google GenAI SDK not installed. Run: pip install google-genai"
            )

    def _convert_messages(self, messages: List[Dict]) -> tuple:
        """
        Convert OpenAI-style messages to Gemini format.

        Gemini uses a different message format than OpenAI:
        - System message becomes system_instruction
        - Messages are in user/model alternating format (not user/assistant)
        - Contents are sent as a list of Content objects with role and parts

        Returns:
            (system_instruction, contents_list)
        """
        from google.genai import types
        import base64 as b64

        system_instruction = ""
        contents = []

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                parts = []

                # Process content
                if isinstance(msg["content"], str):
                    # Create a Part object for text
                    parts.append(types.Part(text=msg["content"]))
                elif isinstance(msg["content"], list):
                    for part in msg["content"]:
                        if part["type"] == "text":
                            # Create a Part object for text
                            parts.append(types.Part(text=part["text"]))
                        elif part["type"] == "image_url":
                            # Extract base64 from data URI
                            image_url = part["image_url"]["url"]
                            if image_url.startswith("data:"):
                                # Format: data:image/png;base64,<base64_data>
                                media_type = image_url.split(";")[0].split(":")[1]
                                base64_data = image_url.split(",")[1]
                                image_bytes = b64.b64decode(base64_data)

                                # Add image part using types.Part.from_bytes
                                parts.append(
                                    types.Part.from_bytes(
                                        data=image_bytes,
                                        mime_type=media_type
                                    )
                                )

                # Add user message as Content with role="user"
                contents.append(types.Content(role="user", parts=parts))

            elif msg["role"] == "assistant":
                # Gemini uses "model" instead of "assistant"
                # Assistant messages are always text-only
                text_content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
                contents.append(types.Content(role="model", parts=[types.Part(text=text_content)]))

        return system_instruction, contents

    def create_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 500,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None
    ) -> Any:
        """Create chat completion using Gemini API."""
        from google.genai import types

        system_instruction, contents = self._convert_messages(messages)

        # Build generation config
        config_params = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # NOTE: Logprobs are NOT supported in Gemini Developer API
        # Only available in Vertex AI
        # Attempting to enable them can cause silent failures or empty responses
        # So we intentionally skip logprobs configuration for Gemini

        config = types.GenerateContentConfig(**config_params)

        # Add system instruction if present
        if system_instruction:
            config.system_instruction = system_instruction

        # Generate response using new API
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config
        )

        # Convert to OpenAI-like format
        class MockLogprob:
            def __init__(self, token, logprob):
                self.token = token
                self.logprob = logprob

        class MockLogprobsContent:
            def __init__(self, token, top_logprobs):
                self.token = token
                self.top_logprobs = top_logprobs

        class MockLogprobs:
            def __init__(self, content):
                self.content = content

        class MockChoice:
            def __init__(self, content, gemini_response, logprobs_requested):
                self.message = type('obj', (object,), {'content': content})

                # Gemini Developer API does NOT support logprobs (only Vertex AI does)
                # Always set logprobs to None to avoid confusion
                self.logprobs = None

        class MockResponse:
            def __init__(self, response, logprobs_requested):
                # Get response text, handle case where it might be None or empty
                response_text = None

                # Try to extract text from response
                if hasattr(response, 'text'):
                    response_text = response.text
                elif hasattr(response, 'content'):
                    response_text = response.content
                elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                    # Try to get text from first candidate
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        parts = candidate.content.parts
                        if len(parts) > 0 and hasattr(parts[0], 'text'):
                            response_text = parts[0].text

                # If still None or empty, generate error message
                if not response_text:
                    # Check if response was blocked by safety filters
                    error_msg = "[ERROR: Gemini returned empty response. "
                    if hasattr(response, 'candidates') and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'finish_reason'):
                            error_msg += f"Finish reason: {candidate.finish_reason}. "
                        if hasattr(candidate, 'safety_ratings'):
                            error_msg += f"Safety ratings: {candidate.safety_ratings}. "
                    error_msg += f"Response type: {type(response).__name__}]"
                    response_text = error_msg

                self.choices = [MockChoice(response_text, response, logprobs_requested)]

        return MockResponse(response, logprobs)


def get_cloud_adapter(provider: str, model_name: str, api_key: str, base_url: Optional[str] = None):
    """
    Factory function to get the appropriate cloud provider adapter.

    Args:
        provider: "openai", "anthropic", or "gemini"
        model_name: Model identifier
        api_key: API key for the provider
        base_url: Base URL (only for OpenAI-compatible)

    Returns:
        CloudProviderAdapter instance
    """
    if provider == "openai":
        return OpenAIAdapter(model_name, api_key, base_url or "https://api.openai.com/v1/")
    elif provider == "anthropic":
        return AnthropicAdapter(model_name, api_key)
    elif provider == "gemini":
        return GeminiAdapter(model_name, api_key)
    elif provider == "vllm":
        # For vLLM, return OpenAI adapter (OpenAI-compatible)
        return OpenAIAdapter(model_name, api_key or "dummy", base_url)
    else:
        raise ValueError(f"Unknown provider: {provider}")
