# Gemini API Integration Fix

## Issues Fixed

### 1. AttributeError: 'GeminiAdapter' object has no attribute 'chat'

**Root Cause:** Two separate issues:
- Using deprecated `google-generativeai` SDK
- Code bypassing adapter pattern and calling `.chat.completions.create()` directly

**Solution:**
- Updated to new `google-genai` SDK (v0.2.0+)
- Added conditional logic to use adapter's `create_completion()` method for cloud providers

### 2. Request 2 (A/B Classification) Returning ~0 Tokens

**Root Cause:**
- The `_convert_messages()` function in GeminiAdapter was **ignoring assistant messages**
- Gemini only received the system instruction and the verdict prompt ("Answer A or B"), but had no context of what it had just analyzed
- Without the conversation history, Gemini couldn't provide an answer, resulting in empty responses (~0 tokens)

**Solution:**
- Updated `_convert_messages()` to properly handle multi-turn conversations
- Assistant messages are now converted to Gemini's "model" role
- Messages are structured as `Content` objects with proper role/parts format
- This ensures Gemini receives the full conversation context for Request 2

### 3. Logprobs Not Available in Developer API

**Root Cause:**
- Gemini Developer API does NOT support logprobs (only Vertex AI does)
- Attempting to enable logprobs can cause silent failures or confusing error messages

**Solution:**
- Removed logprobs configuration from GeminiAdapter
- Added clear documentation that logprobs are not available
- Fallback verdict extraction uses text parsing to find A/B classification
- Confidence score shows "not available" instead of synthetic 50%

## Changes Made

### File: cloud_providers.py

**1. GeminiAdapter.__init__:**
```python
# Before (deprecated)
import google.generativeai as genai
genai.configure(api_key=api_key)
self.client = genai.GenerativeModel(model_name)

# After (new SDK)
from google import genai
self.client = genai.Client(api_key=api_key)
```

**2. GeminiAdapter._convert_messages - Added Assistant Message Handling:**
```python
# NEW: Handle assistant messages in conversation history
elif msg["role"] == "assistant":
    # Gemini uses "model" instead of "assistant"
    text_content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
    contents.append(types.Content(role="model", parts=[text_content]))

# NEW: Structure messages as Content objects
contents.append(types.Content(role="user", parts=parts))
```

**3. GeminiAdapter.create_completion - Removed Logprobs Configuration:**
```python
# Build generation config
config_params = {
    "temperature": temperature,
    "max_output_tokens": max_tokens,
}

# NOTE: Logprobs are NOT supported in Gemini Developer API
# Only available in Vertex AI
# Attempting to enable them can cause silent failures or empty responses
# So we intentionally skip logprobs configuration for Gemini
```

**4. MockChoice - Simplified to Always Return None for Logprobs:**
```python
class MockChoice:
    def __init__(self, content, gemini_response, logprobs_requested):
        self.message = type('obj', (object,), {'content': content})

        # Gemini Developer API does NOT support logprobs (only Vertex AI does)
        # Always set logprobs to None to avoid confusion
        self.logprobs = None
```

**5. MockResponse - Enhanced Error Handling:**
```python
# Try to extract text from multiple possible response structures
if hasattr(response, 'text'):
    response_text = response.text
elif hasattr(response, 'candidates') and len(response.candidates) > 0:
    candidate = response.candidates[0]
    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
        parts = candidate.content.parts
        if len(parts) > 0 and hasattr(parts[0], 'text'):
            response_text = parts[0].text

# If empty, check for safety filter blocks or errors
if not response_text:
    error_msg = "[ERROR: Gemini returned empty response. "
    if hasattr(candidate, 'finish_reason'):
        error_msg += f"Finish reason: {candidate.finish_reason}. "
    # ... more debugging info
```

### File: detector.py

**SPAI-assisted detection API calls:**
```python
# Use adapter pattern for cloud providers
if hasattr(self.client, 'create_completion'):
    # Cloud provider adapter (Gemini, Anthropic)
    response = self.client.create_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=2000
    )
else:
    # Direct OpenAI client (vLLM, OpenAI)
    response = self.client.chat.completions.create(
        model=self.model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=2000
    )
```

### File: requirements.txt

```diff
- google-generativeai>=0.8.0
+ google-genai>=0.2.0
```

## Gemini Logprobs Format

### Request
```python
config = types.GenerateContentConfig(
    temperature=0.0,
    max_output_tokens=1,
    response_logprobs=True,  # Enable logprobs
    logprobs=5               # Return top 5 alternatives
)
```

### Response Structure
```python
response.candidates[0].logprobs_result = {
    "chosen_candidates": [
        {
            "token": "A",
            "log_probability": -0.1
        }
    ],
    "top_candidates": [
        {
            "candidates": [
                {"token": "A", "log_probability": -0.1},
                {"token": "B", "log_probability": -2.3},
                ...
            ]
        }
    ]
}
```

### OpenAI-Compatible Conversion
```python
response.choices[0].logprobs.content[0] = {
    "token": "A",
    "top_logprobs": [
        {"token": "A", "logprob": -0.1},
        {"token": "B", "logprob": -2.3},
        ...
    ]
}
```

## Important: Logprobs Availability

**Logprobs is only available in Vertex AI, NOT in the Gemini Developer API.**

- **Vertex AI:** Full logprobs support with token-level probabilities
- **Developer API:** No native logprobs support

### Behavior with Developer API

Since the Developer API doesn't support native logprobs:
- Logprobs will show as "Not available"
- Confidence scores will default to neutral (50%)
- The VLM analysis text will still be provided

For full logprobs support, use Vertex AI instead of the Developer API.

## Testing

To verify the fix:

1. **Install google-genai SDK:**
   ```bash
   pip install google-genai>=0.2.0
   ```

2. **Configure Gemini in models.json:**

   **For Developer API (most common):**
   ```json
   {
     "gemini-2.0-flash-exp": {
       "provider": "gemini",
       "model_name": "gemini-2.0-flash-exp",
       "api_key": "YOUR_API_KEY"
     }
   }
   ```

3. **Test SPAI-assisted detection:**
   - Upload an image
   - Select "SPAI + VLM Analysis"
   - Choose Gemini model
   - Verify that Request 2 now generates a response (should see >0 tokens)

## Expected Behavior After Fix

**Request 1 (Analysis):**
- Latency: ~20-30s (full image analysis)
- Tokens: ~1500-2000 (includes image tokens + analysis text)
- Response: Full detailed analysis of the image

**Request 2 (Verdict - FIXED):**
- Latency: ~1-2s (KV-cache optimized)
- Tokens: ~10-50 (short A/B response) - **NO LONGER ~0!**
- Response: "A" or "B" with optional explanation

**Output Example (Developer API - text-based A/B extraction):**
```
VLM Analysis: [Full analysis text from Gemini Request 1]

Verdict: A
Classification: Authentic
Confidence: not available (Gemini Developer API doesn't support logprobs)

Note: The A/B verdict is now successfully extracted from Request 2.
For calibrated confidence scores, use Vertex AI instead of Developer API.
```

**Why Logprobs Don't Work:**
- Gemini Developer API does NOT support the `logprobs` parameter
- Only Vertex AI (enterprise) API supports logprobs
- The system will fall back to text-based A/B extraction
- Confidence score will show "not available" instead of a percentage

## References

- [Gemini Logprobs Documentation (Vertex AI)](https://developers.googleblog.com/unlock-gemini-reasoning-with-logprobs-on-vertex-ai/)
- [Google GenAI Python SDK](https://ai.google.dev/gemini-api/docs/text-generation)
- [Gemini Vision API](https://ai.google.dev/gemini-api/docs/vision)
- [Logprobs Limitations (Developer Forum)](https://discuss.ai.google.dev/t/logprobs-is-not-enabled-for-gemini-models/107989)
