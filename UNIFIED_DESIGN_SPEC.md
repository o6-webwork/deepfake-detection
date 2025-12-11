# Unified Design Specification: Forensic Deepfake Detection Platform

**Version:** 2.0
**Date:** December 11, 2025
**Language:** Python 3.10+
**Framework:** Streamlit

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Feature 1: Forensic Image Scanner (ELA/FFT)](#feature-1-forensic-image-scanner-elafft)
4. [Feature 2: Dynamic Model Configuration UI](#feature-2-dynamic-model-configuration-ui)
5. [Feature 3: Logit Calibration System](#feature-3-logit-calibration-system)
6. [Implementation Modules](#implementation-modules)
7. [Data Flow](#data-flow)
8. [API Specifications](#api-specifications)
9. [UI/UX Specifications](#uiux-specifications)
10. [Security & Best Practices](#security--best-practices)
11. [Testing & Validation](#testing--validation)

---

## Overview

### Mission Statement

Build a **forensically-grounded, VLM-powered deepfake detection platform** that combines:
- **Forensic signal processing** (ELA, FFT) for objective artifact detection
- **Vision-Language Models** as expert forensic interpreters
- **Logit calibration** for scientifically-calibrated confidence scores
- **Flexible model management** for hotswapping local/online models

### Key Differentiators

1. **Glass-box forensic approach**: VLMs analyze explicit forensic artifacts (ELA/FFT), not just visual pixels
2. **Calibrated probabilities**: Extract raw logprobs for true confidence scores, eliminating "Timidity Bias"
3. **Model-agnostic architecture**: Support any OpenAI-compatible API endpoint (local vLLM, OpenAI, Gemini, etc.)
4. **Multi-modal evaluation**: Single image detection + batch evaluation with comprehensive metrics

---

## Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Application                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │   Tab 1:         │  │   Tab 2:         │  │   Tab 3:      │ │
│  │   Detection      │  │   Evaluation     │  │   Config      │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Core Processing Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │  Forensics   │  │  Classifier  │  │  Model Manager      │   │
│  │  (ELA/FFT)   │  │  (Logit Cal) │  │  (Config Handler)   │   │
│  └──────────────┘  └──────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Model Endpoints (OpenAI-compatible)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐ │
│  │  vLLM    │  │  OpenAI  │  │  Gemini  │  │  Custom APIs   │ │
│  │  (Local) │  │  (Cloud) │  │  (Cloud) │  │  (Any)         │ │
│  └──────────┘  └──────────┘  └──────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Core Dependencies:**
```
streamlit>=1.28.0          # Web UI framework
pillow>=10.0.0             # Image processing
opencv-python-headless     # ELA/FFT generation
numpy>=1.24.0              # Numerical operations
pandas>=2.0.0              # Data manipulation
openai>=1.0.0              # OpenAI-compatible client
xlsxwriter>=3.1.0          # Excel export
```

**Optional Dependencies:**
```
google-generativeai        # Gemini API support
scikit-image               # Advanced ELA (if needed)
```

---

## Feature 1: Forensic Image Scanner (ELA/FFT)

### Rationale

**Problem:** Current VLM-based detection relies on subjective visual assessment without explicit forensic evidence.

**Solution:** Generate and provide **Error Level Analysis (ELA)** and **Fast Fourier Transform (FFT)** forensic artifacts to the VLM, transforming it from a "visual guesser" to a "forensic signal interpreter."

### Scientific Foundation

#### Error Level Analysis (ELA)
- **Purpose:** Detect compression level inconsistencies
- **Principle:** AI-generated images have uniform compression across entire image; real photos have varying compression (camera sensors compress differently across regions)
- **Signature:**
  - **Uniform rainbow static** → AI-generated (consistent compression)
  - **Dark with edge noise** → Real photograph (varying compression)

#### Fast Fourier Transform (FFT)
- **Purpose:** Detect frequency domain artifacts
- **Principle:** GANs and diffusion models introduce periodic patterns in frequency domain
- **Signatures:**
  - **Grid/Starfield/Cross patterns** → AI-generated (GAN/Diffusion artifacts)
  - **Chaotic starburst** → Real photograph (natural frequency distribution)

### Module Specification: `forensics.py`

```python
"""
Forensic artifact generation for deepfake detection.
Generates ELA and FFT maps to expose AI generation signatures.
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Union


class ArtifactGenerator:
    """Generate forensic artifacts for image authentication."""

    @staticmethod
    def generate_ela(
        image_input: Union[str, Image.Image, np.ndarray],
        quality: int = 90,
        scale_factor: int = 15
    ) -> bytes:
        """
        Generate Error Level Analysis (ELA) map.

        ELA highlights compression inconsistencies by comparing the original
        image to a recompressed version. Uniform compression (AI-generated)
        appears as rainbow static; varying compression (real photos) shows
        dark regions with edge noise.

        Args:
            image_input: Path to image, PIL Image, or numpy array
            quality: JPEG compression quality (default: 90)
            scale_factor: Amplification factor for visibility (default: 15)

        Returns:
            PNG-encoded bytes of the ELA map

        Algorithm:
            1. Load original image
            2. Compress to JPEG at specified quality
            3. Compute |Original - Compressed|
            4. Amplify differences: diff * scale_factor
            5. Normalize to 0-255 range
            6. Return as PNG bytes
        """
        # Load image
        if isinstance(image_input, str):
            original = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            original = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            original = image_input

        # Compress to JPEG in memory
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, compressed_bytes = cv2.imencode('.jpg', original, encode_params)

        # Decode compressed image
        compressed = cv2.imdecode(compressed_bytes, cv2.IMREAD_COLOR)

        # Compute absolute difference
        diff = cv2.absdiff(original, compressed).astype('float')

        # Scale for visibility
        ela = np.clip(diff * scale_factor, 0, 255).astype('uint8')

        # Convert to PNG bytes
        _, png_bytes = cv2.imencode('.png', ela)

        return png_bytes.tobytes()

    @staticmethod
    def generate_fft(
        image_input: Union[str, Image.Image, np.ndarray]
    ) -> bytes:
        """
        Generate Fast Fourier Transform (FFT) magnitude spectrum.

        FFT reveals frequency domain patterns. AI-generated images often
        exhibit grid, starfield, or cross patterns due to GAN/Diffusion
        architecture. Real photos show chaotic starburst patterns.

        Args:
            image_input: Path to image, PIL Image, or numpy array

        Returns:
            PNG-encoded bytes of the FFT magnitude spectrum

        Algorithm:
            1. Convert to grayscale
            2. Convert to float32
            3. Compute 2D DFT using cv2.dft
            4. Shift zero-frequency to center
            5. Compute magnitude spectrum: 20 * log(magnitude)
            6. Normalize to 0-255 range
            7. Return as PNG bytes
        """
        # Load and convert to grayscale
        if isinstance(image_input, str):
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_input, Image.Image):
            gray = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2GRAY)
        else:
            if len(image_input.shape) == 3:
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_input

        # Convert to float32
        gray_float = np.float32(gray)

        # Compute DFT
        dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)

        # Shift zero-frequency to center
        dft_shift = np.fft.fftshift(dft)

        # Compute magnitude spectrum
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

        # Apply log transform for visibility
        magnitude_log = 20 * np.log(magnitude + 1)  # +1 to avoid log(0)

        # Normalize to 0-255
        magnitude_normalized = cv2.normalize(
            magnitude_log, None, 0, 255, cv2.NORM_MINMAX
        )
        fft_image = np.uint8(magnitude_normalized)

        # Convert to PNG bytes
        _, png_bytes = cv2.imencode('.png', fft_image)

        return png_bytes.tobytes()
```

---

## Feature 2: Dynamic Model Configuration UI

### Rationale

**Problem:** Current model configuration is hardcoded in `config.py`, requiring code changes to add/modify models.

**Solution:** Provide a **Configuration Tab** in the Streamlit UI allowing users to:
1. Add model endpoints via text input (one at a time)
2. Bulk import model configurations via CSV upload
3. Edit/delete existing model configurations
4. Toggle models on/off without removing them
5. Test model connectivity before saving

### UI Specification: Tab 3 - Model Configuration

#### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                   ⚙️ Model Configuration                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Active Models: 4                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ✅ Qwen3 VL 32B          │ http://100.64.0.3:8006/v1/   │   │
│  │ ✅ InternVL 3.5 8B       │ http://localhost:1234/v1/    │   │
│  │ ✅ MiniCPM-V 4.5         │ http://100.64.0.3:8001/v1/   │   │
│  │ ✅ InternVL 2.5 8B       │ http://100.64.0.1:8000/v1/   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  📝 Add Model Manually                                  │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  Display Name:  [_________________________]             │    │
│  │  Base URL:      [_________________________]             │    │
│  │  Model Name:    [_________________________]             │    │
│  │  API Key:       [_________________________] (optional)  │    │
│  │                                                         │    │
│  │  [Test Connection]  [Add Model]                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  📤 Bulk Import via CSV                                 │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  Upload CSV with columns:                               │    │
│  │  - display_name                                         │    │
│  │  - base_url                                             │    │
│  │  - model_name                                           │    │
│  │  - api_key (optional)                                   │    │
│  │                                                         │    │
│  │  [Choose File]  [Import Models]                        │    │
│  │                                                         │    │
│  │  [Download Template CSV]                               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  📋 Manage Existing Models                              │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  [Table with: Name | URL | Status | Actions]            │    │
│  │  Qwen3 VL 32B   | http://100... | ✅ Online | [Edit][❌] │    │
│  │  InternVL 3.5   | http://local..| ✅ Online | [Edit][❌] │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  [💾 Save Configuration]  [🔄 Reset to Defaults]                │
└─────────────────────────────────────────────────────────────────┘
```

### Module Specification: `model_manager.py`

```python
"""
Dynamic model configuration management.
Handles adding, editing, deleting, and testing model endpoints.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI


class ModelManager:
    """Manage model configurations dynamically."""

    CONFIG_FILE = "model_configs.json"

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ModelManager.

        Args:
            config_path: Path to JSON config file (default: model_configs.json)
        """
        self.config_path = Path(config_path or self.CONFIG_FILE)
        self.models = self._load_configs()

    def _load_configs(self) -> Dict:
        """Load model configurations from JSON file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Return default configs from config.py
            return self._get_default_configs()

    def _get_default_configs(self) -> Dict:
        """Return default model configurations."""
        from config import MODEL_CONFIGS
        return MODEL_CONFIGS.copy()

    def save_configs(self):
        """Save current configurations to JSON file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.models, f, indent=2)

    def add_model(
        self,
        model_key: str,
        display_name: str,
        base_url: str,
        model_name: str,
        api_key: str = "dummy",
        enabled: bool = True
    ) -> bool:
        """
        Add a new model configuration.

        Args:
            model_key: Unique identifier for the model
            display_name: Human-readable name
            base_url: OpenAI-compatible API endpoint
            model_name: Model identifier for API calls
            api_key: API key (default: "dummy" for vLLM)
            enabled: Whether model is active

        Returns:
            True if added successfully, False if key exists
        """
        if model_key in self.models:
            return False

        self.models[model_key] = {
            "display_name": display_name,
            "base_url": base_url,
            "model_name": model_name,
            "api_key": api_key,
            "enabled": enabled
        }
        self.save_configs()
        return True

    def remove_model(self, model_key: str) -> bool:
        """Remove a model configuration."""
        if model_key in self.models:
            del self.models[model_key]
            self.save_configs()
            return True
        return False

    def update_model(self, model_key: str, **kwargs) -> bool:
        """Update model configuration fields."""
        if model_key not in self.models:
            return False

        self.models[model_key].update(kwargs)
        self.save_configs()
        return True

    def test_connection(self, model_key: str) -> Dict:
        """
        Test connection to a model endpoint.

        Returns:
            {
                "success": bool,
                "message": str,
                "latency_ms": float (if success)
            }
        """
        import time

        if model_key not in self.models:
            return {"success": False, "message": "Model not found"}

        config = self.models[model_key]

        try:
            client = OpenAI(
                base_url=config["base_url"],
                api_key=config.get("api_key", "dummy")
            )

            start = time.time()

            # Simple test call
            response = client.chat.completions.create(
                model=config["model_name"],
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                temperature=0.0
            )

            latency = (time.time() - start) * 1000  # Convert to ms

            return {
                "success": True,
                "message": "Connection successful",
                "latency_ms": round(latency, 2)
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Connection failed: {str(e)}"
            }

    def import_from_csv(self, csv_path: str) -> Dict:
        """
        Import model configurations from CSV.

        CSV format:
            display_name,base_url,model_name,api_key

        Returns:
            {
                "success": int (count of successful imports),
                "failed": List[str] (failed model names),
                "errors": List[str] (error messages)
            }
        """
        df = pd.read_csv(csv_path)

        required_cols = {"display_name", "base_url", "model_name"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            return {
                "success": 0,
                "failed": [],
                "errors": [f"Missing required columns: {missing}"]
            }

        success_count = 0
        failed_names = []
        errors = []

        for idx, row in df.iterrows():
            # Generate model_key from display_name
            model_key = row['display_name'].replace(" ", "_").replace("-", "_")

            api_key = row.get('api_key', 'dummy')
            if pd.isna(api_key):
                api_key = 'dummy'

            try:
                added = self.add_model(
                    model_key=model_key,
                    display_name=row['display_name'],
                    base_url=row['base_url'],
                    model_name=row['model_name'],
                    api_key=api_key
                )

                if added:
                    success_count += 1
                else:
                    failed_names.append(row['display_name'])
                    errors.append(f"{row['display_name']}: Model key already exists")

            except Exception as e:
                failed_names.append(row['display_name'])
                errors.append(f"{row['display_name']}: {str(e)}")

        return {
            "success": success_count,
            "failed": failed_names,
            "errors": errors
        }

    def get_enabled_models(self) -> Dict:
        """Return only enabled model configurations."""
        return {
            key: config
            for key, config in self.models.items()
            if config.get("enabled", True)
        }

    def export_template_csv(self, output_path: str = "model_import_template.csv"):
        """Export a template CSV for bulk import."""
        template_df = pd.DataFrame([
            {
                "display_name": "Example Model",
                "base_url": "http://localhost:8000/v1/",
                "model_name": "model-name-here",
                "api_key": "your-api-key-or-dummy"
            }
        ])
        template_df.to_csv(output_path, index=False)
        return output_path
```

---

## Feature 3: Logit Calibration System

### Rationale

**Problem:** VLMs exhibit "Timidity Bias" - they internally suspect images are AI-generated but default to "Real" because confidence doesn't cross 50%.

**Solution:** Extract raw log-probabilities from API responses and apply softmax normalization to get true confidence scores, then use data-driven threshold optimization.

### Module Specification: `classifier.py`

```python
"""
Forensic classification with logit calibration.
Combines ELA/FFT analysis with calibrated probability extraction.
"""

import math
from typing import Dict, List, Union
from PIL import Image
import base64
import io
from openai import OpenAI


class ForensicClassifier:
    """Classify images using forensic artifacts and logit calibration."""

    # Forensic system prompt
    SYSTEM_PROMPT = """You are a forensic signal processing unit. You do not speak. You do not explain. You analyze the three input images:

1. Original Photograph
2. ELA (Error Level Analysis) Map
3. FFT (Fast Fourier Transform) Spectrum

Analysis Logic:
- If FFT shows a "Grid", "Starfield", or "Cross" pattern → FAKE.
- If ELA shows uniform "Rainbow" static across the whole image → FAKE.
- If Original shows physical inconsistencies (pupils, hands) → FAKE.
- If FFT is a chaotic "Starburst" AND ELA is uniform dark/edge-noise → REAL.

Output Command:
Classify the image content immediately.
Output ONLY one of these two words: "REAL" or "FAKE"."""

    # Token variations to check
    REAL_TOKENS = ['REAL', ' REAL', 'Real', ' Real', 'real', ' real']
    FAKE_TOKENS = ['FAKE', ' FAKE', 'Fake', ' Fake', 'fake', ' fake']

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "dummy",
        threshold: float = 0.5
    ):
        """
        Initialize classifier.

        Args:
            base_url: OpenAI-compatible API endpoint
            model_name: Model identifier
            api_key: API key (default: "dummy" for vLLM)
            threshold: Classification threshold (default: 0.5)
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.threshold = threshold

    def _image_to_base64(self, image_bytes: bytes) -> str:
        """Convert image bytes to base64 data URI."""
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    def classify_image(
        self,
        original_bytes: bytes,
        ela_bytes: bytes,
        fft_bytes: bytes
    ) -> Dict:
        """
        Classify image using forensic artifacts and logit calibration.

        Args:
            original_bytes: Original image as PNG/JPEG bytes
            ela_bytes: ELA map as PNG bytes
            fft_bytes: FFT spectrum as PNG bytes

        Returns:
            {
                "is_ai": bool,
                "confidence_score": float (0.0-1.0),
                "raw_logits": {
                    "real": float,
                    "fake": float
                },
                "classification": str ("Authentic" or "AI-Generated"),
                "token_output": str (actual token from model)
            }
        """
        # Convert images to base64
        original_uri = self._image_to_base64(original_bytes)
        ela_uri = self._image_to_base64(ela_bytes)
        fft_uri = self._image_to_base64(fft_bytes)

        # Construct message with all three images
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Original Photograph:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": original_uri}
                    },
                    {
                        "type": "text",
                        "text": "ELA (Error Level Analysis) Map:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": ela_uri}
                    },
                    {
                        "type": "text",
                        "text": "FFT (Fast Fourier Transform) Spectrum:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": fft_uri}
                    },
                    {
                        "type": "text",
                        "text": "Classify: REAL or FAKE?"
                    }
                ]
            }
        ]

        # API call with logprobs
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT}
                ] + messages,
                temperature=0.0,  # Deterministic
                max_tokens=1,     # Force single token
                logprobs=True,    # Enable logprob extraction
                top_logprobs=5    # Get top 5 tokens
            )

            # Parse logprobs
            result = self._parse_logprobs(response)

            return result

        except Exception as e:
            # Graceful error handling
            return {
                "is_ai": False,
                "confidence_score": 0.5,
                "raw_logits": {"real": -0.69, "fake": -0.69},
                "classification": "Error",
                "token_output": None,
                "error": str(e)
            }

    def _parse_logprobs(self, response) -> Dict:
        """
        Parse logprobs from API response.

        Handles tokenization variations (REAL vs Real vs real, with/without space).
        """
        try:
            # Extract logprobs from first token
            logprobs_content = response.choices[0].logprobs.content[0]
            top_logprobs = logprobs_content.top_logprobs
            token_output = logprobs_content.token

            # Initialize scores (log space)
            score_real = -100.0
            score_fake = -100.0

            # Scan top logprobs for REAL/FAKE tokens
            for logprob_obj in top_logprobs:
                token = logprob_obj.token
                logprob = logprob_obj.logprob

                if token in self.REAL_TOKENS and score_real == -100.0:
                    score_real = logprob
                elif token in self.FAKE_TOKENS and score_fake == -100.0:
                    score_fake = logprob

            # Convert to linear space
            p_real = math.exp(score_real)
            p_fake = math.exp(score_fake)

            # Softmax normalization
            confidence_fake = p_fake / (p_fake + p_real)

            # Apply threshold
            is_ai = confidence_fake > self.threshold
            classification = "AI-Generated" if is_ai else "Authentic"

            return {
                "is_ai": is_ai,
                "confidence_score": confidence_fake,
                "raw_logits": {
                    "real": score_real,
                    "fake": score_fake
                },
                "classification": classification,
                "token_output": token_output
            }

        except Exception as e:
            # Fallback on parsing error
            return {
                "is_ai": False,
                "confidence_score": 0.5,
                "raw_logits": {"real": -0.69, "fake": -0.69},
                "classification": "Error",
                "token_output": None,
                "error": f"Parsing error: {str(e)}"
            }
```

---

## Data Flow

### Detection Flow (Tab 1)

```
User uploads image
       ↓
Generate ELA map (forensics.py)
       ↓
Generate FFT spectrum (forensics.py)
       ↓
Send all 3 images to VLM (classifier.py)
       ↓
Extract logprobs and calculate confidence
       ↓
Display: Classification + Confidence + Forensic artifacts
```

### Evaluation Flow (Tab 2)

```
User uploads batch images + ground truth CSV
       ↓
Select models from configured endpoints
       ↓
For each image:
  ├─ Generate ELA + FFT
  ├─ Classify with each selected model
  └─ Record: prediction, confidence, ground truth
       ↓
Calculate metrics: Accuracy, Precision, Recall, F1
       ↓
Generate confusion matrices per model
       ↓
Export Excel + visualizations
```

### Configuration Flow (Tab 3)

```
User adds model (manual or CSV)
       ↓
Test connection (optional)
       ↓
Save to model_configs.json
       ↓
Model appears in Tab 1/2 dropdowns
```

---

## API Specifications

### OpenAI-Compatible Endpoint Requirements

All model endpoints must support:

```python
# Standard chat completion
response = client.chat.completions.create(
    model="model-name",
    messages=[...],
    temperature=0.0,
    max_tokens=1,
    logprobs=True,      # REQUIRED for logit calibration
    top_logprobs=5      # REQUIRED (minimum 5)
)

# Response structure
response.choices[0].logprobs.content[0].top_logprobs
# Must return list of objects with:
# - token: str
# - logprob: float
```

**Supported APIs:**
- vLLM (local)
- OpenAI GPT-4o / GPT-4 Turbo
- Google Gemini (via OpenAI compatibility layer)
- Any custom API implementing OpenAI spec

---

## UI/UX Specifications

### Tab 1: Detection (Updated)

**Changes from current:**
1. Display forensic artifacts (ELA + FFT) alongside original
2. Show calibrated confidence score with progress bar
3. Add "View Forensic Analysis" expander with:
   - ELA map
   - FFT spectrum
   - Raw logprob values
   - Confidence breakdown

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  Original Image     │  ELA Map        │  FFT Spectrum       │
│  [Image]            │  [Image]        │  [Image]            │
└─────────────────────────────────────────────────────────────┘

Classification: AI-Generated
Confidence: 87.3% [████████████████████░░░]

▼ View Forensic Analysis
  Raw Logprobs:
    - P(FAKE): -0.14  →  exp(-0.14) = 0.873
    - P(REAL): -1.95  →  exp(-1.95) = 0.142

  Forensic Indicators:
    ✓ FFT shows grid pattern (GAN artifact)
    ✓ ELA shows uniform compression
    ✗ No physical inconsistencies detected
```

### Tab 2: Evaluation (Minor Updates)

**Changes:**
- Model selector now pulls from `ModelManager.get_enabled_models()`
- Add "Refresh Models" button to reload from config
- Display confidence scores in results table

### Tab 3: Configuration (New)

See [Feature 2 UI Specification](#ui-specification-tab-3---model-configuration)

---

## Security & Best Practices

### Docker Security

**Non-Root User Execution (CRITICAL)**

Containers should never run as root to limit security impact of potential compromises.

```dockerfile
# Create non-root user
RUN useradd -m -s /bin/bash --uid 1001 appuser

# Set ownership of files
COPY --chown=appuser:appuser app.py .
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser
```

**Benefits:**
- Limits blast radius if container is compromised
- Prevents privilege escalation attacks
- Follows principle of least privilege
- Required for many enterprise security policies

### Code Quality

**Temperature Settings**

- **Forensic Analysis:** Use `temperature=0.0` for deterministic, reproducible results
- **Chat Interactions:** Use `temperature=0.7` to allow natural conversation
- **Rationale:** Forensic analysis requires consistency across runs; chat benefits from variability

```python
# Forensic classification
response = client.chat.completions.create(
    model=model_name,
    messages=[...],
    temperature=0.0  # Deterministic for forensic analysis
)

# Chat interaction
response = client.chat.completions.create(
    model=model_name,
    messages=[...],
    temperature=0.7  # Allow creativity for chat
)
```

**Import Style (PEP 8 Compliance)**

Follow PEP 8 style guide for clean, readable code:

```python
# Bad
import io, tempfile, cv2, pandas as pd

# Good
import io
import tempfile
import cv2
import pandas as pd
```

**Resource Cleanup**

Always clean up temporary resources using try/finally blocks:

```python
# Temporary file handling
tmp_path = None
try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    # Process file
    process_video(tmp_path)
finally:
    # Ensure cleanup even if errors occur
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.unlink(tmp_path)
        except Exception:
            pass  # Ignore cleanup errors
```

### Performance Optimization

**DataFrame Lookups**

Convert DataFrame columns to sets for O(1) lookup instead of O(N):

```python
# Bad: O(N) lookup inside loop
for img_file in eval_images:
    filename = img_file.name
    if filename not in gt_df["filename"].values:  # O(N) lookup
        continue

# Good: O(1) lookup with set
gt_filenames = set(gt_df["filename"].values)  # One-time conversion
for img_file in eval_images:
    filename = img_file.name
    if filename not in gt_filenames:  # O(1) lookup
        continue
```

**Impact:** For dataset with N images, reduces complexity from O(N²) to O(N).

### Configuration Management

**Environment Variables vs Hardcoding**

The `model_manager.py` module supersedes hardcoded configurations by providing:
- Dynamic model addition/removal via UI
- Persistent JSON-based configuration
- No code changes required for new models
- Support for environment-specific endpoints

**Rationale:** While environment variables are useful for deployment-time configuration, the ModelManager provides runtime flexibility without redeployment.

### Documentation Standards

**Docker Compose v2 Syntax**

Use modern `docker compose` (v2) syntax instead of legacy `docker-compose` (v1):

```bash
# Modern syntax (v2)
docker compose up --build
docker compose logs -f
docker compose down

# Legacy syntax (v1) - avoid
docker-compose up --build
```

**Benefits:**
- v2 is integrated into Docker CLI (no separate installation)
- Improved performance and features
- Better compatibility with modern Docker versions

### Repository Hygiene

**Remove Obsolete Files**

- Keep only actively used scripts (e.g., `generate_report_updated.py`)
- Remove deprecated versions (e.g., `generate_report.py`)
- Reduces confusion and maintenance burden
- Keeps repository clean and navigable

**Package Management**

- Remove duplicate dependencies in Dockerfile
- Use `--no-install-recommends` for minimal image size
- Clean apt cache after installation

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    # No duplicate entries
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
```

---

## Testing & Validation

### Unit Tests

**forensics.py:**
- Test ELA generation with known AI/real images
- Test FFT generation with synthetic grid patterns
- Verify output format (PNG bytes)

**classifier.py:**
- Test logprob parsing with mock responses
- Test token variation handling (REAL vs real vs Real)
- Test threshold application

**model_manager.py:**
- Test add/remove/update operations
- Test CSV import with valid/invalid data
- Test connection testing with mock endpoints

### Integration Tests

**End-to-End Detection:**
1. Upload known AI image → Expect "AI-Generated" with high confidence
2. Upload known real photo → Expect "Authentic" with high confidence
3. Upload edge case → Verify forensic artifacts visible

**Batch Evaluation:**
1. Run on validation set with ground truth
2. Verify metrics match manual calculation
3. Test with multiple models simultaneously

### Validation Criteria

**Forensic Artifacts:**
- ELA maps visually distinct for AI vs real images
- FFT spectra show expected patterns (grid for AI, starburst for real)

**Calibration:**
- Confidence scores correlate with accuracy (well-calibrated)
- Histogram of scores shows separation between classes

**Model Management:**
- All OpenAI-compatible endpoints work correctly
- CSV import handles errors gracefully

---

## Implementation Roadmap

### Phase 1: Core Forensic System (Week 1)
- [ ] Implement `forensics.py` (ELA + FFT generation)
- [ ] Implement `classifier.py` (logit calibration)
- [ ] Update Tab 1 to display forensic artifacts
- [ ] Test on sample images

### Phase 2: Model Management (Week 1-2)
- [ ] Implement `model_manager.py`
- [ ] Create Tab 3 UI (configuration interface)
- [ ] Add manual model addition
- [ ] Add CSV bulk import
- [ ] Add connection testing

### Phase 3: Integration (Week 2)
- [ ] Integrate `ModelManager` with Tabs 1 & 2
- [ ] Update model selectors to use dynamic configs
- [ ] Add "Refresh Models" functionality
- [ ] Test cross-tab integration

### Phase 4: Evaluation Enhancements (Week 2-3)
- [ ] Update batch evaluation to use forensic classification
- [ ] Add confidence score columns to results
- [ ] Generate calibration plots (histogram of scores)
- [ ] Update report generation with new metrics

### Phase 5: Testing & Documentation (Week 3)
- [ ] Write unit tests for all new modules
- [ ] Run integration tests with real models
- [ ] Create user documentation
- [ ] Update README with new features

---

## Appendix

### CSV Import Template

```csv
display_name,base_url,model_name,api_key
GPT-4o,https://api.openai.com/v1/,gpt-4o,sk-your-key-here
Qwen3 VL Local,http://localhost:8000/v1/,Qwen3-VL-32B-Instruct,dummy
Gemini Pro Vision,https://generativelanguage.googleapis.com/v1beta/,gemini-pro-vision,your-gemini-key
```

### Configuration File Schema (`model_configs.json`)

```json
{
  "model_key": {
    "display_name": "Human-readable name",
    "base_url": "https://api.endpoint.com/v1/",
    "model_name": "model-identifier",
    "api_key": "api-key-or-dummy",
    "enabled": true
  }
}
```

### Environment Variables

```bash
# Optional: Set default API keys
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."

# Optional: Set default model config path
export MODEL_CONFIG_PATH="/path/to/model_configs.json"
```

---

## Version History

**v2.0 (2025-12-11)**
- Added forensic scanner design (ELA/FFT)
- Added logit calibration system
- Added dynamic model configuration UI
- Consolidated all feature specs into unified document

**v1.0 (2025-11-26)**
- Initial batch evaluation system
- Docker containerization
- Report generation

---

**Document Status:** ✅ Ready for Implementation
**Next Step:** Phase 1 - Core Forensic System Implementation
