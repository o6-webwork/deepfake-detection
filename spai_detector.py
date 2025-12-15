"""
SPAI (Spectral AI-Generated Image Detector) wrapper module.

This module provides a clean interface to the SPAI CVPR2025 model for deepfake detection
using spectral learning. Replaces ELA/FFT forensic analysis with state-of-the-art
frequency-domain detection.

SPAI analyzes the spectral distribution of images using masked feature modeling with
Vision Transformers to detect AI-generated content with high accuracy.
"""

import torch
import cv2
import numpy as np
from PIL import Image
import io
import logging
from pathlib import Path
from typing import Dict, Optional

# Import SPAI components from spai/ directory
import sys
sys.path.insert(0, str(Path(__file__).parent / 'spai'))

try:
    from spai.config import get_config
    from spai.models import build_cls_model
    from spai.utils import load_pretrained
    from spai.data.data_finetune import build_transform
except ImportError as e:
    raise ImportError(
        f"Failed to import SPAI modules. Ensure spai/ directory exists with required files. Error: {e}"
    )

# Configure logging
logger = logging.getLogger(__name__)


class SPAIDetector:
    """
    SPAI spectral deepfake detector.

    Uses masked feature modeling with Vision Transformers to analyze frequency-domain
    patterns characteristic of AI-generated images.

    Replaces traditional ELA/FFT forensics with learned spectral representations.
    """

    def __init__(
        self,
        config_path: str = "spai/configs/spai.yaml",
        weights_path: str = "spai/weights/spai.pth",
        device: Optional[str] = None
    ):
        """
        Initialize SPAI model.

        Args:
            config_path: Path to SPAI YAML config file
            weights_path: Path to pre-trained SPAI weights
            device: Device to run model on ('cuda' or 'cpu'). Auto-detects if None.

        Raises:
            FileNotFoundError: If weights file not found
            RuntimeError: If model initialization fails
        """
        # Validate paths
        config_path = Path(config_path)
        weights_path = Path(weights_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"SPAI config not found at {config_path}. "
                f"Ensure spai/configs/spai.yaml exists."
            )

        if not weights_path.exists():
            raise FileNotFoundError(
                f"SPAI weights not found at {weights_path}.\n"
                f"Download from: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view\n"
                f"Place in: spai/weights/spai.pth"
            )

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logger.info(f"Initializing SPAI on device: {self.device}")

        # Load config
        try:
            self.config = get_config({
                "cfg": str(config_path),
                "batch_size": 1,
                "opts": []
            })
        except Exception as e:
            raise RuntimeError(f"Failed to load SPAI config: {e}")

        # Build model
        try:
            self.model = build_cls_model(self.config)
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to build SPAI model: {e}")

        # Load pre-trained weights
        try:
            load_pretrained(
                self.config,
                self.model,
                logger,
                checkpoint_path=str(weights_path),
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SPAI weights: {e}")

        self.model.eval()

        # Build transform pipeline
        try:
            self.transform = build_transform(self.config, is_train=False)
        except Exception as e:
            raise RuntimeError(f"Failed to build SPAI transform: {e}")

        logger.info("SPAI initialized successfully")

    def analyze(
        self,
        image_bytes: bytes,
        generate_heatmap: bool = True,
        alpha: float = 0.6,
        max_size: Optional[int] = 1280
    ) -> Dict:
        """
        Run SPAI spectral analysis on image.

        Args:
            image_bytes: Input image as bytes (PNG/JPEG)
            generate_heatmap: If True, generate blended attention overlay
            alpha: Transparency for blending (0.0-1.0)
                   Higher alpha = more original image visible
                   Default 0.6 = 60% original, 40% heatmap
            max_size: Maximum resolution for longest edge (512-2048 or None for original)
                     Higher = more accurate but slower. Default 1280.

        Returns:
            {
                "spai_score": float,         # 0.0-1.0 (AI generation probability)
                "spai_prediction": str,      # "Real" or "AI Generated"
                "spai_confidence": float,    # 0.0-1.0 (confidence in prediction)
                "tier": str,                 # "Authentic" / "Suspicious" / "Deepfake"
                "heatmap_bytes": bytes,      # Blended overlay PNG (if requested)
                "analysis_text": str         # Human-readable analysis report
            }

        Raises:
            ValueError: If image cannot be loaded
            RuntimeError: If inference fails
        """
        # Clone config and set resolution
        config = self.config.clone()
        config.defrost()
        config.TEST.MAX_SIZE = max_size  # None = original resolution
        if generate_heatmap:
            config.MODEL.RESOLUTION_MODE = "arbitrary"  # Required for attention extraction
        config.freeze()

        # Rebuild transform with updated config
        transform = build_transform(config, is_train=False)

        # Load image
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

        # Preprocess
        try:
            tensor = transform(pil_image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess image: {e}")

        # Run inference
        try:
            with torch.no_grad():
                output = self.model(tensor)
                score = torch.sigmoid(output).item()  # Convert logits to probability
        except Exception as e:
            raise RuntimeError(f"SPAI inference failed: {e}")

        # Determine prediction and confidence
        prediction = "AI Generated" if score >= 0.5 else "Real"
        confidence = score if score >= 0.5 else (1.0 - score)

        # Map to three-tier system
        tier = self._map_score_to_tier(score)

        # Generate heatmap if requested
        heatmap_bytes = None
        if generate_heatmap:
            try:
                heatmap_bytes = self._generate_blended_heatmap(
                    tensor,
                    pil_image,
                    alpha
                )
            except Exception as e:
                logger.warning(f"Heatmap generation failed: {e}. Continuing without heatmap.")
                heatmap_bytes = None

        # Format analysis text
        analysis_text = self._format_analysis(score, prediction, confidence, tier)

        return {
            "spai_score": score,
            "spai_prediction": prediction,
            "spai_confidence": confidence,
            "tier": tier,
            "heatmap_bytes": heatmap_bytes,
            "analysis_text": analysis_text
        }

    def _generate_blended_heatmap(
        self,
        tensor: torch.Tensor,
        original_image: Image.Image,
        alpha: float
    ) -> bytes:
        """
        Generate blended attention overlay.

        Creates a semi-transparent heatmap showing regions with spectral anomalies,
        blended with the original image using cv2.addWeighted.

        This matches the implementation in spai/app.py create_transparent_overlay().

        Args:
            tensor: Model input tensor
            original_image: Original PIL Image
            alpha: Transparency weight (0.0-1.0)
                   0.6 = 60% original, 40% heatmap

        Returns:
            PNG bytes of blended overlay

        Raises:
            RuntimeError: If heatmap generation fails
        """
        try:
            # Get attention maps from model
            # Note: This assumes the SPAI model has a get_attention_maps method
            # If not available, we'll need to hook into the model's internals
            with torch.no_grad():
                # Attempt to get attention maps
                # This may need adjustment based on actual SPAI model architecture
                if hasattr(self.model, 'get_attention_maps'):
                    attention_maps = self.model.get_attention_maps(tensor)
                else:
                    # Fallback: run inference again and extract attention from hooks
                    logger.warning("Model doesn't expose get_attention_maps, using fallback")
                    attention_maps = self._extract_attention_fallback(tensor)

            # Aggregate attention across layers/heads
            # Shape typically: [batch, layers, height, width] or similar
            if attention_maps.dim() == 4:
                aggregated = attention_maps.mean(dim=1).squeeze(0)  # Average over layers
            elif attention_maps.dim() == 3:
                aggregated = attention_maps.mean(dim=0)  # Average over layers/heads
            else:
                aggregated = attention_maps.squeeze()

            # Resize to original image size
            heatmap = torch.nn.functional.interpolate(
                aggregated.unsqueeze(0).unsqueeze(0),
                size=(original_image.height, original_image.width),
                mode='bilinear',
                align_corners=False
            ).squeeze()

            # Convert to numpy and normalize to 0-255
            heatmap_np = heatmap.cpu().numpy()
            heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
            heatmap_np = (heatmap_np * 255).clip(0, 255).astype(np.uint8)

            # Apply colormap (JET: red = suspicious, blue = normal)
            heatmap_colored = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

            # Convert BGR (OpenCV default) to RGB to match PIL
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            # Convert original PIL Image to numpy array
            original_np = np.array(original_image)

            # Ensure dimensions match
            if heatmap_colored.shape[:2] != original_np.shape[:2]:
                heatmap_colored = cv2.resize(
                    heatmap_colored,
                    (original_np.shape[1], original_np.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

            # Blend using cv2.addWeighted (same as spai/app.py)
            # alpha: weight of original (background)
            # beta: weight of heatmap (foreground)
            beta = 1.0 - alpha
            blended = cv2.addWeighted(original_np, alpha, heatmap_colored, beta, 0)

            # Convert back to BGR for PNG encoding
            blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

            # Encode as PNG
            success, png_bytes = cv2.imencode('.png', blended_bgr)

            if not success:
                raise RuntimeError("Failed to encode heatmap as PNG")

            return png_bytes.tobytes()

        except Exception as e:
            raise RuntimeError(f"Heatmap generation failed: {e}")

    def _extract_attention_fallback(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Fallback method to extract attention if model doesn't expose it directly.

        This is a placeholder - actual implementation depends on SPAI model architecture.
        """
        # For now, return a dummy attention map based on model output gradients
        # TODO: Implement proper attention extraction based on SPAI architecture
        logger.warning("Using dummy attention map - implement proper extraction")

        # Create a simple gradient-based attention map
        tensor.requires_grad_(True)
        output = self.model(tensor)
        output.backward()

        # Use input gradients as proxy for attention
        attention = tensor.grad.abs().mean(dim=1)  # Average over channels

        return attention

    def _map_score_to_tier(self, score: float) -> str:
        """
        Map SPAI score to three-tier risk classification.

        Args:
            score: SPAI score (0.0-1.0, higher = more likely AI-generated)

        Returns:
            "Authentic", "Suspicious", or "Deepfake"
        """
        if score >= 0.9:
            return "Deepfake"
        elif score >= 0.5:
            return "Suspicious"
        else:
            return "Authentic"

    def _format_analysis(
        self,
        score: float,
        prediction: str,
        confidence: float,
        tier: str
    ) -> str:
        """
        Format SPAI analysis as human-readable text report.

        Args:
            score: SPAI score (0.0-1.0)
            prediction: "Real" or "AI Generated"
            confidence: Confidence level (0.0-1.0)
            tier: Risk tier classification

        Returns:
            Formatted analysis text
        """
        analysis = f"""--- SPAI SPECTRAL ANALYSIS ---

Prediction: {prediction}
SPAI Score: {score:.3f} (0.0=Real, 1.0=AI-Generated)
Confidence: {confidence:.1%}
Risk Tier: {tier}

Spectral Analysis Summary:
The image's frequency spectrum was analyzed using masked feature modeling with
Vision Transformer architecture (CVPR2025). The spectral reconstruction similarity
score indicates {self._get_likelihood_text(score)}.

"""

        # Add interpretation based on score
        if score >= 0.9:
            analysis += "🚨 HIGH CONFIDENCE AI-GENERATED: Strong spectral artifacts detected.\n"
            analysis += "   The frequency distribution shows clear signatures of generative models.\n"
        elif score >= 0.7:
            analysis += "⚠️ LIKELY AI-GENERATED: Moderate spectral inconsistencies found.\n"
            analysis += "   Frequency patterns deviate from natural image distributions.\n"
        elif score >= 0.5:
            analysis += "⚠️ SUSPICIOUS: Subtle spectral anomalies present.\n"
            analysis += "   Some frequency components suggest possible AI generation.\n"
        elif score >= 0.3:
            analysis += "✓ LIKELY REAL: Spectral patterns consistent with natural images.\n"
            analysis += "   Frequency distribution matches authentic photography characteristics.\n"
        else:
            analysis += "✅ HIGH CONFIDENCE REAL: Natural spectral distribution detected.\n"
            analysis += "   Frequency patterns strongly indicate authentic capture.\n"

        analysis += "\nNote: SPAI analyzes frequency-domain patterns invisible to human perception.\n"
        analysis += "      These spectral signatures are characteristic of AI generation processes."

        return analysis

    def _get_likelihood_text(self, score: float) -> str:
        """
        Convert SPAI score to likelihood description.

        Args:
            score: SPAI score (0.0-1.0)

        Returns:
            Human-readable likelihood text
        """
        if score >= 0.9:
            return "very high likelihood of AI generation"
        elif score >= 0.7:
            return "high likelihood of AI generation"
        elif score >= 0.5:
            return "moderate likelihood of AI generation"
        elif score >= 0.3:
            return "low likelihood of AI generation"
        else:
            return "very low likelihood of AI generation"

    def __repr__(self) -> str:
        """String representation of SPAIDetector."""
        return f"SPAIDetector(device={self.device}, model_loaded={self.model is not None})"
