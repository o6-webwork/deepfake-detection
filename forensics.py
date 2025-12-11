"""
Forensic artifact generation for deepfake detection.

This module generates Error Level Analysis (ELA) and Fast Fourier Transform (FFT)
forensic maps to expose AI generation signatures in images.

Scientific Rationale:
    - ELA: Detects compression inconsistencies. AI-generated images have uniform
      compression levels (rainbow static), while real photos show varying compression
      (dark regions with edge noise) due to camera sensor behavior.

    - FFT: Reveals frequency domain patterns. GANs and diffusion models introduce
      periodic artifacts (grid/starfield/cross patterns), while real photos exhibit
      chaotic starburst patterns in frequency space.
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Union, Tuple


class ArtifactGenerator:
    """
    Generate forensic artifacts for image authentication analysis.

    This class provides static methods to create ELA and FFT maps that reveal
    forensic signatures indicative of AI generation or manipulation.
    """

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
            image_input: Input image as file path, PIL Image, or numpy array
            quality: JPEG compression quality for recompression (default: 90)
                    Lower quality = more visible differences
                    Range: 0-100
            scale_factor: Amplification factor for visibility (default: 15)
                         Higher value = more visible differences
                         Typical range: 10-20

        Returns:
            PNG-encoded bytes of the ELA map (can be displayed or saved)

        Algorithm:
            1. Load original image as numpy array
            2. Compress to JPEG at specified quality in memory
            3. Decode the compressed JPEG back to array
            4. Compute absolute difference: |Original - Compressed|
            5. Amplify differences: diff * scale_factor
            6. Clip to valid range [0, 255]
            7. Return as PNG bytes

        Example:
            >>> ela_bytes = ArtifactGenerator.generate_ela('photo.jpg')
            >>> with open('photo_ela.png', 'wb') as f:
            ...     f.write(ela_bytes)
        """
        # Load image to numpy array
        if isinstance(image_input, str):
            original = cv2.imread(image_input)
            if original is None:
                raise ValueError(f"Could not load image from path: {image_input}")
        elif isinstance(image_input, Image.Image):
            # Convert PIL Image to OpenCV format (RGB -> BGR)
            original = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            original = image_input.copy()
        else:
            raise TypeError(
                "image_input must be str (path), PIL.Image, or numpy.ndarray"
            )

        # Validate quality parameter
        if not 0 <= quality <= 100:
            raise ValueError(f"quality must be between 0 and 100, got {quality}")

        # Compress to JPEG in memory
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, compressed_bytes = cv2.imencode('.jpg', original, encode_params)

        if not success:
            raise RuntimeError("Failed to encode image as JPEG")

        # Decode compressed image back to array
        compressed = cv2.imdecode(compressed_bytes, cv2.IMREAD_COLOR)

        if compressed is None:
            raise RuntimeError("Failed to decode compressed JPEG")

        # Ensure dimensions match (should always match, but safety check)
        if original.shape != compressed.shape:
            raise RuntimeError(
                f"Shape mismatch: original {original.shape} vs "
                f"compressed {compressed.shape}"
            )

        # Compute absolute difference
        diff = cv2.absdiff(original, compressed).astype('float')

        # Scale for visibility
        ela = np.clip(diff * scale_factor, 0, 255).astype('uint8')

        # Encode as PNG (lossless for artifact preservation)
        success, png_bytes = cv2.imencode('.png', ela)

        if not success:
            raise RuntimeError("Failed to encode ELA map as PNG")

        return png_bytes.tobytes()

    @staticmethod
    def generate_fft(
        image_input: Union[str, Image.Image, np.ndarray]
    ) -> bytes:
        """
        Generate Fast Fourier Transform (FFT) magnitude spectrum.

        FFT reveals frequency domain patterns. AI-generated images often
        exhibit grid, starfield, or cross patterns due to GAN/Diffusion
        architecture artifacts. Real photos show chaotic starburst patterns
        characteristic of natural scenes.

        Args:
            image_input: Input image as file path, PIL Image, or numpy array

        Returns:
            PNG-encoded bytes of the FFT magnitude spectrum

        Algorithm:
            1. Convert image to grayscale
            2. Convert to float32 for DFT computation
            3. Compute 2D Discrete Fourier Transform (DFT)
            4. Shift zero-frequency component to center (fftshift)
            5. Compute magnitude spectrum from complex output
            6. Apply log transform: 20 * log(magnitude + 1)
               - Log scale makes patterns more visible
               - +1 prevents log(0) errors
            7. Normalize to 0-255 range for visualization
            8. Return as PNG bytes

        Frequency Domain Signatures:
            - Grid patterns: Regular GAN artifacts
            - Starfield/Cross: Diffusion model artifacts
            - Chaotic starburst: Natural image frequencies

        Example:
            >>> fft_bytes = ArtifactGenerator.generate_fft('photo.jpg')
            >>> with open('photo_fft.png', 'wb') as f:
            ...     f.write(fft_bytes)
        """
        # Load and convert to grayscale
        if isinstance(image_input, str):
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Could not load image from path: {image_input}")
        elif isinstance(image_input, Image.Image):
            # Convert PIL Image to grayscale
            if image_input.mode == 'L':
                gray = np.array(image_input)
            else:
                gray = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2GRAY)
        elif isinstance(image_input, np.ndarray):
            # Convert to grayscale if needed
            if len(image_input.shape) == 2:
                gray = image_input.copy()
            elif len(image_input.shape) == 3:
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(
                    f"Invalid image array shape: {image_input.shape}"
                )
        else:
            raise TypeError(
                "image_input must be str (path), PIL.Image, or numpy.ndarray"
            )

        # Convert to float32 for DFT
        gray_float = np.float32(gray)

        # Compute DFT
        # Output is complex array with 2 channels: real and imaginary
        dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)

        # Shift zero-frequency component to center
        # This places DC component (average brightness) at image center
        dft_shift = np.fft.fftshift(dft)

        # Compute magnitude spectrum from complex DFT
        # magnitude = sqrt(real^2 + imaginary^2)
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

        # Apply log transform for better visibility
        # Log scale compresses dynamic range, making subtle patterns visible
        magnitude_log = 20 * np.log(magnitude + 1)  # +1 to avoid log(0)

        # Normalize to 0-255 range for visualization
        magnitude_normalized = cv2.normalize(
            magnitude_log,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )
        fft_image = np.uint8(magnitude_normalized)

        # Encode as PNG
        success, png_bytes = cv2.imencode('.png', fft_image)

        if not success:
            raise RuntimeError("Failed to encode FFT spectrum as PNG")

        return png_bytes.tobytes()

    @staticmethod
    def generate_both(
        image_input: Union[str, Image.Image, np.ndarray],
        ela_quality: int = 90,
        ela_scale: int = 15
    ) -> Tuple[bytes, bytes]:
        """
        Convenience method to generate both ELA and FFT in one call.

        Args:
            image_input: Input image as file path, PIL Image, or numpy array
            ela_quality: JPEG quality for ELA (default: 90)
            ela_scale: Scale factor for ELA visibility (default: 15)

        Returns:
            Tuple of (ela_bytes, fft_bytes) as PNG-encoded bytes

        Example:
            >>> ela, fft = ArtifactGenerator.generate_both('photo.jpg')
            >>> with open('ela.png', 'wb') as f:
            ...     f.write(ela)
            >>> with open('fft.png', 'wb') as f:
            ...     f.write(fft)
        """
        ela_bytes = ArtifactGenerator.generate_ela(
            image_input,
            quality=ela_quality,
            scale_factor=ela_scale
        )
        fft_bytes = ArtifactGenerator.generate_fft(image_input)

        return ela_bytes, fft_bytes


# Convenience functions for direct usage
def generate_ela(
    image_input: Union[str, Image.Image, np.ndarray],
    quality: int = 90,
    scale_factor: int = 15
) -> bytes:
    """
    Generate Error Level Analysis (ELA) map.

    Convenience function that calls ArtifactGenerator.generate_ela().
    See ArtifactGenerator.generate_ela() for detailed documentation.
    """
    return ArtifactGenerator.generate_ela(image_input, quality, scale_factor)


def generate_fft(
    image_input: Union[str, Image.Image, np.ndarray]
) -> bytes:
    """
    Generate Fast Fourier Transform (FFT) magnitude spectrum.

    Convenience function that calls ArtifactGenerator.generate_fft().
    See ArtifactGenerator.generate_fft() for detailed documentation.
    """
    return ArtifactGenerator.generate_fft(image_input)


def generate_both(
    image_input: Union[str, Image.Image, np.ndarray],
    ela_quality: int = 90,
    ela_scale: int = 15
) -> Tuple[bytes, bytes]:
    """
    Generate both ELA and FFT artifacts.

    Convenience function that calls ArtifactGenerator.generate_both().
    Returns tuple of (ela_bytes, fft_bytes).
    """
    return ArtifactGenerator.generate_both(image_input, ela_quality, ela_scale)


if __name__ == "__main__":
    # Example usage and basic testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python forensics.py <image_path>")
        print("Generates ELA and FFT maps for the specified image")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Generating forensic artifacts for: {image_path}")

    try:
        # Generate both artifacts
        ela_bytes, fft_bytes = generate_both(image_path)

        # Save outputs
        ela_output = image_path.rsplit('.', 1)[0] + '_ela.png'
        fft_output = image_path.rsplit('.', 1)[0] + '_fft.png'

        with open(ela_output, 'wb') as f:
            f.write(ela_bytes)
        print(f"✓ ELA map saved to: {ela_output}")

        with open(fft_output, 'wb') as f:
            f.write(fft_bytes)
        print(f"✓ FFT spectrum saved to: {fft_output}")

        print("\nForensic Analysis:")
        print("  - Check ELA map: Uniform rainbow = AI, Dark with edges = Real")
        print("  - Check FFT spectrum: Grid/Cross = AI, Chaotic starburst = Real")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
