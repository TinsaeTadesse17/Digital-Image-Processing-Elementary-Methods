"""
image_processing.py — Core Digital Image Processing Algorithms
===============================================================
Addis Ababa University | Computer Vision Individual Assignment
Author: Tinsae Tadesse
Submitted to: Dr. Fantahun (PhD)

Implements 7 elementary intensity transformation & histogram processing methods
with academic rigor. Each function includes docstrings referencing foundational
literature (Gonzalez & Woods, Stockham, Pizer et al.).

All functions handle both grayscale (H,W) and RGB (H,W,3) images transparently.
"""

import numpy as np
import cv2


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. IMAGE NEGATIVE                                                       ║
# ║  Reference: Gonzalez & Woods, DIP 4th Ed., §3.2.1                       ║
# ║  s = (L - 1) - r, where L = 256 for 8-bit images                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def image_negative(img):
    """
    Compute the photographic negative of an image.
    
    The negative transformation reverses the intensity levels:
        s = (L - 1) - r
    
    This is analogous to obtaining the photographic negative of a film.
    It is particularly useful for enhancing white or gray detail embedded
    in dark regions of an image (e.g., mammograms, astronomical images).
    
    Parameters
    ----------
    img : np.ndarray (uint8)
        Input image, shape (H,W) for grayscale or (H,W,3) for RGB.
    
    Returns
    -------
    np.ndarray (uint8)
        Negative image.
    
    References
    ----------
    Gonzalez, R.C. & Woods, R.E. (2018). Digital Image Processing, §3.2.1.
    """
    return 255 - img


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. GAMMA ENCODING / CORRECTION                                         ║
# ║  Reference: Gonzalez & Woods §3.2.3; Poynton (1998)                     ║
# ║  s = c · r^γ  (power-law transformation)                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def gamma_correction(img, gamma=1.0, c=1.0):
    """
    Apply power-law (gamma) transformation.
    
    The power-law transform is defined as:
        s = c · r^γ
    
    where r is normalized to [0, 1], and the result is rescaled to [0, 255].
    
    Key gamma values:
        γ < 1  → Brightens dark regions (expands low intensities)
        γ = 1  → Identity transformation
        γ > 1  → Darkens the image (compresses low intensities)
    
    Historical context:
        CRT monitors have an inherent gamma of ~2.2. The sRGB standard
        (IEC 61966-2-1) specifies gamma encoding to ensure perceptual
        uniformity across display devices. Poynton (1998) provides an
        excellent deep-dive into gamma's role in imaging pipelines.
    
    Parameters
    ----------
    img : np.ndarray (uint8)
        Input image.
    gamma : float
        Gamma value. Default 1.0 (identity).
    c : float
        Scaling constant. Default 1.0.
    
    Returns
    -------
    np.ndarray (uint8)
        Gamma-corrected image.
    
    References
    ----------
    Gonzalez & Woods (2018), §3.2.3.
    Poynton, C. (1998). "The Rehabilitation of Gamma."
    IEC 61966-2-1:1999. Multimedia systems — sRGB colour space.
    """
    # Normalize to [0, 1]
    normalized = img.astype(np.float64) / 255.0
    # Apply power-law
    corrected = c * np.power(normalized, gamma)
    # Rescale and clip
    return (corrected * 255).clip(0, 255).astype(np.uint8)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. LOGARITHMIC TRANSFORMATION                                          ║
# ║  Reference: Gonzalez & Woods §3.2.2; Stockham (1972)                    ║
# ║  s = c · log(1 + r)                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def log_transform(img, c=None):
    """
    Apply logarithmic intensity transformation.
    
    The log transform is defined as:
        s = c · log₂(1 + r)
    
    If c is not provided, it is computed as:
        c = 255 / log₂(1 + max(r))
    
    This transformation compresses the dynamic range of images with
    large variations in pixel values. It maps a narrow range of low
    intensities to a wider range and vice versa.
    
    Historical significance:
        Stockham (1972) demonstrated that logarithmic processing models
        human visual perception (Weber-Fechner Law: perceived brightness
        is proportional to the logarithm of actual intensity). The log
        transform is also essential for displaying Fourier spectra, where
        raw magnitudes span several orders of magnitude.
    
    Parameters
    ----------
    img : np.ndarray (uint8)
        Input image.
    c : float or None
        Scaling constant. If None, auto-computed for full dynamic range.
    
    Returns
    -------
    np.ndarray (uint8)
        Log-transformed image.
    
    References
    ----------
    Stockham, T.G. Jr. (1972). "Image Processing in the Context of a 
        Visual Model." Proceedings of the IEEE, 60(7), 828-842.
    Gonzalez & Woods (2018), §3.2.2.
    """
    img_float = img.astype(np.float64)
    
    if c is None:
        max_val = np.max(img_float)
        c = 255.0 / np.log2(1 + max_val) if max_val > 0 else 1.0
    
    result = c * np.log2(1 + img_float)
    return result.clip(0, 255).astype(np.uint8)


def inverse_log_transform(img, c=None):
    """
    Apply inverse logarithmic transformation.
    
    The inverse log expands the higher intensity values while compressing
    the lower ones — the opposite effect of the standard log transform.
    
        s = c · (2^r - 1)
    
    Parameters
    ----------
    img : np.ndarray (uint8)
    c : float or None
    
    Returns
    -------
    np.ndarray (uint8)
    """
    img_float = img.astype(np.float64) / 255.0  # Normalize first
    result = np.power(2, img_float * 8) - 1  # Map to exponential scale
    # Normalize to [0, 255]
    if result.max() > 0:
        result = (result / result.max()) * 255
    return result.clip(0, 255).astype(np.uint8)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. CONTRAST STRETCHING                                                  ║
# ║  Reference: Gonzalez & Woods §3.2.4                                      ║
# ║  Piecewise linear transformation                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def contrast_stretch_minmax(img):
    """
    Min-Max contrast stretching.
    
    Maps the intensity range [min(r), max(r)] → [0, 255] linearly:
        s = 255 · (r - r_min) / (r_max - r_min)
    
    This is the simplest form of contrast stretching and is sensitive
    to outlier pixels (single very dark or bright pixel dominates).
    
    Parameters
    ----------
    img : np.ndarray (uint8)
    
    Returns
    -------
    np.ndarray (uint8)
    """
    def _stretch_channel(ch):
        r_min, r_max = ch.min(), ch.max()
        if r_max == r_min:
            return ch
        return ((ch.astype(np.float64) - r_min) / (r_max - r_min) * 255).clip(0, 255).astype(np.uint8)
    
    if len(img.shape) == 2:
        return _stretch_channel(img)
    else:
        return np.stack([_stretch_channel(img[:, :, i]) for i in range(img.shape[2])], axis=2)


def contrast_stretch_percentile(img, low_pct=2, high_pct=98):
    """
    Percentile-based contrast stretching (robust to outliers).
    
    Instead of using absolute min/max, uses percentile values to determine
    the stretching range. Pixels below the low percentile are clipped to 0,
    and pixels above the high percentile are clipped to 255.
    
    This approach is more robust than min-max stretching because it is
    not influenced by a small number of extreme outlier pixels.
    
    Parameters
    ----------
    img : np.ndarray (uint8)
    low_pct : float
        Lower percentile (default: 2).
    high_pct : float
        Upper percentile (default: 98).
    
    Returns
    -------
    np.ndarray (uint8)
    
    References
    ----------
    Gonzalez & Woods (2018), §3.2.4.
    """
    def _stretch_channel(ch):
        p_low = np.percentile(ch, low_pct)
        p_high = np.percentile(ch, high_pct)
        if p_high == p_low:
            return ch
        stretched = (ch.astype(np.float64) - p_low) / (p_high - p_low) * 255
        return stretched.clip(0, 255).astype(np.uint8)
    
    if len(img.shape) == 2:
        return _stretch_channel(img)
    else:
        return np.stack([_stretch_channel(img[:, :, i]) for i in range(img.shape[2])], axis=2)


def contrast_stretch_piecewise(img, r1=70, s1=0, r2=180, s2=255):
    """
    Piecewise linear contrast stretching.
    
    Defines a piecewise linear transformation with breakpoints (r1,s1) and (r2,s2):
        - [0, r1]     → [0, s1]       (compress dark tones)
        - [r1, r2]    → [s1, s2]      (stretch midtones)
        - [r2, 255]   → [s2, 255]     (compress highlights)
    
    Parameters
    ----------
    img : np.ndarray (uint8)
    r1, s1 : int
        First breakpoint.
    r2, s2 : int
        Second breakpoint.
    
    Returns
    -------
    np.ndarray (uint8)
    """
    lut = np.zeros(256, dtype=np.uint8)
    
    for r in range(256):
        if r < r1:
            lut[r] = int(s1 / max(r1, 1) * r) if r1 > 0 else 0
        elif r <= r2:
            lut[r] = int(s1 + (s2 - s1) / max(r2 - r1, 1) * (r - r1))
        else:
            lut[r] = int(s2 + (255 - s2) / max(255 - r2, 1) * (r - r2))
    
    if len(img.shape) == 2:
        return cv2.LUT(img, lut)
    else:
        return np.stack([cv2.LUT(img[:, :, i], lut) for i in range(img.shape[2])], axis=2)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5. HISTOGRAM EQUALIZATION                                               ║
# ║  Reference: Gonzalez & Woods §3.3.1; Pizer et al. (1987)                ║
# ║  CDF-based mapping for uniform intensity distribution                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def histogram_equalization_manual(img):
    """
    Manual histogram equalization using CDF computation.
    
    The theoretical basis for HE comes from probability theory: if we
    apply the CDF of the input histogram as a transformation function,
    the output histogram will be approximately uniform.
    
    Algorithm:
        1. Compute histogram h(rk) for each intensity level rk
        2. Compute normalized CDF: sk = Σ(j=0 to k) p(rj)
           where p(rj) = h(rj) / (M×N)
        3. Map: output = round((L-1) × sk)
    
    For RGB images, we convert to HSV and equalize only the V (value)
    channel to preserve color relationships (hue and saturation).
    
    Parameters
    ----------
    img : np.ndarray (uint8)
    
    Returns
    -------
    np.ndarray (uint8)
    
    References
    ----------
    Gonzalez & Woods (2018), §3.3.1.
    """
    def _equalize_channel(channel):
        # Step 1: Compute histogram
        hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
        
        # Step 2: Compute CDF
        cdf = hist.cumsum()
        
        # Mask zeros in CDF
        cdf_masked = np.ma.masked_equal(cdf, 0)
        
        # Normalize CDF to [0, 255]
        cdf_normalized = ((cdf_masked - cdf_masked.min()) * 255 /
                          (cdf_masked.max() - cdf_masked.min()))
        cdf_final = np.ma.filled(cdf_normalized, 0).astype(np.uint8)
        
        # Step 3: Map
        return cdf_final[channel]
    
    if len(img.shape) == 2:
        return _equalize_channel(img)
    else:
        # For RGB: convert to HSV, equalize V channel only
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = _equalize_channel(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def histogram_equalization_opencv(img):
    """
    Histogram equalization using OpenCV (for comparison).
    
    Parameters
    ----------
    img : np.ndarray (uint8)
    
    Returns
    -------
    np.ndarray (uint8)
    """
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Unlike standard HE which uses a global transformation, CLAHE divides
    the image into tiles and equalizes each tile independently. A contrast
    limit is applied to prevent over-amplification of noise.
    
    Pizer et al. (1987) introduced AHE and its contrast-limited variant
    to address the limitations of global HE in medical imaging, where
    local contrast enhancement is crucial.
    
    Parameters
    ----------
    img : np.ndarray (uint8)
    clip_limit : float
        Threshold for contrast limiting. Higher values give more contrast.
    tile_grid_size : tuple
        Size of the grid for local histogram equalization.
    
    Returns
    -------
    np.ndarray (uint8)
    
    References
    ----------
    Pizer, S.M. et al. (1987). "Adaptive Histogram Equalization and Its
        Variations." Computer Vision, Graphics, and Image Processing, 39, 355-368.
    """
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if len(img.shape) == 2:
        return clahe_obj.apply(img)
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = clahe_obj.apply(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6. INTENSITY LEVEL SLICING                                              ║
# ║  Reference: Gonzalez & Woods §3.2.5                                      ║
# ║  Highlight a specific range of intensities                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def intensity_level_slicing_binary(img, low=100, high=200, highlight_val=255):
    """
    Binary intensity level slicing (without background preservation).
    
    All pixels within [low, high] are set to highlight_val (white),
    and all other pixels are set to 0 (black).
    
    This is the simpler variant and produces a binary output that clearly
    shows which regions of the image contain the target intensity range.
    Used extensively in medical imaging for tissue segmentation.
    
    Parameters
    ----------
    img : np.ndarray (uint8)
    low, high : int
        Intensity range to highlight [low, high].
    highlight_val : int
        Value to assign to highlighted pixels.
    
    Returns
    -------
    np.ndarray (uint8)
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = np.zeros_like(gray)
    mask = (gray >= low) & (gray <= high)
    result[mask] = highlight_val
    return result


def intensity_level_slicing_preserve(img, low=100, high=200, highlight_val=255):
    """
    Intensity level slicing with background preservation.
    
    Pixels within [low, high] are set to highlight_val, while pixels
    outside that range retain their original intensity values. This
    preserves the context of the image while highlighting the region
    of interest.
    
    Parameters
    ----------
    img : np.ndarray (uint8)
    low, high : int
        Intensity range to highlight.
    highlight_val : int
        Value for highlighted pixels.
    
    Returns
    -------
    np.ndarray (uint8)
    
    References
    ----------
    Gonzalez & Woods (2018), §3.2.5.
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = gray.copy()
    mask = (gray >= low) & (gray <= high)
    result[mask] = highlight_val
    return result


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  7. BIT PLANE SLICING                                                    ║
# ║  Reference: Gonzalez & Woods §3.2.6                                      ║
# ║  Decompose image into individual bit planes                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def bit_plane_slice(img, bit):
    """
    Extract a single bit plane from an image.
    
    For an 8-bit image, each pixel can be represented in binary as:
        pixel = a7·2⁷ + a6·2⁶ + ... + a1·2¹ + a0·2⁰
    
    Bit plane n contains the binary image formed by the n-th bit of
    each pixel. The MSB (bit 7) carries the most significant visual
    information, while the LSB (bit 0) appears as random noise for
    most natural images.
    
    Applications:
        - Image compression: discard lower bit planes
        - Watermarking: embed data in LSB planes
        - Image analysis: understand contribution of each bit
    
    Parameters
    ----------
    img : np.ndarray (uint8)
        Input image (grayscale recommended).
    bit : int
        Bit plane to extract (0 = LSB, 7 = MSB).
    
    Returns
    -------
    np.ndarray (uint8)
        Binary image (0 or 255) representing the bit plane.
    
    References
    ----------
    Gonzalez & Woods (2018), §3.2.6.
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plane = (gray >> bit) & 1
    return (plane * 255).astype(np.uint8)


def bit_plane_decompose(img):
    """
    Decompose an image into all 8 bit planes.
    
    Returns
    -------
    list of np.ndarray
        8 binary images, from bit 0 (LSB) to bit 7 (MSB).
    """
    return [bit_plane_slice(img, bit) for bit in range(8)]


def bit_plane_reconstruct(img, num_planes=4):
    """
    Reconstruct an image using only the top-k most significant bit planes.
    
    This demonstrates that the upper bit planes carry the majority of
    visually meaningful information. Reconstructing from only the top 4
    planes (out of 8) typically produces a recognizable image, demonstrating
    that 50% of the data can be discarded with acceptable quality loss.
    
    Parameters
    ----------
    img : np.ndarray (uint8)
    num_planes : int
        Number of MSB planes to use (1-8). Default 4.
    
    Returns
    -------
    np.ndarray (uint8)
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = np.zeros_like(gray, dtype=np.uint8)
    
    for bit in range(8 - num_planes, 8):
        plane = (gray >> bit) & 1
        result += (plane << bit).astype(np.uint8)
    
    return result
