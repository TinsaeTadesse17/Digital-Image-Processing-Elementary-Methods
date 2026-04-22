"""
run_all.py — Orchestrator: Generate All Visualizations
=======================================================
Addis Ababa University | Computer Vision Individual Assignment
Author: Tinsae Tadesse
Submitted to: Dr. Fantahun (PhD)

Runs all 7 image processing methods on multiple test images and saves
publication-quality visualizations to the outputs/ directory.
"""

import os
import sys
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_image, to_grayscale, display_comparison, display_multi_comparison,
    display_with_histogram, get_image_info
)
from image_processing import (
    image_negative,
    gamma_correction,
    log_transform, inverse_log_transform,
    contrast_stretch_minmax, contrast_stretch_percentile, contrast_stretch_piecewise,
    histogram_equalization_manual, histogram_equalization_opencv, clahe,
    intensity_level_slicing_binary, intensity_level_slicing_preserve,
    bit_plane_slice, bit_plane_decompose, bit_plane_reconstruct,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def section_header(num, title):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {num}. {title}")
    print(f"{'=' * 70}")


# -----------------------------------------------------------------------
# 1. IMAGE NEGATIVE
# ─────────────────────────────────────────────────────────────────────────────

def run_negative():
    section_header(1, "IMAGE NEGATIVE  |  s = (L-1) - r")
    out = os.path.join(OUTPUT_DIR, "1_negative")
    
    # Grayscale: Moon
    img = load_image('moon')
    print(f"  Moon: {get_image_info(img)}")
    neg = image_negative(img)
    display_with_histogram(img, neg,
        "Image Negative — Moon (Grayscale)",
        os.path.join(out, "negative_moon.png"),
        subtitle="s = 255 - r  |  Enhances bright details in dark regions")
    
    # RGB: Astronaut
    img = load_image('astronaut')
    print(f"  Astronaut: {get_image_info(img)}")
    neg = image_negative(img)
    display_comparison(img, neg,
        "Image Negative — Astronaut (RGB)",
        os.path.join(out, "negative_astronaut.png"),
        subtitle="Per-channel inversion: s_c = 255 - r_c for c in {R, G, B}")
    
    # RGB: Jebena (Ethiopian coffee pot)
    jebena_path = os.path.join(BASE_DIR, "images", "jebena.png")
    if os.path.exists(jebena_path):
        img = load_image(jebena_path)
        print(f"  Jebena: {get_image_info(img)}")
        neg = image_negative(img)
        display_comparison(img, neg,
            "Image Negative — Jebena ☕ (Ethiopian Coffee Pot)",
            os.path.join(out, "negative_jebena.png"),
            subtitle="Traditional Ethiopian Jebena — per-channel inversion")


# ─────────────────────────────────────────────────────────────────────────────
# 2. GAMMA ENCODING / CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def run_gamma():
    section_header(2, "GAMMA CORRECTION  |  s = c * r^gamma")
    out = os.path.join(OUTPUT_DIR, "2_gamma")
    
    gammas = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    # Grayscale: Camera
    img = load_image('camera')
    print(f"  Camera: {get_image_info(img)}")
    images = [img] + [gamma_correction(img, g) for g in gammas]
    labels = ["Original"] + [f"gamma={g}" for g in gammas]
    display_multi_comparison(images, labels,
        "Gamma Correction — Cameraman (Grayscale)",
        os.path.join(out, "gamma_camera.png"),
        cols=4,
        subtitle="Power-law transform: gamma<1 brightens, gamma>1 darkens  |  Poynton (1998)")
    
    # RGB: Coffee
    img = load_image('coffee')
    print(f"  Coffee: {get_image_info(img)}")
    images = [img] + [gamma_correction(img, g) for g in gammas]
    labels = ["Original"] + [f"γ = {g}" for g in gammas]
    display_multi_comparison(images, labels,
        "Gamma Correction — Coffee (RGB)",
        os.path.join(out, "gamma_coffee.png"),
        cols=4,
        subtitle="sRGB standard uses gamma~2.2 for CRT compensation  |  IEC 61966-2-1")

    # RGB: Jebena
    jebena_path = os.path.join(BASE_DIR, "images", "jebena.png")
    if os.path.exists(jebena_path):
        img = load_image(jebena_path)
        print(f"  Jebena: {get_image_info(img)}")
        key_gammas = [0.3, 0.5, 1.0, 2.0, 3.0]
        images = [img] + [gamma_correction(img, g) for g in key_gammas]
        labels = ["Original"] + [f"gamma={g}" for g in key_gammas]
        display_multi_comparison(images, labels,
            "Gamma Correction — Jebena ☕",
            os.path.join(out, "gamma_jebena.png"),
            cols=3,
            subtitle="Ethiopian Jebena under various gamma values")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOGARITHMIC TRANSFORMATION
# ─────────────────────────────────────────────────────────────────────────────

def run_logarithmic():
    section_header(3, "LOGARITHMIC TRANSFORM  |  s = c * log(1 + r)")
    out = os.path.join(OUTPUT_DIR, "3_logarithmic")
    
    # Grayscale: Moon (low contrast — perfect for log)
    img = load_image('moon')
    print(f"  Moon: {get_image_info(img)}")
    log_img = log_transform(img)
    inv_log_img = inverse_log_transform(img)
    
    display_with_histogram(img, log_img,
        "Log Transform — Moon (Grayscale)",
        os.path.join(out, "log_moon.png"),
        processed_label="Log Transformed",
        subtitle="s = c * log2(1 + r)  |  Stockham (1972) -- Weber-Fechner Law")
    
    display_multi_comparison(
        [img, log_img, inv_log_img],
        ["Original", "Log Transform", "Inverse Log"],
        "Log vs. Inverse Log — Moon",
        os.path.join(out, "log_vs_invlog_moon.png"),
        cols=3,
        subtitle="Log compresses highlights; Inverse log expands them")
    
    # RGB: Astronaut
    img = load_image('astronaut')
    print(f"  Astronaut: {get_image_info(img)}")
    log_img = log_transform(img)
    display_with_histogram(img, log_img,
        "Log Transform — Astronaut (RGB)",
        os.path.join(out, "log_astronaut.png"),
        processed_label="Log Transformed",
        subtitle="Dynamic range compression for Fourier spectrum visualization")
    
    # Jebena
    jebena_path = os.path.join(BASE_DIR, "images", "jebena.png")
    if os.path.exists(jebena_path):
        img = load_image(jebena_path)
        print(f"  Jebena: {get_image_info(img)}")
        log_img = log_transform(img)
        display_with_histogram(img, log_img,
            "Log Transform — Jebena ☕",
            os.path.join(out, "log_jebena.png"),
            processed_label="Log Transformed",
            subtitle="Compresses wide dynamic range of natural scene")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONTRAST STRETCHING
# ─────────────────────────────────────────────────────────────────────────────

def run_contrast_stretching():
    section_header(4, "CONTRAST STRETCHING  |  Piecewise Linear Mapping")
    out = os.path.join(OUTPUT_DIR, "4_contrast_stretching")
    
    # Grayscale: Moon (low contrast)
    img = load_image('moon')
    print(f"  Moon: {get_image_info(img)}")
    
    stretched_mm = contrast_stretch_minmax(img)
    stretched_pct = contrast_stretch_percentile(img, 2, 98)
    stretched_pw = contrast_stretch_piecewise(img, r1=80, s1=10, r2=180, s2=245)
    
    display_multi_comparison(
        [img, stretched_mm, stretched_pct, stretched_pw],
        ["Original", "Min-Max", "Percentile (2%-98%)", "Piecewise Linear"],
        "Contrast Stretching Comparison — Moon",
        os.path.join(out, "contrast_moon.png"),
        cols=4,
        subtitle="Three variants: Min-Max | Percentile (robust) | Piecewise Linear (custom)")
    
    display_with_histogram(img, stretched_pct,
        "Contrast Stretching — Moon (Percentile)",
        os.path.join(out, "contrast_moon_hist.png"),
        processed_label="Percentile Stretched",
        subtitle="2nd-98th percentile stretching — robust to outliers")
    
    # RGB: Astronaut
    img = load_image('astronaut')
    print(f"  Astronaut: {get_image_info(img)}")
    stretched = contrast_stretch_percentile(img)
    display_with_histogram(img, stretched,
        "Contrast Stretching — Astronaut (RGB)",
        os.path.join(out, "contrast_astronaut.png"),
        processed_label="Percentile Stretched",
        subtitle="Per-channel percentile stretching preserves color balance")

    # Jebena
    jebena_path = os.path.join(BASE_DIR, "images", "jebena.png")
    if os.path.exists(jebena_path):
        img = load_image(jebena_path)
        print(f"  Jebena: {get_image_info(img)}")
        stretched = contrast_stretch_percentile(img)
        display_with_histogram(img, stretched,
            "Contrast Stretching — Jebena ☕",
            os.path.join(out, "contrast_jebena.png"),
            processed_label="Percentile Stretched",
            subtitle="Enhancing contrast of natural Ethiopian scene")


# ─────────────────────────────────────────────────────────────────────────────
# 5. HISTOGRAM EQUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def run_histogram_equalization():
    section_header(5, "HISTOGRAM EQUALIZATION  |  CDF-based Mapping")
    out = os.path.join(OUTPUT_DIR, "5_histogram_equalization")
    
    # Grayscale: Moon
    img = load_image('moon')
    print(f"  Moon: {get_image_info(img)}")
    
    he_manual = histogram_equalization_manual(img)
    he_opencv = histogram_equalization_opencv(img)
    he_clahe = clahe(img)
    
    display_multi_comparison(
        [img, he_manual, he_opencv, he_clahe],
        ["Original", "Manual HE", "OpenCV HE", "CLAHE"],
        "Histogram Equalization — Moon (Grayscale)",
        os.path.join(out, "histeq_moon.png"),
        cols=4,
        subtitle="Global HE vs. CLAHE (Pizer et al., 1987)  |  CDF -> Uniform distribution")
    
    display_with_histogram(img, he_manual,
        "Histogram Equalization — Moon (Manual CDF)",
        os.path.join(out, "histeq_moon_hist.png"),
        processed_label="Manual HE",
        subtitle="sk = Sum p(rj) mapped to [0, 255]  |  CDF-based equalization")
    
    display_with_histogram(img, he_clahe,
        "CLAHE — Moon",
        os.path.join(out, "clahe_moon_hist.png"),
        processed_label="CLAHE",
        subtitle="Contrast Limited AHE -- clip_limit=2.0, tile=8x8  |  Pizer et al. (1987)")
    
    # RGB: Astronaut
    img = load_image('astronaut')
    print(f"  Astronaut: {get_image_info(img)}")
    he_manual = histogram_equalization_manual(img)
    he_clahe = clahe(img)
    
    display_multi_comparison(
        [img, he_manual, he_clahe],
        ["Original", "Global HE (HSV-V)", "CLAHE (HSV-V)"],
        "Histogram Equalization — Astronaut (RGB)",
        os.path.join(out, "histeq_astronaut.png"),
        cols=3,
        subtitle="For RGB: equalize V channel in HSV space to preserve hue & saturation")
    
    # Jebena
    jebena_path = os.path.join(BASE_DIR, "images", "jebena.png")
    if os.path.exists(jebena_path):
        img = load_image(jebena_path)
        print(f"  Jebena: {get_image_info(img)}")
        he = histogram_equalization_manual(img)
        he_c = clahe(img)
        display_multi_comparison(
            [img, he, he_c],
            ["Original", "Global HE", "CLAHE"],
            "Histogram Equalization — Jebena ☕",
            os.path.join(out, "histeq_jebena.png"),
            cols=3,
            subtitle="Enhancing tonal range of traditional Ethiopian Jebena")


# ─────────────────────────────────────────────────────────────────────────────
# 6. INTENSITY LEVEL SLICING
# ─────────────────────────────────────────────────────────────────────────────

def run_intensity_slicing():
    section_header(6, "INTENSITY LEVEL SLICING  |  Range Highlighting")
    out = os.path.join(OUTPUT_DIR, "6_intensity_slicing")
    
    # Grayscale: Camera
    img = load_image('camera')
    print(f"  Camera: {get_image_info(img)}")
    
    # Multiple ranges
    ranges = [(50, 120), (100, 200), (150, 220)]
    
    for low, high in ranges:
        binary = intensity_level_slicing_binary(img, low, high)
        preserve = intensity_level_slicing_preserve(img, low, high)
        
        display_multi_comparison(
            [img, binary, preserve],
            ["Original", f"Binary [{low}-{high}]", f"Preserved [{low}-{high}]"],
            f"Intensity Level Slicing — Camera [{low}-{high}]",
            os.path.join(out, f"slice_camera_{low}_{high}.png"),
            cols=3,
            subtitle=f"Range [{low}, {high}]: Binary (no bg) vs. Preserved (with bg)")
    
    # RGB → Gray: Astronaut
    img = load_image('astronaut')
    print(f"  Astronaut: {get_image_info(img)}")
    binary = intensity_level_slicing_binary(img, 100, 200)
    preserve = intensity_level_slicing_preserve(img, 100, 200)
    
    display_multi_comparison(
        [img, binary, preserve],
        ["Original (RGB)", "Binary [100-200]", "Preserved [100-200]"],
        "Intensity Level Slicing — Astronaut",
        os.path.join(out, "slice_astronaut.png"),
        cols=3,
        subtitle="RGB converted to grayscale for slicing  |  Medical imaging application")
    
    # Jebena
    jebena_path = os.path.join(BASE_DIR, "images", "jebena.png")
    if os.path.exists(jebena_path):
        img = load_image(jebena_path)
        print(f"  Jebena: {get_image_info(img)}")
        binary = intensity_level_slicing_binary(img, 80, 180)
        preserve = intensity_level_slicing_preserve(img, 80, 180)
        display_multi_comparison(
            [img, binary, preserve],
            ["Original", "Binary [80-180]", "Preserved [80-180]"],
            "Intensity Level Slicing — Jebena ☕",
            os.path.join(out, "slice_jebena.png"),
            cols=3,
            subtitle="Isolating mid-tone intensities of the clay pot")


# ─────────────────────────────────────────────────────────────────────────────
# 7. BIT PLANE SLICING
# ─────────────────────────────────────────────────────────────────────────────

def run_bit_plane_slicing():
    section_header(7, "BIT PLANE SLICING  |  Binary Decomposition")
    out = os.path.join(OUTPUT_DIR, "7_bit_plane_slicing")
    
    # Grayscale: Camera
    img = load_image('camera')
    print(f"  Camera: {get_image_info(img)}")
    
    planes = bit_plane_decompose(img)
    images = [img] + planes[::-1]  # MSB first for visual clarity
    labels = ["Original"] + [f"Bit {7-i} ({'MSB' if i==0 else 'LSB' if i==7 else ''})"
              for i in range(8)]
    
    display_multi_comparison(images, labels,
        "Bit Plane Decomposition — Cameraman",
        os.path.join(out, "bitplane_camera.png"),
        cols=3,
        subtitle="pixel = a7*2^7 + a6*2^6 + ... + a1*2^1 + a0*2^0  |  MSB carries most info")
    
    # Reconstruction from top-k planes
    reconstructions = [bit_plane_reconstruct(img, k) for k in [1, 2, 3, 4, 5, 6, 7, 8]]
    images = [img] + reconstructions
    labels = ["Original"] + [f"Top {k} planes" for k in [1, 2, 3, 4, 5, 6, 7, 8]]
    
    display_multi_comparison(images, labels,
        "Bit Plane Reconstruction — Cameraman",
        os.path.join(out, "bitplane_reconstruct_camera.png"),
        cols=3,
        subtitle="Progressive reconstruction: top-4 planes ~ recognizable  |  Compression insight")
    
    # RGB: Astronaut (show per-channel decomposition of MSB)
    img = load_image('astronaut')
    print(f"  Astronaut: {get_image_info(img)}")
    
    # Extract MSB plane for each channel
    import cv2
    r_msb = bit_plane_slice(img[:, :, 0], 7)
    g_msb = bit_plane_slice(img[:, :, 1], 7)
    b_msb = bit_plane_slice(img[:, :, 2], 7)
    gray_msb = bit_plane_slice(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 7)
    
    display_multi_comparison(
        [img, r_msb, g_msb, b_msb, gray_msb],
        ["Original RGB", "R Channel MSB", "G Channel MSB", "B Channel MSB", "Gray MSB"],
        "Bit Plane Slicing (MSB) — Astronaut Per-Channel",
        os.path.join(out, "bitplane_astronaut_channels.png"),
        cols=3,
        subtitle="MSB (Bit 7) per RGB channel vs. grayscale  |  Channel-wise decomposition")

    # Jebena
    jebena_path = os.path.join(BASE_DIR, "images", "jebena.png")
    if os.path.exists(jebena_path):
        img = load_image(jebena_path)
        print(f"  Jebena: {get_image_info(img)}")
        planes = bit_plane_decompose(img)
        images = [img] + planes[::-1]
        labels = ["Original"] + [f"Bit {7-i}" for i in range(8)]
        display_multi_comparison(images, labels,
            "Bit Plane Decomposition — Jebena ☕",
            os.path.join(out, "bitplane_jebena.png"),
            cols=3,
            subtitle="8-bit decomposition of Ethiopian coffee pot image")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("+" + "=" * 70 + "+")
    print("|  Digital Image Processing -- Elementary Methods                      |")
    print("|  Addis Ababa University | Computer Vision                           |")
    print("|  Author: Tinsae Tadesse  |  Submitted to: Dr. Fantahun (PhD)       |")
    print("+" + "=" * 70 + "+")
    
    run_negative()
    run_gamma()
    run_logarithmic()
    run_contrast_stretching()
    run_histogram_equalization()
    run_intensity_slicing()
    run_bit_plane_slicing()
    
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE!  Outputs saved to: {OUTPUT_DIR}")
    print(f"  Run 'python generate_ppt.py' to create the PowerPoint report.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
