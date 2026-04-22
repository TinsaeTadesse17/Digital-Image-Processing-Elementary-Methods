"""
utils.py — Utility Functions for Digital Image Processing
==========================================================
Addis Ababa University | Computer Vision Individual Assignment
Author: Tinsae Tadesse
Submitted to: Dr. Fantahun (PhD)

Provides image loading, display, comparison, and histogram visualization helpers.
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage import data, color, img_as_ubyte, img_as_float
from PIL import Image

# ─── Global Style ────────────────────────────────────────────────────────────

plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eee',
    'text.color': '#eee',
    'xtick.color': '#aaa',
    'ytick.color': '#aaa',
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
})


# ─── Image Loading ───────────────────────────────────────────────────────────

SKIMAGE_IMAGES = {
    'camera': lambda: data.camera(),
    'cameraman': lambda: data.camera(),
    'astronaut': lambda: data.astronaut(),
    'moon': lambda: data.moon(),
    'coffee': lambda: data.coffee(),
    'coins': lambda: data.coins(),
    'horse': lambda: data.horse(),
    'chelsea': lambda: data.chelsea(),    # cat image
    'hubble': lambda: data.hubble_deep_field(),
}


def load_image(name_or_path, as_gray=False):
    """
    Load an image by scikit-image name or file path.
    
    Parameters
    ----------
    name_or_path : str
        Either a key from SKIMAGE_IMAGES (e.g., 'camera', 'astronaut')
        or an absolute/relative file path.
    as_gray : bool
        If True, force conversion to grayscale.
    
    Returns
    -------
    img : np.ndarray (uint8)
        Image array, either (H, W) grayscale or (H, W, 3) RGB.
    """
    key = name_or_path.lower().strip()
    
    if key in SKIMAGE_IMAGES:
        img = SKIMAGE_IMAGES[key]()
    elif os.path.exists(name_or_path):
        img = cv2.imread(name_or_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {name_or_path}")
        # Convert BGR → RGB if color
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unknown image: '{name_or_path}'. "
                         f"Available: {list(SKIMAGE_IMAGES.keys())} or provide a valid path.")
    
    # Ensure uint8
    img = ensure_uint8(img)
    
    if as_gray and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    return img


def is_grayscale(img):
    """Check if an image is grayscale."""
    return len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)


def to_grayscale(img):
    """Convert RGB to grayscale; return as-is if already grayscale."""
    if is_grayscale(img):
        return img.squeeze() if len(img.shape) == 3 else img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def ensure_uint8(img):
    """
    Safely convert image to uint8, handling float [0,1] and other dtypes.
    """
    if img.dtype == np.uint8:
        return img
    if img.dtype in [np.float32, np.float64]:
        if img.max() <= 1.0:
            return (img * 255).clip(0, 255).astype(np.uint8)
        return img.clip(0, 255).astype(np.uint8)
    if img.dtype == np.bool_:
        return (img.astype(np.uint8)) * 255
    return img.clip(0, 255).astype(np.uint8)


# ─── Display & Comparison ───────────────────────────────────────────────────

def display_comparison(original, processed, title, save_path,
                       original_label="Original", processed_label="Processed",
                       subtitle=None):
    """
    Side-by-side comparison of original and processed images.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold', color='#e94560', y=0.98)
    
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha='center', fontsize=9, 
                 color='#aaa', style='italic')
    
    _show_image(axes[0], original, original_label)
    _show_image(axes[1], processed, processed_label)
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Saved: {save_path}")


def display_multi_comparison(images, labels, title, save_path,
                              cols=4, subtitle=None):
    """
    Grid comparison of multiple images.
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(title, fontsize=16, fontweight='bold', color='#e94560', y=0.98)
    
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha='center', fontsize=9, 
                 color='#aaa', style='italic')
    
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)
    
    for idx in range(rows * cols):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        if idx < n:
            _show_image(ax, images[idx], labels[idx])
        else:
            ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Saved: {save_path}")


def display_with_histogram(original, processed, title, save_path,
                            original_label="Original", processed_label="Processed",
                            subtitle=None):
    """
    Side-by-side images with their histograms below.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(title, fontsize=16, fontweight='bold', color='#e94560', y=0.98)
    
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha='center', fontsize=9, 
                 color='#aaa', style='italic')
    
    # Images
    ax1 = fig.add_subplot(gs[0, 0])
    _show_image(ax1, original, original_label)
    
    ax2 = fig.add_subplot(gs[0, 1])
    _show_image(ax2, processed, processed_label)
    
    # Histograms
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_histogram(ax3, original, f"{original_label} Histogram")
    
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_histogram(ax4, processed, f"{processed_label} Histogram")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"  [OK] Saved: {save_path}")


# ─── Internal Helpers ────────────────────────────────────────────────────────

def _show_image(ax, img, title):
    """Display an image on a matplotlib axis."""
    if is_grayscale(img):
        ax.imshow(img.squeeze(), cmap='gray', vmin=0, vmax=255)
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis('off')


def _plot_histogram(ax, img, title):
    """Plot histogram of an image."""
    ax.set_title(title, fontsize=10, pad=6)
    
    if is_grayscale(img):
        hist = cv2.calcHist([img.squeeze()], [0], None, [256], [0, 256])
        ax.fill_between(range(256), hist.flatten(), alpha=0.7, color='#e94560')
        ax.plot(hist, color='#e94560', linewidth=0.8)
    else:
        colors_hex = ['#ff4444', '#44ff44', '#4488ff']
        colors_name = ['Red', 'Green', 'Blue']
        for i, (ch, cn) in enumerate(zip(colors_hex, colors_name)):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(hist, color=ch, linewidth=1.0, alpha=0.8, label=cn)
        ax.legend(fontsize=8, loc='upper right',
                  facecolor='#16213e', edgecolor='#444', labelcolor='#eee')
    
    ax.set_xlim([0, 255])
    ax.set_xlabel('Intensity', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.grid(True, alpha=0.15)


def get_image_info(img):
    """Return a string with image metadata."""
    shape_str = f"{img.shape[1]}x{img.shape[0]}"
    mode = "Grayscale" if is_grayscale(img) else f"RGB ({img.shape[2]} channels)"
    return f"{shape_str} | {mode} | {img.dtype}"
