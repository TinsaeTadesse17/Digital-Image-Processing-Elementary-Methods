# Digital Image Processing: Elementary Methods

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/NumPy-Data%20Science-blueid?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-Data%20Viz-orange?logo=plotly&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Status-Complete-success" alt="Status">
</p>

## 📖 Overview

This repository contains a comprehensive implemention of **7 fundamental intensity transformation and histogram processing methods** in Digital Image Processing. The project is designed with academic rigor, referencing foundational literature while utilizing modern Python data science workflows.

Each method supports both **RGB** and **Grayscale** images. Classic computer vision benchmarks (e.g., Cameraman, Astronaut, Moon) are processed alongside a custom, traditional Ethiopian **Jebena** ($\textbf{ጀበና}$) test image.

This is an individual assignment built for the Computer Vision course at **Addis Ababa University**.

---

## 🛠️ Methods Implemented

| # | Method | Transfer Function | Key Reference | Core Application |
|---|--------|-------------------|---------------|------------------|
| **1** | **Image Negative** | `s = (L-1) - r` | Gonzalez & Woods §3.2.1 | Medical imaging (mammograms, X-rays), astronomy |
| **2** | **Gamma Correction** | `s = c · r^γ` | Poynton (1998) | Exposure correction, CRT display compensation |
| **3** | **Logarithmic Transform** | `s = c · log(1 + r)` | Stockham (1972) | Dynamic range compression, Fourier magnitude spectra |
| **4** | **Contrast Stretching** | Piecewise linear mapping | Gonzalez & Woods §3.2.4 | Low-contrast/washed-out image recovery |
| **5** | **Histogram Equalization** | CDF-based mapping | Pizer et al. (1987) | Global contrast enhancement, CLAHE |
| **6** | **Intensity Level Slicing** | Range highlighting | Gonzalez & Woods §3.2.5 | Segmentation, defect inspection |
| **7** | **Bit Plane Slicing** | Binary decomposition | Gonzalez & Woods §3.2.6 | Image compression, steganography |

---

## 📂 Repository Structure

```text
├── image_processing.py         # Core algorithms for all 7 transformation methods
├── utils.py                    # Utility functions for I/O, display, and histograms
├── run_all.py                  # Main execution script to generate outputs
├── DIP_Elementary_Methods.ipynb# Complete self-contained Google Colab notebook
├── requirements.txt            # Python environment dependencies
├── README.md                   # Project documentation
├── images/                     # Input custom images (e.g., jebena.png)
└── outputs/                    # Processed output visualizations (28 items)
```
*(Note: PowerPoint generation scripts and generated reports have been excluded from this public release).*

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/TinsaeTadesse17/Digital-Image-Processing-Elementary-Methods.git
cd Digital-Image-Processing-Elementary-Methods

# Install dependencies
pip install -r requirements.txt
```

### Execution

To run all methods and generate the comparative visualization grids, execute the main orchestrator script:

```bash
python run_all.py
```
This will automatically process all test images through all 7 methods and save publication-quality `.png` figures to the `outputs/` directory.

---

## ☁️ Google Colab

For an interactive, zero-setup experience, use the provided Jupyter Notebook. 

1. Upload `DIP_Elementary_Methods.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Drag and drop any custom test images (like `jebena.png`) directly into the Colab file explorer.
3. Run all cells (`Ctrl + F9`) to view inline plotting with full mathematical context written in Markdown/LaTeX.

---

## 📚 Academic References

1. **Gonzalez, R.C. & Woods, R.E.** (2018). *Digital Image Processing*, 4th Ed. Pearson.
2. **Stockham, T.G. Jr.** (1972). *Image Processing in the Context of a Visual Model*. Proceedings of the IEEE, 60(7), 828-842.
3. **Pizer, S.M., et al.** (1987). *Adaptive Histogram Equalization and Its Variations*. Computer Vision, Graphics, and Image Processing, 39, 355-368.
4. **Poynton, C.** (1998). *The Rehabilitation of Gamma*. Proceedings of the SPIE/IS&T Conference.
5. **Weber-Fechner Law**: Foundational psychophysics principle relating stimulus intensity to perceived magnitude.

---

## 👨‍💻 Author

**Tinsae Tadesse**  
Addis Ababa University
