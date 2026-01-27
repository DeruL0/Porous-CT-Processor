# Porous CT Analysis Suite

A scientific computing application for analyzing porous materials (rocks, ceramics, foams, etc.) using Micro-CT data. Built with **Python**, **PyQt5**, and **PyVista**.

## Overview

This application provides a comprehensive workflow for Digital Rock Physics (DRP):

1. **Ingestion**: Load industrial DICOM series, 4D CT time-series, or generate synthetic test data.
2. **Visualization**: Interactive 3D rendering with orthogonal slices, isosurfaces, and time-step navigation.
3. **Quantification**: Extract porosity, segment void space, and track pore evolution over time.
4. **Modeling**: Generate Pore Network Models (PNM) using watershed segmentation.
5. **Export**: Save results to VTK standards (.vtp/.vti) for simulation software.

## Project Structure

```
Porous/
├── App.py              # Application Controller (MVC Entry Point)
├── config.py           # Configuration settings
├── core/               # Core Logic
│   ├── base.py         # Abstract Base Classes
│   ├── gpu_backend.py  # GPU Acceleration Backend
│   └── time_series.py  # 4DCT Time Series Logic
├── loaders/            # Data loading strategies
│   ├── dicom.py        # DICOM series loaders
│   └── dummy.py        # Synthetic data generator
├── processors/         # Analysis algorithms & Logic
│   ├── pore.py         # Void extraction
│   ├── pnm.py          # Pore Network Modeling
│   ├── pnm_tracker.py  # 4D Pore Tracking
│   └── gpu_pipeline.py # GPU processing pipelines
├── exporters/          # Data export handlers (VTK)
├── data/               # Data Layer
│   └── manager.py      # Central Data Manager
├── gui/                # User Interface
│   ├── main_window.py  # Main Visualizer Window
│   ├── panels/         # Reusable UI Panels (ROI, Processing, TimeSeries)
│   └── handlers/       # UI Logic Handlers (Workflow, TimeSeries)
├── rendering/          # 3D Rendering Engine
│   └── render_engine.py
└── web_intro/          # Web Introduction / Landing Page material
```

## Features

### Visualization Modes

* **📊 Volume Rendering**: Full 3D density rendering with adjustable opacity transfer functions.
* **🔳 Orthogonal Slices**: Interactive X, Y, Z planes with mouse probe.
* **🏔️ Isosurface**: Solid-void interface with multiple coloring modes.
* **⚪ PNM Mesh**: Network topology visualization (Pores & Throats).
* **⏱️ 4D Playback**: Navigate through time steps for temporal CT data.

### Structural Analysis

* **Void Extraction**: Segment air/void voxels using intensity thresholding.
* **Pore Network Modeling (PNM)**: Watershed segmentation (Ball-and-Stick model).
* **4D Tracking**: Track individual pores across time steps to analyze evolution.
  - **Optimized Algorithms**: Batch IoU calculation (3-5x faster), GPU acceleration (5-10x for 500+ pores)
  - **Automatic Selection**: Chooses best algorithm based on dataset size and hardware
  - **Hungarian Matching**: Optional global optimal assignment for complex scenarios
* **Smart Caching**: 
  - PNM results are cached after generation, avoiding regeneration when switching views.
  - Extracted pores are cached per timepoint for instant navigation in 4D series.
* **GPU Acceleration**: CuPy-based acceleration for tracking, watershed, and EDT operations.

### Data IO

* **Load DICOM**: Standard CT image series support.
* **Import 4D-CT**: Load multiple time steps from folder series.
* **Fast Load**: Downsampled preview for large datasets.
* **Synthetic Generator**: Gaussian Random Field volume for testing.

## Installation

### Requirements

* Python 3.8+
* See `requirements.txt` for dependencies

### Setup

```bash
pip install -r requirements.txt
python App.py
```

## Dependencies

| Package | Purpose |
|---------|---------|
| PyQt5 | GUI framework |
| pyvista, pyvistaqt | 3D rendering |
| numpy, scipy, scikit-image | Image processing |
| pydicom | DICOM data loading |
| cupy (optional) | GPU acceleration |
