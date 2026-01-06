# Porous CT Analysis Suite

A scientific computing application for analyzing porous materials (rocks, ceramics, foams, etc.) using Micro-CT data. Built with **Python**, **PyQt5**, and **PyVista**.

## Overview

This application provides a comprehensive workflow for Digital Rock Physics (DRP):
1.  **Ingestion**: Load industrial DICOM series or generate synthetic test data.
2.  **Visualization**: Interactive 3D rendering with orthogonal slices and isosurfaces.
3.  **Quantification**: Extract porosity and segment void space.
4.  **Modeling**: Generate Pore Network Models (PNM) using watershed segmentation.
5.  **Export**: Save results to VTK standards (.vtp/.vti) for simulation software.

## Project Structure

```
Porous/
├── app.py              # Application entry point
├── config.py           # Configuration settings
├── core/               # Base classes (VolumeData, BaseLoader, BaseProcessor)
├── loaders/            # Data loading strategies
│   ├── dicom.py        # DICOM series loaders
│   └── dummy.py        # Synthetic data generator
├── processors/         # Analysis algorithms
│   ├── pore.py         # Void space extraction
│   └── pnm.py          # Pore Network Modeling (PNM)
├── exporters/          # Data export handlers
│   └── vtk.py          # VTK format exporter
├── data/               # Data management
│   └── manager.py      # Scientific workflow state
├── gui/                # User interface
│   ├── main_window.py  # Main application window
│   └── panels/         # Reusable UI panels
└── rendering/          # 3D rendering engine
    ├── render_engine.py
    ├── clip_handler.py
    └── roi_handler.py
```

## Features

### Visualization Modes
* **📊 Volume Rendering**: Full 3D density rendering with adjustable opacity transfer functions.
* **🔳 Orthogonal Slices**: Interactive X, Y, Z planes with mouse probe (shows XYZ coordinates and HU values).
* **🏔️ Isosurface**: Solid-void interface with multiple coloring modes (Solid, Depth, Radial Distance).
* **⚪ PNM Mesh**: Network topology visualization with Pores (Spheres) and Throats (Tubes).

### Structural Analysis
* **Void Extraction**: Segments air/void voxels from solid matrix using intensity thresholding.
* **Pore Network Modeling (PNM)**: Watershed segmentation with Ball-and-Stick model generation.

### Data IO
* **Load DICOM**: Standard CT image series support.
* **Fast Load**: Downsampled preview for large datasets.
* **Synthetic Generator**: Gaussian Random Field volume for testing.

## Installation

### Requirements
* Python 3.8+
* See `requirements.txt` for dependencies

### Setup
```bash
pip install -r requirements.txt
python app.py
```

## Dependencies

| Package | Purpose |
|---------|---------|
| PyQt5 | GUI framework |
| pyvista, pyvistaqt, vtk | 3D rendering |
| numpy, scipy, scikit-image | Image processing |
| pydicom | DICOM data loading |
| joblib, numba (optional) | Performance optimization |