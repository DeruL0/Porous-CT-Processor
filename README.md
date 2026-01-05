
# Porous CT Analysis Suite

A scientific computing application for analyzing porous materials (rocks, ceramics, foams, etc.) using Micro-CT data. Built with **Python**, **PyQt5**, and **PyVista**.

## Overview

This application provides a comprehensive workflow for Digital Rock Physics (DRP):
1.  **Ingestion**: Load industrial DICOM series or generate synthetic test data.
2.  **Visualization**: Interactive 3D rendering with orthogonal slices and isosurfaces.
3.  **Quantification**: Extract porosity and segment void space.
4.  **Modeling**: Generate Pore Network Models (PNM) using watershed segmentation.
5.  **Export**: Save results to VTK standards (.vtp/.vti) for simulation software.

## System Architecture

The application follows a strict **Model-View-Controller (MVC)** pattern for extensibility:

* **`App.py`**: Main entry point and Application Controller.
* **`Core.py`**: Defines standard data structures (`VolumeData`) for Voxel and Mesh data.
* **`Processors.py`**: Algorithms for segmentation (Watershed, Distance Transform) and topology extraction.
* **`Visualizers.py`**: Manages the PyVista 3D canvas and rendering logic.
* **`Loaders.py`**: Strategies for loading DICOM folders and handling downsampling.
* **`Exporters.py`**: Handles conversion of internal data to VTK formats.

## Features

### 1. Visualization Modes
* **📊 Volume Rendering**: Full 3D density rendering with adjustable opacity transfer functions (Sigmoid, Linear).
* **🔳 Orthogonal Slices**: Interactive X, Y, Z planes to inspect internal defects.
* **🏔️ Isosurface**: Extract the solid-void interface.
    * *Coloring Modes*: Solid Color, Depth (Z-Axis), and **Radial Distance** (visualization of core vs. shell structure).
* **⚪ PNM Mesh**: Visualizes the network topology with Pores (Spheres) and Throats (Tubes).

### 2. Structural Analysis
* **Void Extraction**: Segments air/void voxels from the solid matrix based on intensity thresholding.
* **Pore Network Modeling (PNM)**:
    * Uses **Watershed Segmentation** on the distance map.
    * Generates a **Ball-and-Stick model**:
        * **Nodes**: Represent pore bodies (sized by equivalent radius).
        * **Edges**: Represent throats (connections) between pores.

### 3. Data IO
* **Load Dicom**: Reads standard CT image series.
* **Fast Load**: Downsamples large datasets (Step=2) for quick previewing.
* **Synthetic Generator**: Creates a Gaussian Random Field volume with a solid shell for testing algorithms without external data.

## Installation

### Requirements
* Python 3.8+
* Dependencies listed in `requirements.txt`:
    * `PyQt5` (GUI)
    * `pyvista`, `pyvistaqt`, `vtk` (3D Rendering)
    * `numpy`, `scipy`, `scikit-image` (Image Processing)
    * `pydicom` (Data Loading)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python App.py