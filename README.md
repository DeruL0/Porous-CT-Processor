
# Porous CT Analysis Suite

A comprehensive Python-based desktop application for analyzing porous materials (rocks, ceramics, foams, etc.) using Micro-CT data. This tool is designed for Digital Rock Physics (DRP) and materials science, offering robust 3D volume visualization, advanced Pore Network Modeling (PNM), and 4D time-series tracking of pore evolution.

## Overview

The Porous CT Analysis Suite allows users to ingest industrial DICOM series or 4D-CT time-series datasets, extract void spaces, and build detailed analytical models of internal structures. It features GPU-accelerated processing via CuPy, making it highly capable of tracking pore evolution over time and generating complex Ball-and-Stick pore network models for scientific simulations.

## Key Features

**Data Import and Generation**

-   Support for standard DICOM image series and multi-folder 4D-CT time-series datasets.
    
-   Fast loading with optional downsampled previews for large datasets.
    
-   Built-in Synthetic Generator (Gaussian Random Field) for testing and validation.
    

**Pore Extraction & Modeling**

-   Automated void/air segmentation using customizable intensity thresholding.
    
-   Pore Network Modeling (PNM) using watershed segmentation to generate physical Ball-and-Stick network models.
    
-   Smart caching system that stores generated PNM results to prevent redundant recalculations during view switching.
    

**4D Tracking & Evolution**

-   Advanced tracking of individual pores across multiple time steps to analyze structural evolution.
    
-   Optimized Batch IoU calculation (3-5x faster) and GPU-accelerated tracking algorithms (5-10x speedup for 500+ pores).
    
-   Automatic hardware-based algorithm selection and optional Hungarian matching for globally optimal assignments in complex scenarios.
    

**High-Performance Computing**

-   GPU acceleration (via CuPy) for computationally heavy tasks including Euclidean Distance Transforms (EDT), watershed operations, and 4D tracking.
    
-   Fallback CPU implementations (via scikit-image and SciPy) to ensure compatibility across all hardware.
    

**Visualization and Export**

-   Interactive 3D mesh and volume rendering (powered by PyVista), including orthogonal slices and isosurfaces.
    
-   Dedicated GUI panels for Time-Series Control and Tracking Analysis.
    
-   Export of resulting geometries and structures to standard VTK formats (`.vtp`, `.vti`) and statistical summaries to JSON.
    

## Prerequisites

-   Python 3.8 or higher
    
-   A compatible NVIDIA GPU (highly recommended for performance and 4D tracking acceleration)
    

## Core Dependencies

-   `PyQt5` (GUI framework)
    
-   `numpy`
    
-   `scipy`
    
-   `scikit-image` (CPU-based watershed and image processing)
    
-   `pydicom` (DICOM data ingestion)
    
-   `networkx` (PNM graph structure analysis)
    

## Optional but Recommended Dependencies

-   `cupy` (Required for GPU acceleration, fast EDT, and optimized 4D tracking)
    
-   `pyvista` and `pyvistaqt` (Required for high-performance 3D rendering and visualization)
    
-   `pandas` (For advanced statistical dataframe exports)
    

## Installation

Clone the repository:

```
git clone [https://github.com/yourusername/porous-ct-analysis.git](https://github.com/yourusername/porous-ct-analysis.git)
cd porous-ct-analysis
```

Create a virtual environment (recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Install the required dependencies:

```
pip install -r requirements.txt
```

_(Optional)_ Install CuPy for your specific CUDA version to enable GPU acceleration. For example, for CUDA 11.x:

```
pip install cupy-cuda11x
```

## Usage

Start the application by running the main entry point:

```
python App.py
```

## Basic Workflow

1.  **Import Data:** Open the application and use the "File" menu to load a DICOM series, a 4D CT folder, or generate synthetic test data.
    
2.  **Process Structure:** Navigate to the "Structure Processing" panel on the left. Set your target density/intensity threshold to segment the air/void voxels.
    
3.  **Generate PNM:** Click "Generate PNM". The software will run a watershed segmentation algorithm to build a comprehensive Pore Network Model (nodes and edges).
    
4.  **4D Tracking (Optional):** If a time-series was loaded, open the "Tracking & Analysis" panel. Select your preferred tracking mode (Fast/Accurate) to calculate pore evolution across time steps.
    
5.  **Visualize & Navigate:** Use the "Time Series Control" panel to smoothly transition between time steps. Utilize the PyVista-powered viewport to inspect 3D isosurfaces, orthogonal slices, and tracked connection lines.
    
6.  **Export:** Export the processed simulation ready meshes to VTK format (`.vtp`/`.vti`) or save the statistical breakdown to JSON using the "Export" options.
    

## Project Structure

-   `App.py`: Application entry point and MVC controller coordinating UI state and workflow handlers.
    
-   `config.py`: Default configuration parameters and constants.
    
-   `core/`: Base data structures, GPU backend wrappers, and `TimeSeriesPNM` logic.
    
-   `data/`: Centralized `ScientificDataManager` and state/caching management.
    
-   `exporters/`: Modules for exporting data (`VTKExporter`, `JSONExporter`).
    
-   `gui/`: PyQt5 user interface components, modular panels (`TrackingAnalysisPanel`, `TimeSeriesControlPanel`), main window, and workflow handlers.
    
-   `loaders/`: Data ingestion strategies utilizing `pydicom` and synthetic generation (`DicomLoader`, `DummyLoader`).
    
-   `processors/`: The core algorithmic engines (`PoreProcessor`, `PNMExtractor`, `PNMTracker`).
    
-   `tests/`: Automated unit tests for core functionality and caching mechanisms.