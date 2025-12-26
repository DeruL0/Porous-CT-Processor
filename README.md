
# Porous CT Analysis Suite

An analysis suite for **Porous CT**, built with PyQt5 and PyVista.

## Overview

This application provides a comprehensive suite for analyzing porous materials (such as sedimentary rocks, ceramics, foams, and foods) using Micro-CT data.

-   **Interactive 3D visualization** using PyVista.
    
-   **Pore Network Modeling (PNM)**: Extract pore bodies and throats from raw scans.
    
-   **Pore Space Extraction**: Quantify and visualize porosity.
    
-   **Industrial CT Support**: Compatible with DICOM series from Micro-CT scanners.
    

## Features

### Visualization Modes

1.  **Volume Rendering** - Full 3D volumetric visualization of the material matrix with adjustable opacity transfer functions.
    
2.  **Orthogonal Slices** - Inspect internal structure via X, Y, Z planes.
    
3.  **Isosurface** - Extract and display the solid-void interface (surface mesh). Includes coloring and lighting options.
    

### Structural Analysis

1.  **Void Space Extraction** - Identify and segregate the pore phase from the solid matrix using morphological operations.
    
2.  **Pore Network Model (PNM)** - Advanced watershed segmentation to create topological ball-and-stick models:
    
    -   **Nodes (Spheres)**: Represent pore bodies.
        
    -   **Edges (Cylinders)**: Represent throats/constrictions connecting pores.
        
    -   Used for fluid flow simulation and permeability analysis.
        

### Loading Options

-   **Sample Scan Load** - Full resolution loading of DICOM series.
    
-   **Fast Load** - Downsampled data for quick previews of large datasets.
    
-   **Synthetic Sample** - Generate a synthetic porous structure (Hollow Sphere Phantom) for testing.
    

## Installation

### Requirements

-   Python 3.8 or higher
    
-   PyQt5
    
-   PyVista and pyvistaqt
    
-   NumPy, SciPy, scikit-image
    
-   pydicom
    

### Install Dependencies

```
pip install -r requirements.txt
```

## Usage

### Running the Application

**Option 1: Start with GUI**

```
python App.py
```

### GUI Controls

The interface features a **Dynamic Control Panel** that automatically adjusts based on your current viewing mode (Volume, Slices, or Isosurface).

#### Analysis Modes

-   📊 **Volume Rendering** - View density distribution.
    
-   🔳 **Orthogonal Slices** - Inspect internal defects or pores.
    
-   🏔️ **Isosurface** - Visualize the solid surface or pore boundary.
    
-   🗑️ **Clear View** / 🎥 **Reset Camera** - View controls.
    

#### Rendering Parameters (Context-Aware)

**Global / Volume Mode:**

-   **Colormap** - Analysis-friendly palettes (Viridis, Plasma, Bone, Jet, Magma).
    
-   **Opacity Preset** - Adjust transparency (Sigmoid, Linear, Geometric) to see inside the volume.
    

**Isosurface Mode:**

-   **Iso-Threshold** - Adjust intensity cutoff to separate solid from void.
    
-   **Coloring Mode**:
    
    -   _Solid Color_: Clean, uniform color (Ivory, Red, Gold, etc.).
        
    -   _Depth (Z-Axis)_: Colors the mesh based on height/depth.
        
    -   _Radial (Center Dist)_: Colors based on distance from the center (useful for visualizing core vs. shell).
        
-   **Light Source Angle** - Rotatable directional light (0-360°) to enhance surface details and depth perception.
    

#### Structure Processing

-   📁 **Load Sample Scan** - Load Micro-CT data.
    
-   🔬 **Extract Pore Space** - Segment the pore phase.
    
-   ⚪ **Pore Network Model** - Generate topological network.
    
-   ↩️ **Reset to Raw Data** - Revert to original scan.
    

## Workflow Example

1.  **Load Data**: Click "📁 Load Sample Scan" to open a folder of CT slice images.
    
2.  **Inspect**: Use "📊 Volume Rendering" to see the material density.
    
3.  **Segment**: Click "🔬 Extract Void Space" to isolate pores.
    
4.  **Visualize Pores**: Switch to "🏔️ Isosurface". Change "Coloring Mode" to "Radial" to see the depth of the pore structure from the center outwards. Adjust the "Light Angle" to catch shadows in the crevices.
    
5.  **Model**: Click "⚪ Pore Network Model" to generate the Ball-and-Stick network.
    

## Algorithm Details

### Pore Network Extraction (Watershed)

This tool implements a standard Digital Rock Physics pipeline:

1.  **Binarization**: Thresholding to separate Void vs Solid.
    
2.  **Distance Map**: Compute distance from pore voxels to solid walls.
    
3.  **Seed Detection**: Find centers of largest pore spaces (local maxima).
    
4.  **Watershed Segmentation**: Partition pore space into discrete regions.
    
5.  **Adjacency Matrix**: Determine connectivity between regions to build the topological graph.
    

## License

This software is provided for research and educational purposes in the field of Porous Media Analysis.