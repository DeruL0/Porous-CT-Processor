# Medical Imaging Visualization Suite - GUI Version

A modern graphical user interface (GUI) for medical imaging visualization and pore network analysis, built with PyQt5 and PyVista.

## Overview

This application replaces the command-line interface with a fully-featured GUI that provides:
- **Interactive 3D visualization** using PyVista
- **Multiple rendering modes**: Volume rendering, orthogonal slices, and isosurfaces
- **Advanced processing**: Pore extraction and pore network modeling (watershed segmentation)
- **Real-time parameter adjustment**: Threshold, colormap, and opacity controls
- **DICOM support**: Load medical imaging data with both standard and fast (low-resolution) modes

## Features

### Visualization Modes
1. **Volume Rendering** - Full 3D volumetric visualization with customizable colormaps and opacity
2. **Orthogonal Slices** - View data in three perpendicular planes
3. **Isosurface** - Extract and display surfaces at specific threshold values

### Data Processing
1. **Pore Extraction (Raw)** - Identify and extract pore spaces using morphological operations
2. **Pore Network Model** - Advanced watershed segmentation to create ball-and-stick network models
   - Detects individual pores as spheres
   - Identifies connections (throats) between pores
   - Visualizes as a connected network

### Loading Options
- **Standard DICOM Load** - Full resolution data loading
- **Fast Load** - Downsampled data for quick previews (1/8th volume size)
- **Dummy Data** - Generate synthetic test data (hollow sphere)

## Installation

### Requirements
- Python 3.8 or higher
- PyQt5
- PyVista and pyvistaqt
- NumPy, SciPy, scikit-image
- pydicom

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy scipy scikit-image pydicom pyvista pyvistaqt PyQt5 vtk
```

## Usage

### Running the Application

**Option 1: Start with GUI (recommended)**
```bash
python AppGUI.py
```

**Option 2: Auto-load DICOM data on startup**
```python
# Edit AppGUI.py, set dicom_path to your folder:
dicom_path = r'C:\path\to\your\dicom\folder'
```

Then run:
```bash
python AppGUI.py
```

### GUI Controls

#### Left Control Panel

**Data Information**
- Displays current data type, dimensions, spacing, and metadata

**Visualization Modes**
- 📊 **Volume Rendering** - 3D volumetric view
- 🔲 **Orthogonal Slices** - Multi-plane slicing
- 🎯 **Isosurface** - Surface extraction
- 🗑️ **Clear View** - Remove all visualizations
- 📷 **Reset Camera** - Reset to default view

**Rendering Parameters**
- **Isosurface Threshold** - Slider to adjust threshold value (-1000 to 2000 HU)
- **Colormap** - Choose from bone, viridis, plasma, gray, coolwarm, jet
- **Opacity Preset** - Select opacity transfer function

**Data Processing**
- 📁 **Load DICOM Series** - Browse and load full-resolution DICOM data
- ⚡ **Fast Load (Low-Res)** - Load downsampled data for quick preview
- 🧪 **Load Dummy Data** - Generate synthetic test data
- 🔬 **Extract Pores (Raw)** - Binary pore space extraction
- ⚪ **Pore Network (Spheres)** - Create ball-and-stick pore network model
- ↩️ **Reset to Original** - Revert to original loaded data

#### Right Panel - 3D Viewer
- **Rotate**: Left mouse button + drag
- **Pan**: Middle mouse button + drag (or Shift + left mouse button)
- **Zoom**: Mouse wheel (or right mouse button + drag)

## File Structure

```
├── AppGUI.py           # Main application with GUI integration
├── GuiVisualizer.py    # GUI visualizer implementation (PyQt5 + PyVista)
├── Core.py             # Core data structures and base classes
├── Loaders.py          # Data loaders (DICOM, Fast, Dummy)
├── Processors.py       # Image processing algorithms
├── Visualizers.py      # Original PyVista visualizer (legacy)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Workflow Example

1. **Launch the application**
   ```bash
   python AppGUI.py
   ```

2. **Load data**
   - Click "📁 Load DICOM Series" and select your DICOM folder
   - Or click "🧪 Load Dummy Data" to test with synthetic data

3. **Visualize**
   - Click "📊 Volume Rendering" to see the full 3D volume
   - Try different colormaps and opacity presets

4. **Process pores**
   - Click "🔬 Extract Pores (Raw)" to identify pore spaces
   - Click "🎯 Isosurface" to view the extracted pores in 3D (shown in red)

5. **Create network model**
   - Click "⚪ Pore Network (Spheres)" to generate the ball-and-stick model
   - Click "🎯 Isosurface" to view the pore network
   - The model shows individual pores as spheres connected by cylindrical throats

6. **Adjust parameters**
   - Use the threshold slider to find the optimal isosurface value
   - Change colormaps to highlight different features

7. **Reset**
   - Click "↩️ Reset to Original" to return to the original data

## Algorithm Details

### Pore Extraction (Raw)
Uses morphological operations:
1. Threshold to create foreground mask
2. Binary hole filling to create solid mask
3. XOR operation to find the difference (pores)

### Pore Network Model (Watershed)
Advanced segmentation pipeline:
1. **Binary pore detection** - Identify pore space
2. **Euclidean Distance Transform** - Map distance from solid walls
3. **Peak detection** - Find local maxima as pore centers
4. **Watershed segmentation** - Split connected pores into individual regions
5. **Adjacency analysis** - Detect connections between pores
6. **Network construction** - Draw spheres (pores) and cylinders (throats)

## Troubleshooting

### ImportError: No module named 'PyQt5'
```bash
pip install PyQt5
```

### ImportError: No module named 'pyvistaqt'
```bash
pip install pyvistaqt
```

### Visualization not showing
- Make sure you have a compatible graphics driver
- Try updating VTK: `pip install --upgrade vtk`

### DICOM files not loading
- Ensure the folder contains valid DICOM files (.dcm)
- Try the "Fast Load" option for large datasets
- Check console output for error messages

### Out of memory
- Use "Fast Load (Low-Res)" option for large datasets
- This reduces memory usage by 8x while maintaining structure

## Performance Tips

1. **Use Fast Load** for initial exploration of large datasets
2. **Clear View** before switching between visualization modes
3. **Adjust threshold** before rendering isosurface for faster updates
4. For very large datasets (>500 slices), consider:
   - Using Fast Load mode
   - Processing only a subset of data
   - Increasing downsampling factor in `FastDicomLoader`

## Technical Notes

- The GUI uses `BackgroundPlotter` from pyvistaqt for embedded 3D rendering
- PyVista requires VTK for rendering (installed automatically)
- All processing is done in-memory; large datasets may require significant RAM
- Watershed segmentation can be computationally intensive for large volumes

## Future Enhancements

Potential additions:
- [ ] Export processed data to standard formats (VTK, STL)
- [ ] Quantitative analysis tools (pore size distribution, connectivity metrics)
- [ ] Multiple threshold isosurfaces in one view
- [ ] Animation and time-series support
- [ ] GPU-accelerated processing
- [ ] Save/load session state

## License

This software is provided as-is for educational and research purposes.

## Credits

Built with:
- **PyVista** - 3D visualization
- **PyQt5** - GUI framework  
- **scikit-image** - Image processing
- **pydicom** - DICOM file handling
- **NumPy/SciPy** - Numerical computing
