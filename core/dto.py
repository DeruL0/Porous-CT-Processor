"""
Data Transfer Objects (DTOs) for the 4-DCT analysis pipeline.

Design rules
------------
* All DTOs are immutable (frozen=True).  Core logic never calls into the GUI;
  the GUI builds a new DTO and *pushes* it to the engine.
* No PyQt5 imports anywhere in this module.
* ``from_dict`` / ``from_panel`` factory methods keep serialisation in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List


# ---------------------------------------------------------------------------
# Render parameters DTO
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RenderParamsDTO:
    """
    Immutable snapshot of every rendering parameter.

    Replaces direct calls to ``params_panel.get_current_values()`` inside the
    render engine, making the engine headless-capable.
    """

    threshold:      int                     = 300
    coloring_mode:  str                     = "Scientific Colormap"
    colormap:       str                     = "viridis"
    solid_color:    str                     = "ivory"
    light_angle:    int                     = 45
    render_style:   str                     = "Surface"
    opacity:        str                     = "linear"
    slice_x:        int                     = 50
    slice_y:        int                     = 50
    slice_z:        int                     = 50
    clim:           Tuple[float, float]     = (0.0, 1000.0)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RenderParamsDTO":
        """Build from a plain dictionary (e.g. from params_panel.get_current_values())."""
        clim_raw = d.get("clim", [0.0, 1000.0])
        return RenderParamsDTO(
            threshold     = int(d.get("threshold",     300)),
            coloring_mode = str(d.get("coloring_mode", "Scientific Colormap")),
            colormap      = str(d.get("colormap",      "viridis")),
            solid_color   = str(d.get("solid_color",   "ivory")),
            light_angle   = int(d.get("light_angle",   45)),
            render_style  = str(d.get("render_style",  "Surface")),
            opacity       = str(d.get("opacity",       "linear")),
            slice_x       = int(d.get("slice_x",       50)),
            slice_y       = int(d.get("slice_y",       50)),
            slice_z       = int(d.get("slice_z",       50)),
            clim          = (float(clim_raw[0]), float(clim_raw[1])),
        )

    @staticmethod
    def from_panel(panel) -> "RenderParamsDTO":
        """
        Build from a live RenderingParametersPanel without importing PyQt.

        This is the *only* place where GUI state is read out; after this call
        the resulting DTO can be passed deep into the rendering stack or
        serialised to YAML without touching the UI thread again.
        """
        return RenderParamsDTO.from_dict(panel.get_current_values())

    def to_dict(self) -> Dict[str, Any]:
        """Serialise back to a plain dictionary (for YAML / JSON export)."""
        return {
            "threshold":     self.threshold,
            "coloring_mode": self.coloring_mode,
            "colormap":      self.colormap,
            "solid_color":   self.solid_color,
            "light_angle":   self.light_angle,
            "render_style":  self.render_style,
            "opacity":       self.opacity,
            "slice_x":       self.slice_x,
            "slice_y":       self.slice_y,
            "slice_z":       self.slice_z,
            "clim":          list(self.clim),
        }


# ---------------------------------------------------------------------------
# Volume processing DTO
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VolumeProcessDTO:
    """
    Immutable configuration for a headless processing run.

    Used by the CLI and by unit tests that bypass the GUI entirely.
    """

    # Input
    input_path:       str                       = ""
    loader_type:      str                       = "dicom"    # "dicom" | "dummy"
    load_strategy:    str                       = "auto"     # "auto" | "full" | "fast" | "mmap" | "chunked"

    # Segmentation
    threshold:        float                     = 300.0
    auto_threshold:   bool                      = False
    threshold_algorithm: str                    = "auto"     # "auto" | "otsu" | "li" | "yen" | "triangle" | "minimum"

    # PNM
    min_pore_size:    int                       = 100
    min_throat_dist:  int                       = 3

    # Output
    output_dir:       Optional[str]             = None
    export_formats:   Tuple[str, ...]           = ("vtk",)   # "vtk", "tiff", "npy"

    # Chunking
    chunk_shape:      Tuple[int, int, int]      = (128, 128, 128)
    halo_voxels:      int                       = 8

    # Render (headless preview / thumbnails)
    render_params:    Optional[RenderParamsDTO] = None

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "VolumeProcessDTO":
        rp_raw = d.get("render_params")
        rp     = RenderParamsDTO.from_dict(rp_raw) if rp_raw else None
        return VolumeProcessDTO(
            input_path      = str(d.get("input_path",      "")),
            loader_type     = str(d.get("loader_type",     "dicom")),
            load_strategy   = str(d.get("load_strategy",   "auto")),
            threshold       = float(d.get("threshold",     300.0)),
            auto_threshold  = bool(d.get("auto_threshold", False)),
            threshold_algorithm = str(d.get("threshold_algorithm", "auto")),
            min_pore_size   = int(d.get("min_pore_size",   100)),
            min_throat_dist = int(d.get("min_throat_dist", 3)),
            output_dir      = d.get("output_dir"),
            export_formats  = tuple(d.get("export_formats", ["vtk"])),
            chunk_shape     = tuple(int(x) for x in d.get("chunk_shape", [128, 128, 128])),
            halo_voxels     = int(d.get("halo_voxels",     8)),
            render_params   = rp,
        )

    @staticmethod
    def from_yaml(path: str) -> "VolumeProcessDTO":
        """Load config from a YAML file (no PyQt dependency)."""
        import yaml  # soft dependency — only needed for CLI
        with open(path, encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
        return VolumeProcessDTO.from_dict(d or {})

    @staticmethod
    def from_json(path: str) -> "VolumeProcessDTO":
        """Load config from a JSON file."""
        import json
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return VolumeProcessDTO.from_dict(d)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_path":      self.input_path,
            "loader_type":     self.loader_type,
            "load_strategy":   self.load_strategy,
            "threshold":       self.threshold,
            "auto_threshold":  self.auto_threshold,
            "threshold_algorithm": self.threshold_algorithm,
            "min_pore_size":   self.min_pore_size,
            "min_throat_dist": self.min_throat_dist,
            "output_dir":      self.output_dir,
            "export_formats":  list(self.export_formats),
            "chunk_shape":     list(self.chunk_shape),
            "halo_voxels":     self.halo_voxels,
            "render_params":   self.render_params.to_dict() if self.render_params else None,
        }
