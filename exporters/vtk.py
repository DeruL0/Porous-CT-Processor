"""
VTK format exporter for volumetric and mesh data.
"""

import numpy as np
import pyvista as pv

from core import VolumeData


class VTKExporter:
    """
    负责将 VolumeData (体素或网格) 导出为 VTK 标准格式文件。
    Handles logic for .vti (Image Data) and .vtp (Poly Data).
    """

    @staticmethod
    def export(data: VolumeData, filepath: str) -> bool:
        """
        根据数据类型自动选择导出策略。

        Args:
            data: 包含 raw_data 或 mesh 的 VolumeData 对象。
            filepath: 目标文件路径。

        Returns:
            bool: 成功返回 True。
        """
        if data is None:
            raise ValueError("No data to export.")

        if data.has_mesh:
            return VTKExporter._export_mesh(data.mesh, filepath)
        elif data.raw_data is not None:
            return VTKExporter._export_volume(data, filepath)
        else:
            raise ValueError("No exportable data found (neither Mesh nor Volume).")

    @staticmethod
    def _export_mesh(mesh: pv.PolyData, filepath: str) -> bool:
        """导出 PNM 网格数据 (.vtp, .vtk)"""
        if "IsPore" in mesh.array_names:
            print(f"[Exporter] Detected 'IsPore' attribute. Pore=1, Throat=0.")
        else:
            print(f"[Exporter] Warning: 'IsPore' attribute missing in mesh.")

        mesh.save(filepath)
        print(f"[Exporter] Mesh saved to {filepath}")
        return True

    @staticmethod
    def _export_volume(data: VolumeData, filepath: str) -> bool:
        """导出体素数据 (.vti)"""
        grid = pv.ImageData()
        grid.dimensions = np.array(data.raw_data.shape) + 1
        grid.origin = data.origin
        grid.spacing = data.spacing

        grid.cell_data["values"] = data.raw_data.flatten(order="F")

        grid.save(filepath)
        print(f"[Exporter] Volume saved to {filepath}")
        return True
