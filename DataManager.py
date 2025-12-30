from typing import Optional
from Core import VolumeData


class ScientificDataManager:
    """
    Central Data Management Class for Scientific Computing.

    Manages the three critical stages of the Porous Media Analysis workflow:
    1. Raw CT Data (Original Scan)
    2. Segmented Volume (Distinguishing Air/Void from Solid)
    3. Pore Network Model (PNM) Mesh (Distinguishing Pores vs Throats)
    """

    def __init__(self):
        # 1. 未提取孔洞之前的原始CT数据
        self.raw_ct_data: Optional[VolumeData] = None

        # 2. 提取孔洞后的划分体积数据 (Air/Solid Binary Volume)
        self.segmented_volume: Optional[VolumeData] = None

        # 3. 构建PNW后能够区分孔洞，空气，吼道 (Optimized Mesh)
        # Note: 'Air' context is preserved in segmented_volume,
        # while PNM Mesh explicitly distinguishes Pores (Nodes) and Throats (Edges).
        self.pnm_model: Optional[VolumeData] = None

    def load_raw_data(self, data: VolumeData):
        """Sets the raw input data."""
        self.raw_ct_data = data
        # Reset downstream data when new raw data is loaded
        self.segmented_volume = None
        self.pnm_model = None

    def set_segmented_data(self, data: VolumeData):
        """Stores the intermediate segmented void space."""
        self.segmented_volume = data

    def set_pnm_data(self, data: VolumeData):
        """Stores the generated Pore Network Model mesh."""
        self.pnm_model = data

    def get_current_state_info(self) -> str:
        """Returns a status string describing what data is available."""
        status = []
        if self.raw_ct_data:
            status.append("✔ Raw CT Data")
        else:
            status.append("✘ Raw CT Data")

        if self.segmented_volume:
            status.append("✔ Segmented Void Volume")
        else:
            status.append("✘ Segmented Void Volume")

        if self.pnm_model:
            status.append("✔ PNM Mesh (Pores & Throats)")
        else:
            status.append("✘ PNM Mesh")

        return " | ".join(status)

    def has_raw(self) -> bool:
        return self.raw_ct_data is not None

    def has_segmented(self) -> bool:
        return self.segmented_volume is not None