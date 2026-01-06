from typing import Optional
from core import VolumeData


class ScientificDataManager:
    """
    Central Data Management Class for Scientific Computing.

    Manages the three critical stages of the Porous Media Analysis workflow:
    1. Raw CT Data (Original Scan)
    2. Segmented Volume (Distinguishing Air/Void from Solid)
    3. Pore Network Model (PNM) Mesh (Distinguishing Pores vs Throats)
    4. ROI Data (Region of Interest extracted sub-volume)
    """

    def __init__(self):
        # 1. 未提取孔洞之前的原始CT数据
        self.raw_ct_data: Optional[VolumeData] = None

        # 2. 提取孔洞后的划分体积数据 (Air/Solid Binary Volume)
        self.segmented_volume: Optional[VolumeData] = None

        # 3. 构建PNW后能够区分孔洞，空气，吼道 (Optimized Mesh)
        self.pnm_model: Optional[VolumeData] = None
        
        # 4. ROI提取后的子体积
        self.roi_data: Optional[VolumeData] = None

    @property
    def active_data(self) -> Optional[VolumeData]:
        """
        Returns the most relevant data for processing.
        Priority: ROI > Raw (we process on extracted region if available)
        """
        if self.roi_data is not None:
            return self.roi_data
        return self.raw_ct_data

    def load_raw_data(self, data: VolumeData):
        """Sets the raw input data."""
        self.raw_ct_data = data
        # Reset downstream data when new raw data is loaded
        self.segmented_volume = None
        self.pnm_model = None
        self.roi_data = None  # Clear ROI when loading new data

    def set_segmented_data(self, data: VolumeData):
        """Stores the intermediate segmented void space."""
        self.segmented_volume = data

    def set_pnm_data(self, data: VolumeData):
        """Stores the generated Pore Network Model mesh."""
        self.pnm_model = data

    def set_roi_data(self, data: VolumeData):
        """Stores ROI-extracted sub-volume for focused analysis."""
        self.roi_data = data
        # Clear downstream when ROI changes
        self.segmented_volume = None
        self.pnm_model = None

    def clear_roi(self):
        """Clears ROI data, reverting to full raw volume."""
        self.roi_data = None
        self.segmented_volume = None
        self.pnm_model = None

    def get_current_state_info(self) -> str:
        """Returns a status string describing what data is available."""
        status = []
        if self.raw_ct_data:
            status.append("✔ Raw CT Data")
        else:
            status.append("✘ Raw CT Data")
            
        if self.roi_data:
            status.append("✔ ROI Extracted")

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
    
    def has_active_data(self) -> bool:
        return self.active_data is not None

    def has_segmented(self) -> bool:
        return self.segmented_volume is not None