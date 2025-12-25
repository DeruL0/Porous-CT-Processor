import os
from typing import Optional

# Import separated modules
from Core import BaseLoader, VolumeData
from Loaders import DicomSeriesLoader, DummyLoader, FastDicomLoader
from Processors import PoreExtractionProcessor, PoreToSphereProcessor
from Visualizers import PyVistaVisualizer


# ==========================================
# 6. Controller Layer
# ==========================================

class AppController:
    """Application controller, coordinates loader, processor, and visualizer"""

    def __init__(self):
        self.visualizer = PyVistaVisualizer()
        self.loader: Optional[BaseLoader] = None
        self.processor = PoreExtractionProcessor()
        self.sphere_processor = PoreToSphereProcessor()  # Instantiate new processor

        self.original_data: Optional[VolumeData] = None
        self.current_data: Optional[VolumeData] = None

    def run(self, source_path: Optional[str] = None):
        """Main run logic"""
        try:
            # 1. Select loading strategy
            if source_path and os.path.exists(source_path):
                print("\nDetection a valid path. Select loading mode:")
                print("1. Standard Load (Full Resolution)")
                print("2. Fast Load (Low Resolution, Step=2)")
                mode = input("Choice (default 1): ").strip()

                if mode == '2':
                    self.loader = FastDicomLoader(step=2)
                else:
                    self.loader = DicomSeriesLoader()

                load_arg = source_path
            else:
                print("Warning: No valid path provided, switching to dummy data mode.")
                self.loader = DummyLoader()
                load_arg = 128

            # 2. Load data
            self.original_data = self.loader.load(load_arg)
            self.current_data = self.original_data

            # 3. Inject data into visualizer
            self.visualizer.set_data(self.current_data)

            # 4. User interaction loop
            self._interaction_loop()

        except Exception as e:
            print(f"Program error: {e}")
            import traceback
            traceback.print_exc()

    def _interaction_loop(self):
        while True:
            print("\n" + "=" * 40)
            print("   Medical Imaging Visualization Console")
            print("=" * 40)
            print(f"Current Data: {self.current_data.metadata.get('Type', 'Original')}")
            print(f"Dimensions: {self.current_data.dimensions}")
            print("1. Volume Rendering")
            print("2. Orthogonal Slices")
            print("3. Isosurface (Bone/Surface)")
            print("4. Partition Pores (Raw Shape)")
            print("5. Convert Pores to Spheres")
            print("r. Reset to Original")
            print("q. Quit")

            choice = input("\nPlease enter option: ").strip().lower()

            if choice == '1':
                self.visualizer.render_volume()

            elif choice == '2':
                self.visualizer.render_slices()

            elif choice == '3':
                data_type = self.current_data.metadata.get("Type", "")

                if "Pore" in data_type or "Spheres" in data_type:
                    # If it is pore/sphere data, display in red
                    self.visualizer.render_isosurface(threshold=500, color="red")
                else:
                    # If it is original data, use bone color
                    is_synthetic = data_type == "Synthetic"
                    thresh = 800 if is_synthetic else 300
                    self.visualizer.render_isosurface(threshold=thresh, color="ivory")

            elif choice == '4':
                if self.original_data is None: continue
                # Execute processing (Raw Pores)
                print("Executing pore detection...")
                is_synthetic = self.original_data.metadata.get("Type") == "Synthetic"
                thresh = 500 if is_synthetic else -300

                self.current_data = self.processor.process(self.original_data, threshold=thresh)
                self.visualizer.set_data(self.current_data)
                print("Pore extraction complete! Select option 3 to view results in 3D.")

            elif choice == '5':
                if self.original_data is None: continue
                # Execute processing (Spheres)
                print("Executing pore-to-sphere conversion...")
                is_synthetic = self.original_data.metadata.get("Type") == "Synthetic"
                thresh = 500 if is_synthetic else -300

                self.current_data = self.sphere_processor.process(self.original_data, threshold=thresh)
                self.visualizer.set_data(self.current_data)
                print(
                    f"Conversion complete! Found {self.current_data.metadata.get('PoreCount')} pores. Select option 3 to view.")

            elif choice == 'r':
                self.current_data = self.original_data
                self.visualizer.set_data(self.current_data)
                print("Reset to original data.")

            elif choice == 'q':
                print("Exiting program.")
                break
            else:
                print("Invalid option, please try again.")


# ==========================================
# Program Entry Point
# ==========================================

if __name__ == "__main__":
    # Usage example
    # You can specify a path here, or leave it empty to use dummy data
    dicom_path = 'D:\zhu shuyang_25_tart_114913'  # "C:/MedicalData/Patient01"

    app = AppController()
    app.run(dicom_path)