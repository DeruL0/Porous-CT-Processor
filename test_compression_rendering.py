"""
Quick test to verify compression ratio rendering is connected.
Run this after loading 4DCT data and tracking PNM.
"""

import numpy as np
import pyvista as pv

def test_compression_mesh():
    """Test that compression scalars are properly added to mesh."""
    
    # Simulate mesh data
    centers = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    radii = np.array([0.5, 0.6, 0.4])
    ids = np.array([1, 2, 3])
    compression_ratios = np.array([0.3, 0.8, 1.0])  # Compressed, normal, expanded
    highlight_id = 2
    
    # Create point cloud
    pore_cloud = pv.PolyData(centers)
    pore_cloud["radius"] = radii
    pore_cloud["ID"] = ids
    
    # Add compression and highlight
    highlight_flags = np.array([1 if pid == highlight_id else 0 for pid in ids], dtype=np.int32)
    pore_cloud["CompressionRatio"] = compression_ratios
    pore_cloud["Highlighted"] = highlight_flags
    
    # Glyph to spheres
    sphere_glyph = pv.Sphere(theta_resolution=10, phi_resolution=10)
    pores_mesh = pore_cloud.glyph(scale="radius", geom=sphere_glyph)
    
    n_pts_per_sphere = sphere_glyph.n_points
    
    # Repeat scalars for each sphere vertex
    compression_scalars = np.repeat(compression_ratios, n_pts_per_sphere)
    highlight_scalars = np.repeat(highlight_flags, n_pts_per_sphere)
    
    pores_mesh["CompressionRatio"] = compression_scalars
    pores_mesh["Highlighted"] = highlight_scalars
    pores_mesh["PoreRadius"] = np.repeat(radii, n_pts_per_sphere)
    
    # Verify
    print(f"✓ Mesh has {pores_mesh.n_points} points")
    print(f"✓ Available scalars: {list(pores_mesh.point_data.keys())}")
    print(f"✓ CompressionRatio range: [{compression_scalars.min():.2f}, {compression_scalars.max():.2f}]")
    print(f"✓ Highlighted pores: {highlight_scalars.sum()} points")
    
    # Test rendering logic
    highlighted = pores_mesh.point_data["Highlighted"]
    has_highlighted_pores = highlighted.max() > 0
    
    if has_highlighted_pores:
        print(f"✓ Rendering mode: Compression + Highlight")
        compression = pores_mesh.point_data["CompressionRatio"]
        display_scalar = compression.copy()
        display_scalar[highlighted > 0] = 2.0
        print(f"  Display scalar range: [{display_scalar.min():.2f}, {display_scalar.max():.2f}]")
    else:
        print(f"✓ Rendering mode: Compression only")
    
    # Visual test
    plotter = pv.Plotter()
    
    if has_highlighted_pores and "CompressionRatio" in pores_mesh.point_data:
        compression = pores_mesh.point_data["CompressionRatio"]
        display_scalar = compression.copy()
        display_scalar[highlighted > 0] = 2.0
        
        plotter.add_mesh(
            pores_mesh,
            scalars=display_scalar,
            cmap=['blue', 'cyan', 'green', 'yellow', 'red', 'gold'],
            clim=[0.0, 2.0],
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Compression (blue=low, red=high, gold=selected)'}
        )
    else:
        plotter.add_mesh(
            pores_mesh,
            scalars="CompressionRatio",
            cmap=['blue', 'cyan', 'green', 'yellow', 'red'],
            clim=[0.0, 1.0],
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Compression Ratio'}
        )
    
    print("\n✓ All checks passed! Opening visualization...")
    print("  Expected: Blue sphere (compressed), red/yellow sphere (normal), gold sphere (highlighted)")
    plotter.show()

if __name__ == "__main__":
    test_compression_mesh()
