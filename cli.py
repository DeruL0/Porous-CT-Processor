"""
Headless CLI entry point for the 4-DCT Porous Media Analysis System.

Runs the full pipeline (load -> threshold -> segment -> pnm -> export) without
any display server or PyQt runtime.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


if "PyQt5" in sys.modules:
    print(
        "[CLI] WARNING: PyQt5 was imported before cli.py started. "
        "Use cli.py as process entry-point for headless runs."
    )


def _configure_headless_vtk() -> None:
    """Force VTK / PyVista into offscreen mode when no display is available."""
    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if not display:
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
        os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")


_configure_headless_vtk()


from core import VolumeProcessDTO, run_volume_pipeline
from core.progress import ProgressBus, TerminalProgressObserver


def run_batch(dto: VolumeProcessDTO) -> dict:
    """
    Execute the full pipeline using the shared DAG engine.

    Returns:
        Dict keyed by stage name containing each stage output.
    """
    progress_bus = ProgressBus().subscribe(TerminalProgressObserver())

    t_start = time.perf_counter()
    results = run_volume_pipeline(
        dto=dto,
        target_stage="export",
        include_export=True,
        progress_bus=progress_bus,
        dag_progress=progress_bus.dag_callback(),
    )
    elapsed = time.perf_counter() - t_start

    print(f"\nPipeline complete in {elapsed:.2f}s")
    exported = results.get("export", [])
    if exported:
        print("Exported files:")
        for path in exported:
            print(f"  {path}")

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description="Headless 4-DCT porous media batch processor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        metavar="FILE",
        help="Path to YAML or JSON config file. Overrides other flags.",
    )
    parser.add_argument("--input", metavar="PATH", default="", help="Input path (DICOM dir or file).")
    parser.add_argument("--loader", metavar="TYPE", default="dicom", help="Loader type: dicom | dummy.")
    parser.add_argument(
        "--load-strategy",
        metavar="MODE",
        default="auto",
        help="DICOM load strategy: auto | full | fast | mmap | chunked.",
    )
    parser.add_argument("--output", metavar="DIR", default=None, help="Output directory.")
    parser.add_argument("--threshold", metavar="VALUE", type=float, default=300.0, help="Segmentation threshold.")
    parser.add_argument("--auto-threshold", action="store_true", help="Auto-detect threshold.")
    parser.add_argument(
        "--threshold-algorithm",
        metavar="ALG",
        default="auto",
        help="Threshold algorithm when --auto-threshold is set: auto | otsu | li | yen | triangle | minimum.",
    )
    parser.add_argument(
        "--formats",
        metavar="FMT",
        nargs="+",
        default=["vtk"],
        help="Export formats: vtk tiff npy (space-separated).",
    )
    parser.add_argument("--chunk", metavar="N", type=int, default=128, help="Chunk size (single int -> NxNxN).")
    parser.add_argument("--halo", metavar="N", type=int, default=8, help="Halo voxels for chunked EDT.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved DTO without running.")
    return parser


def _resolve_dto(args: argparse.Namespace, parser: argparse.ArgumentParser) -> VolumeProcessDTO:
    """Resolve DTO from config file or inline CLI flags."""
    if args.config:
        cfg_path = args.config
        if cfg_path.endswith((".yaml", ".yml")):
            return VolumeProcessDTO.from_yaml(cfg_path)
        if cfg_path.endswith(".json"):
            return VolumeProcessDTO.from_json(cfg_path)
        try:
            return VolumeProcessDTO.from_yaml(cfg_path)
        except Exception:
            return VolumeProcessDTO.from_json(cfg_path)

    if not args.input:
        parser.error("Provide --config FILE or --input PATH")

    c = int(args.chunk)
    return VolumeProcessDTO(
        input_path=args.input,
        loader_type=args.loader,
        load_strategy=args.load_strategy,
        output_dir=args.output,
        threshold=args.threshold,
        auto_threshold=args.auto_threshold,
        threshold_algorithm=args.threshold_algorithm,
        export_formats=tuple(args.formats),
        chunk_shape=(c, c, c),
        halo_voxels=args.halo,
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    dto = _resolve_dto(args, parser)

    if args.dry_run:
        import json

        print("Resolved VolumeProcessDTO:")
        print(json.dumps(dto.to_dict(), indent=2))
        return 0

    print("=" * 60)
    print("4-DCT Porous Media - Headless Batch Processor")
    print("=" * 60)

    try:
        run_batch(dto)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        return 1
    except Exception as exc:
        import traceback

        print(f"\nPipeline failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
