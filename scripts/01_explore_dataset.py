import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_PATH    = Path(__file__).resolve()
PROJECT_ROOT   = SCRIPT_PATH.parent.parent
DATA_DIR       = PROJECT_ROOT / "data" / "raw"
OUTPUT_TABLES  = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)

def discover_npy_files(data_dir: Path) -> dict[str, list[Path]]:
    """Load and group .npy files by column prefix."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_files = sorted(data_dir.glob("*.npy"))
    if len(all_files) == 0:
        raise FileNotFoundError(f"No .npy files in {data_dir}")

    # Group by column name (e.g., samples_000, samples_001 → samples)
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in all_files:
        stem = path.stem
        last_underscore = stem.rfind("_")
        column = stem[:last_underscore]
        segment = stem[last_underscore + 1:]

        if segment.isdigit():
            groups[column].append(path)
        else:
            groups[stem].append(path)

    for column in groups:
        groups[column].sort()

    return dict(groups)

def inspect_column(files: list[Path]) -> dict:
    """Inspect file shape and size without loading everything into memory."""
    sample = np.load(files[0], mmap_mode="r")
    total_size_bytes = sum(f.stat().st_size for f in files)

    return {
        "n_segments":     len(files),
        "per_segment_shape": list(sample.shape),
        "dtype":          str(sample.dtype),
        "total_size_mb":  round(total_size_bytes / (1024**2), 1),
    }

def count_total_messages(samples_files: list[Path]) -> int:
    """Sum the number of messages across all segment files."""
    total = 0
    # Each file's first dimension is number of messages
    for f in samples_files:
        arr = np.load(f, mmap_mode="r")
        total += arr.shape[0]
    return total

def count_satellites(ra_sat_files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Count messages per satellite and return sorted by frequency."""
    # Merge all satellite IDs from segments
    all_ids = np.concatenate([np.load(f) for f in ra_sat_files])
    unique_ids, counts = np.unique(all_ids, return_counts=True)
    # Sort descending by count
    order = np.argsort(counts)[::-1]
    return unique_ids[order], counts[order]

def save_satellite_counts_csv(satellite_ids: np.ndarray,
                counts: np.ndarray,
                output_path: Path) -> None:
    """Save per-satellite message counts as a simple two-column CSV."""
    with open(output_path, "w") as f:
        f.write("satellite_id,message_count\n")
        for sat_id, n_msgs in zip(satellite_ids, counts):
            f.write(f"{int(sat_id)},{int(n_msgs)}\n")

def save_summary_json(summary: dict, output_path: Path) -> None:
    """Serialize results to JSON, converting NumPy types to native Python types."""
    def to_jsonable(obj):
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_jsonable(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

def save_dataset_description_md(summary: dict, output_path: Path) -> None:
    """Generate a human-readable markdown report of the dataset."""
    # Extract key stats for readability
    samples_info = summary["columns"].get("samples", {})
    samples_shape = samples_info.get("per_segment_shape", ["?", "?", "?"])
    total_msgs = summary["total_messages"]
    n_sats = summary["n_unique_satellites"]

    # Build markdown report
    md = f"""# Iridium Dataset Description

## Dataset overview

- **Data directory:** `{summary["data_directory"]}`
- **Total number of messages:** {total_msgs:,}
- **Number of unique satellites:** {n_sats}
- **Number of metadata columns:** {len(summary["columns"])}

## Dataset columns

| Column | Number of segments | Shape per segment | Data type | Total size (MB) |
|---|---:|---|---|---:|
"""
    for column in sorted(summary["columns"].keys()):
        info = summary["columns"][column]

        md += (
            f"| `{column}` "
            f"| {info['n_segments']} "
            f"| {info['per_segment_shape']} "
            f"| {info['dtype']} "
            f"| {info['total_size_mb']} |\n"
        )

    md += f"""

## Top 10 most-sampled satellites

| Rank | Satellite ID | Message count |
|---:|---:|---:|
"""
    for i, (sat_id, count) in enumerate(
        summary["top_satellites"][:10],
        start=1
    ):
        md += f"| {i} | {sat_id} | {count:,} |\n"

    md += f"""

## Satellite distribution

- **Mean messages per satellite:** {total_msgs / n_sats:.0f}
- **Maximum messages for one satellite:** {summary['max_messages_per_sat']:,}
- **Minimum messages for one satellite:** {summary['min_messages_per_sat']:,}

## One IQ sample

A single IQ sample represents one Iridium Ring Alert message header.

Each sample has shape
`{samples_shape[1:] if len(samples_shape) > 1 else "see table above"}`
and dtype `{samples_info.get('dtype', '?')}`.

The shape `[11000, 2]` means that each message contains:

- 11,000 time-domain sample points
- two values per point: the in-phase component `I` and quadrature component `Q`

## Dataset size

- **Number of messages:** {total_msgs:,}
- **Number of unique satellites:** {n_sats}
- **Total extracted disk size:** approximately
  {sum(c['total_size_mb'] for c in summary['columns'].values()):.0f} MB
"""
    with open(output_path, "w") as f:
        f.write(md)

def main() -> None:
    print("=" * 70)
    print("Iridium Dataset Exploration")
    print("=" * 70)
    print(f"Project root:   {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print()

    # Load and organize files
    print("Discovering .npy files...")
    column_files = discover_npy_files(DATA_DIR)
    n_files_total = sum(len(v) for v in column_files.values())
    print(f"  Found {n_files_total} files across {len(column_files)} columns")
    print()

    # Analyze each column's structure
    print("Inspecting column shapes:")
    columns_summary = {}
    for column in sorted(column_files.keys()):
        info = inspect_column(column_files[column])
        columns_summary[column] = info
        print(f"  {column:<20s} segments={info['n_segments']:<3d} "
              f"shape={str(info['per_segment_shape']):<22s} "
              f"dtype={info['dtype']:<12s} "
              f"size={info['total_size_mb']:>6.1f} MB")
    print()

    # Count messages and satellites
    if "samples" not in column_files:
        raise KeyError("No 'samples' column found — cannot count messages.")
    print("Counting total messages across all 'samples' segments...")
    total_msgs = count_total_messages(column_files["samples"])
    print(f"  Total: {total_msgs:,} messages")
    print()

    if "ra_sat" not in column_files:
        raise KeyError("No 'ra_sat' column found — cannot count satellites.")
    print("Counting unique satellites...")
    sat_ids, sat_counts = count_satellites(column_files["ra_sat"])
    print(f"  Unique satellites: {len(sat_ids)}")
    print(f"  Most-sampled:      {int(sat_counts.max()):,} messages")
    print(f"  Least-sampled:     {int(sat_counts.min()):,} messages")
    print()
    print("  Top 10 most-sampled satellites:")
    for rank, (sat_id, n) in enumerate(zip(sat_ids[:10], sat_counts[:10]), 1):
        print(f"    {rank:>2d}.  Satellite {int(sat_id):>4d}  —  {int(n):,} messages")
    print()

    # Compile results and save
    summary = {
        "data_directory":          str(DATA_DIR),
        "total_messages":          int(total_msgs),
        "n_unique_satellites":     int(len(sat_ids)),
        "max_messages_per_sat":    int(sat_counts.max()),
        "min_messages_per_sat":    int(sat_counts.min()),
        "top_satellites":          [
            (int(sid), int(n)) for sid, n in zip(sat_ids[:20], sat_counts[:20])
        ],
        "columns":                 columns_summary,
    }

    csv_path  = OUTPUT_TABLES  / "satellite_counts.csv"
    json_path = OUTPUT_REPORTS / "dataset_summary.json"
    md_path   = OUTPUT_REPORTS / "dataset_description.md"

    # Write outputs in multiple formats
    save_satellite_counts_csv(sat_ids, sat_counts, csv_path)
    save_summary_json(summary, json_path)
    save_dataset_description_md(summary, md_path)

    print("=" * 70)
    print("Outputs written:")
    print(f"  {csv_path.relative_to(PROJECT_ROOT)}  (per-satellite message counts)")
    print(f"  {json_path.relative_to(PROJECT_ROOT)}  (machine-readable summary)")
    print(f"  {md_path.relative_to(PROJECT_ROOT)}  (dissertation-ready description)")
    print("=" * 70)

if __name__ == "__main__":
    main()
