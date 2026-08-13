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


# ─── STEP 1: DISCOVER FILES ─────────────────────────────────
def discover_npy_files(data_dir: Path) -> dict[str, list[Path]]:

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_files = sorted(data_dir.glob("*.npy"))
    if len(all_files) == 0:
        raise FileNotFoundError(f"No .npy files in {data_dir}")

    # Group files by column prefix
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in all_files:
        stem = path.stem                  # 'ra_sat_000'
        last_underscore = stem.rfind("_") # position of the last '_'

        # Split into (column_name, segment_number)
        # e.g. 'ra_sat_000' -> ('ra_sat', '000')
        column = stem[:last_underscore]
        segment = stem[last_underscore + 1:]

        # Only treat as a numbered segment if the suffix is digits
        if segment.isdigit():
            groups[column].append(path)
        else:
            # Fallback: file doesn't follow the naming convention
            groups[stem].append(path)

    # Sort each column's files by segment number for predictable iteration
    for column in groups:
        groups[column].sort()

    return dict(groups)


# ─── STEP 2: INSPECT COLUMN SHAPES ───────────────────────────────────────────
def inspect_column(files: list[Path]) -> dict:
    # Load first file just to inspect shape (mmap_mode avoids reading into RAM)
    sample = np.load(files[0], mmap_mode="r")
    total_size_bytes = sum(f.stat().st_size for f in files)

    return {
        "n_segments":     len(files),
        "per_segment_shape": list(sample.shape),
        "dtype":          str(sample.dtype),
        "total_size_mb":  round(total_size_bytes / (1024**2), 1),
    }


# ─── STEP 3: COUNT TOTAL MESSAGES ────────────────────────────────────────────
def count_total_messages(samples_files: list[Path]) -> int:
    total = 0
    for f in samples_files:
        arr = np.load(f, mmap_mode="r")
        total += arr.shape[0]
    return total


# ─── STEP 4: COUNT SATELLITES ────────────────────────────────────────────────
def count_satellites(ra_sat_files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    # Concatenate all segments into one flat array of satellite IDs
    all_ids = np.concatenate([np.load(f) for f in ra_sat_files])

    # np.unique returns (unique_values, counts_per_value)
    unique_ids, counts = np.unique(all_ids, return_counts=True)

    # Sort by count descending (argsort gives ascending, [::-1] reverses it)
    order = np.argsort(counts)[::-1]
    return unique_ids[order], counts[order]


# ─── STEP 5: SAVE THE RESULTS ────────────────────────────────────────────────
def save_satellite_counts_csv(satellite_ids: np.ndarray,
                counts: np.ndarray,
                output_path: Path) -> None:
    """Save per-satellite message counts as a simple two-column CSV."""
    with open(output_path, "w") as f:
        f.write("satellite_id,message_count\n")
        for sat_id, n_msgs in zip(satellite_ids, counts):
            f.write(f"{int(sat_id)},{int(n_msgs)}\n")


def save_summary_json(summary: dict, output_path: Path) -> None:
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

    # Pull useful values out for readability
    samples_info = summary["columns"].get("samples", {})
    samples_shape = samples_info.get("per_segment_shape", ["?", "?", "?"])
    total_msgs = summary["total_messages"]
    n_sats = summary["n_unique_satellites"]

    # Start the markdown document
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

    # Add one table row per column
    for column in sorted(summary["columns"].keys()):
        info = summary["columns"][column]

        md += (
            f"| `{column}` "
            f"| {info['n_segments']} "
            f"| {info['per_segment_shape']} "
            f"| {info['dtype']} "
            f"| {info['total_size_mb']} |\n"
        )

    # Add satellite-count table heading
    md += f"""

## Top 10 most-sampled satellites

| Rank | Satellite ID | Message count |
|---:|---:|---:|
"""

    # Add one row for each of the top 10 satellites
    for i, (sat_id, count) in enumerate(
        summary["top_satellites"][:10],
        start=1
    ):
        md += f"| {i} | {sat_id} | {count:,} |\n"

    # Add final dataset statistics
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

    # Save the markdown file
    with open(output_path, "w") as f:
        f.write(md)


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("Iridium Dataset Exploration")
    print("=" * 70)
    print(f"Project root:   {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print()

    # ─── 1. Discover what's in the data folder ───
    print("Discovering .npy files...")
    column_files = discover_npy_files(DATA_DIR)
    n_files_total = sum(len(v) for v in column_files.values())
    print(f"  Found {n_files_total} files across {len(column_files)} columns")
    print()

    # ─── 2. Inspect each column's shape ───
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

    # ─── 3. Count total messages ───
    if "samples" not in column_files:
        raise KeyError("No 'samples' column found — cannot count messages.")
    print("Counting total messages across all 'samples' segments...")
    total_msgs = count_total_messages(column_files["samples"])
    print(f"  Total: {total_msgs:,} messages")
    print()

    # ─── 4. Satellite-level counts ───
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

    # ─── 5. Assemble and save outputs ───
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
