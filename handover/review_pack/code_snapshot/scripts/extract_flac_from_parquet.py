import argparse
from pathlib import Path
import pyarrow.parquet as pq

def extract_flac(parquet_path: str, output_dir: str) -> int:
    p_path = Path(parquet_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(str(p_path))
    count = 0
    for rg_idx in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg_idx, columns=["audio"])
        col = tbl.column("audio")
        for item in col.to_pylist():
            f_bytes = item.get("bytes")
            f_name = Path(item.get("path") or "").name
            if not f_name:
                continue
            if not f_name.endswith(".flac"):
                f_name = f"{f_name}.flac"
            target_path = out_dir / f_name
            with open(target_path, "wb") as f:
                f.write(f_bytes)
            count += 1
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    extracted = extract_flac(args.parquet, args.output_dir)
    print(f"Extracted {extracted} files to {args.output_dir}")

