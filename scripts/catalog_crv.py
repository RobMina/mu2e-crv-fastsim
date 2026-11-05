"""Read-only CosmicCRY dataset census; product/geometry support still requires an art inspection."""
import argparse
import json
from pathlib import Path

def census(client, datasets):
    records = []
    for did in datasets:
        info = client.get_dataset(did=did, exact_file_count=False)
        rows = list(client.get_dataset_files(did=did, with_metadata=False))
        sample = client.get_file(did=f"{rows[0]['namespace']}:{rows[0]['name']}", with_metadata=True) if rows else None
        records.append({"dataset": did, "dataset_info": info, "files": len(rows),
                        "cached_count_matches_enumeration": info.get("file_count") == len(rows),
                        "bytes": sum(row.get("size", 0) or 0 for row in rows),
                        "sample": sample, "file_names": [f"{row['namespace']}:{row['name']}" for row in rows]})
        print(f"{did}: {len(rows)} files", flush=True)
    return {"read_only": True, "product_support": "not verified by catalog census", "datasets": records}


def main():
    from metacat.webapi import MetaCatClient
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(path)
    result = census(MetaCatClient(), args.datasets)
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
