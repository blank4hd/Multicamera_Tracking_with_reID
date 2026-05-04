import argparse
from pathlib import Path

from huggingface_hub import HfApi, upload_folder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload the Hugging Face staging folder to a model repo.")
    parser.add_argument("--repo-id", default="blank4hd/mctrack-reid")
    parser.add_argument("--folder", default="huggingface")
    parser.add_argument("--commit-msg", default="Upload Re-ID model checkpoints and model card")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    api = HfApi()
    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"Folder {folder} does not exist. Run export_reid_for_hf.py first.")

    print(f"Uploading {folder}/ to {args.repo_id}...")
    print("Files to upload:")
    for f in folder.glob("**/*"):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(folder)} ({size_mb:.1f} MB)")
    print()

    result = upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_msg,
    )
    print(f"\nUpload complete: {result}")
    print(f"View at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()