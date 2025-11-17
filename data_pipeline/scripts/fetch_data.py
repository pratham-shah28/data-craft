import os
import argparse
import requests
from pathlib import Path

DEFAULT_OWNER = "Azure-Samples"
DEFAULT_REPO = "azure-ai-document-processing-samples"
DEFAULT_PATH = "samples/assets"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "data" / "unstructured"

def download_content(url, local_path):
    os.makedirs(local_path, exist_ok=True)
    response = requests.get(url).json()

    for item in response:
        item_path = os.path.join(local_path, item["name"])

        if item["type"] == "file":
            print(f"Downloading: {item['name']}")
            file_data = requests.get(item["download_url"]).content
            with open(item_path, "wb") as f:
                f.write(file_data)

        elif item["type"] == "dir":
            download_content(item["url"], item_path)


def main():
    parser = argparse.ArgumentParser(description="Fetch data from a GitHub repository folder")

    parser.add_argument("--owner", default=DEFAULT_OWNER,
                        help=f"GitHub repository owner, default: {DEFAULT_OWNER}")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"GitHub repository name, default: {DEFAULT_REPO}")
    parser.add_argument("--path", default=DEFAULT_PATH,
                        help=f"Folder path inside the repository, default: {DEFAULT_PATH}")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output directory to store fetched data, default: {DEFAULT_OUTPUT}")

    args = parser.parse_args()

    api_url = f"https://api.github.com/repos/{args.owner}/{args.repo}/contents/{args.path}"

    print(f"Fetching from {api_url}")
    download_content(api_url, args.output)
    print("Data sync complete!")


if __name__ == "__main__":
    main()
