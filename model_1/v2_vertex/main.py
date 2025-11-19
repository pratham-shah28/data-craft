"""Backward-compatible entrypoint that proxies to inference.main()."""
from inference import main


if __name__ == "__main__":
    main()
