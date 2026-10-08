"""Preferred entry point for Monolithic Service Orchestration (MSO) training.

The original trainer module remains available for old commands and checkpoint
compatibility; both entry points execute the same implementation.
"""

from train_monolithic import main


if __name__ == "__main__":
    main()
