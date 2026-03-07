from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()

DATA_DIR = ROOT_DIR / "dataset" / "raw"
TRAIN_PARQUET = DATA_DIR / "train.parquet"
TEST_PARQUET = DATA_DIR / "test.parquet"
