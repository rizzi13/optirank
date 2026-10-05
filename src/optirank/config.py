"""Central configuration for OptiRank."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

# Default dummy device_id in Avazu (means no real device_id was available)
DEFAULT_DEVICE_ID = "a99f214a"
SEED = 42


@dataclass(frozen=True)
class SplitConfig:
    # Avazu hour format is YYMMDDHH (Oct 21 to Oct 30, 2014)
    train_end_day: int = 14102823   # Oct 21..28 -> Train
    val_end_day: int = 14102923     # Oct 29     -> Validation
    # Oct 30 (14103000..14103023)   -> Test
    neg_downsample_rate: float = 0.25  # Keep 100% clicks, 25% non-clicks in train


CFG = SplitConfig()
