from optirank.config import CFG, DEFAULT_DEVICE_ID


def test_config_split_order():
    assert CFG.train_end_day < CFG.val_end_day
    assert 0.0 < CFG.neg_downsample_rate <= 1.0
    assert DEFAULT_DEVICE_ID == "a99f214a"
