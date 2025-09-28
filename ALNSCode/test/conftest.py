import pytest
from ALNSCode.alns_config import default_config as ALNSConfig


@pytest.fixture
def dataset_name():
    """Provide a default dataset name for tests that require it."""
    return f"dataset_{ALNSConfig.DATASET_IDX}"
