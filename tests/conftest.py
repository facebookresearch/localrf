import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
import json


@pytest.fixture
def temp_dir():
    """Create a temporary directory that's automatically cleaned up after tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_file():
    """Create a temporary file that's automatically cleaned up after tests."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    yield Path(path)
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def mock_config():
    """Provide a mock configuration object for testing."""
    config = MagicMock()
    config.debug = False
    config.verbose = True
    config.batch_size = 32
    config.learning_rate = 0.001
    config.epochs = 100
    config.device = "cpu"
    return config


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        "id": 1,
        "name": "test_sample",
        "values": [1.0, 2.0, 3.0, 4.0, 5.0],
        "metadata": {
            "created": "2024-01-01",
            "author": "test_user",
            "version": "1.0.0"
        }
    }


@pytest.fixture
def mock_model():
    """Provide a mock model for testing."""
    model = Mock()
    model.train = Mock(return_value=None)
    model.eval = Mock(return_value=None)
    model.forward = Mock(return_value={"output": [0.1, 0.2, 0.7]})
    model.parameters = Mock(return_value=[])
    return model


@pytest.fixture
def mock_dataset():
    """Provide a mock dataset for testing."""
    dataset = Mock()
    dataset.__len__ = Mock(return_value=100)
    dataset.__getitem__ = Mock(return_value=({"input": [1, 2, 3]}, {"target": [0, 1, 0]}))
    return dataset


@pytest.fixture
def json_file(temp_dir, sample_data):
    """Create a temporary JSON file with sample data."""
    file_path = temp_dir / "test_data.json"
    with open(file_path, 'w') as f:
        json.dump(sample_data, f)
    yield file_path


@pytest.fixture
def config_file(temp_dir, sample_data):
    """Create a temporary config file with sample data."""
    file_path = temp_dir / "test_config.json"
    with open(file_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    yield file_path


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    return logger


@pytest.fixture
def mock_file_system(mocker):
    """Mock file system operations."""
    mock_open = mocker.mock_open(read_data="test data")
    mocker.patch("builtins.open", mock_open)
    
    mock_exists = mocker.patch("os.path.exists")
    mock_exists.return_value = True
    
    mock_makedirs = mocker.patch("os.makedirs")
    
    return {
        "open": mock_open,
        "exists": mock_exists,
        "makedirs": mock_makedirs
    }


@pytest.fixture
def mock_time(mocker):
    """Mock time-related functions."""
    mock_time = mocker.patch("time.time")
    mock_time.return_value = 1234567890.0
    
    mock_sleep = mocker.patch("time.sleep")
    
    return {
        "time": mock_time,
        "sleep": mock_sleep
    }


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before and after each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def capture_logs(caplog):
    """Capture log messages for testing."""
    with caplog.at_level("DEBUG"):
        yield caplog


@pytest.fixture
def mock_requests(mocker):
    """Mock HTTP requests."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    mock_response.text = "Success"
    
    mock_get = mocker.patch("requests.get", return_value=mock_response)
    mock_post = mocker.patch("requests.post", return_value=mock_response)
    
    return {
        "get": mock_get,
        "post": mock_post,
        "response": mock_response
    }


@pytest.fixture
def benchmark_timer():
    """Simple timer for benchmarking tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.time()
            
        def stop(self):
            self.end_time = time.time()
            
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


@pytest.fixture
def clean_imports():
    """Clean up imports to ensure test isolation."""
    import sys
    modules_before = set(sys.modules.keys())
    yield
    modules_after = set(sys.modules.keys())
    for module in modules_after - modules_before:
        if module.startswith("localTensoRF"):
            del sys.modules[module]