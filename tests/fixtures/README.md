# Test Fixtures

This directory contains test data and fixtures used by the test suite.

## Usage

```python
from tests.utils import TestData

# Get path to a fixture file
data_path = TestData.get_sample_data('sample_tensor.npy')
```

## Available Fixtures

- `sample_tensor.npy`: Sample tensor data for testing
- `test_config.json`: Test configuration data
- `benchmark_data.json`: Performance benchmark reference data

## Adding New Fixtures

1. Add your test data file to this directory
2. Update this README with a description
3. Use `TestData.get_sample_data()` to access in tests
