import yaml
import tempfile
import os
from src.core.config_loader import ConfigLoader
from src.core.exceptions import ConfigurationError


def test_yaml():
    invalid_yaml = """
    global:
      default_provider: test
    providers:
      test:
        type: integration_mock
        enabled: true
        invalid: [unclosed list
    """
    try:
        yaml.safe_load(invalid_yaml)
        print("YAML load success (Unexpected)")
    except yaml.YAMLError as e:
        print(f"YAML load failed as expected: {e}")


def test_loader_invalid():
    invalid_yaml = """
    global:
      default_provider: test
    providers:
      test:
        type: integration_mock
        enabled: true
        invalid: [unclosed list
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(invalid_yaml)
        temp_path = f.name

    try:
        loader = ConfigLoader(temp_path)
        loader.load_config()
        print("Loader success (Unexpected)")
    except ConfigurationError as e:
        print(f"Loader failed as expected: {e}")
    except Exception as e:
        print(f"Loader failed with unexpected error: {type(e)} {e}")
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    test_yaml()
    test_loader_invalid()
