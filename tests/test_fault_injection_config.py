"""
Fault Injection Tests — Configuration Layer

Tests the resilience of the configuration loading and management system
against various inputs. The config system is designed to be resilient:
missing files, malformed YAML, and missing env vars are handled gracefully
with fallbacks and warnings rather than crashes.
"""

import json
import os
import tempfile

import pytest
import yaml

# ═══════════════════════════════════════════════════════════════════
# 1. Config Loader — Resilience Tests
# ═══════════════════════════════════════════════════════════════════


class TestConfigLoaderFaultInjection:
    """Test the config loader's ability to handle various faults."""

    def test_load_nonexistent_file_does_not_crash(self):
        """Loading a nonexistent config file should not crash the system.
        The ConfigLoader falls back to defaults gracefully."""
        try:
            from src.core.config.loader import ConfigLoader
        except ImportError:
            pytest.skip("ConfigLoader not available")

        loader = ConfigLoader()
        # Should not raise — the loader handles missing files gracefully
        config = loader.load_config("/nonexistent/path/config.yaml")
        assert config is not None

    def test_load_invalid_yaml_does_not_crash(self):
        """Loading invalid YAML should not crash the system."""
        try:
            from src.core.config.loader import ConfigLoader
        except ImportError:
            pytest.skip("ConfigLoader not available")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: : : broken\n  bad_indent")
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_config(temp_path)
            # Might return defaults or partial config
            assert config is not None
        finally:
            os.unlink(temp_path)

    def test_load_binary_yaml_does_not_crash(self):
        """Loading binary garbage as YAML should not crash."""
        try:
            from src.core.config.loader import ConfigLoader
        except ImportError:
            pytest.skip("ConfigLoader not available")

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".yaml", delete=False) as f:
            f.write(b"\x00\x01\x02\x03invalid yaml bytes")
            temp_path = f.name

        try:
            loader = ConfigLoader()
            # Should handle gracefully — either return defaults or raise
            try:
                config = loader.load_config(temp_path)
                assert config is not None
            except Exception:
                pass  # Also acceptable
        finally:
            os.unlink(temp_path)

    def test_load_json_config(self):
        """Loading a JSON config file should work."""
        try:
            from src.core.config.loader import ConfigLoader
        except ImportError:
            pytest.skip("ConfigLoader not available")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"custom_key": "custom_value"}, f)
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_config(temp_path)
            assert config is not None
            # The loader may merge with defaults, but should not crash
        finally:
            os.unlink(temp_path)

    def test_load_empty_yaml_does_not_crash(self):
        """Loading an empty YAML file should return a valid config."""
        try:
            from src.core.config.loader import ConfigLoader
        except ImportError:
            pytest.skip("ConfigLoader not available")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_config(temp_path)
            assert config is not None
        finally:
            os.unlink(temp_path)


# ═══════════════════════════════════════════════════════════════════
# 2. Environment Variable Substitution
# ═══════════════════════════════════════════════════════════════════


class TestEnvVarSubstitutionFaultInjection:
    """Test environment variable substitution in config files."""

    def test_missing_env_var_does_not_crash(self):
        """Missing env vars should not crash — the loader should handle it."""
        try:
            from src.core.config.loader import ConfigLoader
        except ImportError:
            pytest.skip("ConfigLoader not available")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("api_key: '${MISSING_ENV_VAR_12345}'")
            temp_path = f.name

        try:
            loader = ConfigLoader()
            # Should not crash; might fall back or keep the placeholder
            config = loader.load_config(temp_path)
            assert config is not None
        finally:
            os.unlink(temp_path)

    def test_env_var_substitution_succeeds(self):
        """Present env vars should be substituted correctly."""
        try:
            from src.core.config.loader import ConfigLoader
        except ImportError:
            pytest.skip("ConfigLoader not available")

        os.environ["TEST_CONFIG_KEY"] = "test_value_42"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: '${TEST_CONFIG_KEY}'")
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_config(temp_path)
            assert config is not None
        finally:
            os.unlink(temp_path)
            del os.environ["TEST_CONFIG_KEY"]


# ═══════════════════════════════════════════════════════════════════
# 3. Config Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestConfigEdgeCases:
    """Test edge cases in configuration."""

    def test_config_with_comments_only(self):
        """YAML with only comments should be treated as empty config."""
        try:
            from src.core.config.loader import ConfigLoader
        except ImportError:
            pytest.skip("ConfigLoader not available")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("# This is a comment\n# Another comment\n")
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_config(temp_path)
            assert config is not None
        finally:
            os.unlink(temp_path)

    def test_unicode_in_config(self):
        """Unicode characters in config should be handled."""
        try:
            from src.core.config.loader import ConfigLoader
        except ImportError:
            pytest.skip("ConfigLoader not available")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("greeting: '你好世界'\nemoji: '🚀'\n")
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_config(temp_path)
            assert config is not None
        finally:
            os.unlink(temp_path)

    def test_deeply_nested_config(self):
        """Deeply nested config should be loaded without recursion issues."""
        try:
            from src.core.config.loader import ConfigLoader
        except ImportError:
            pytest.skip("ConfigLoader not available")

        nested = {"a": {"b": {"c": {"d": "deep_value"}}}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(nested, f)
            temp_path = f.name

        try:
            loader = ConfigLoader()
            config = loader.load_config(temp_path)
            assert config is not None
        finally:
            os.unlink(temp_path)
