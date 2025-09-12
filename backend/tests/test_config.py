"""
Tests for configuration validation and environment setup
"""
import pytest
import os
from unittest.mock import patch, Mock
import sys

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config, config

class TestConfigValidation:
    """Test configuration values and validation"""
    
    def test_max_results_not_zero(self):
        """Test that MAX_RESULTS is not zero (critical bug)"""
        # This test will FAIL with current configuration
        assert config.MAX_RESULTS > 0, "MAX_RESULTS should be greater than 0 to return search results"
    
    def test_max_results_reasonable_value(self):
        """Test that MAX_RESULTS is a reasonable value"""
        assert config.MAX_RESULTS <= 20, "MAX_RESULTS should not be excessively high"
        assert isinstance(config.MAX_RESULTS, int), "MAX_RESULTS should be an integer"
    
    def test_anthropic_api_key_loaded(self):
        """Test that ANTHROPIC_API_KEY is loaded from environment"""
        # This may fail if .env is not properly set up
        assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY should not be empty"
        assert config.ANTHROPIC_API_KEY != "your-anthropic-api-key-here", "ANTHROPIC_API_KEY should not be placeholder"
    
    def test_model_configuration(self):
        """Test that model configuration is valid"""
        assert config.ANTHROPIC_MODEL, "ANTHROPIC_MODEL should not be empty"
        assert "claude" in config.ANTHROPIC_MODEL.lower(), "Should use Claude model"
    
    def test_embedding_model_configuration(self):
        """Test that embedding model is configured"""
        assert config.EMBEDDING_MODEL, "EMBEDDING_MODEL should not be empty"
        assert isinstance(config.EMBEDDING_MODEL, str), "EMBEDDING_MODEL should be a string"
    
    def test_chunk_settings(self):
        """Test that chunking settings are reasonable"""
        assert config.CHUNK_SIZE > 0, "CHUNK_SIZE should be positive"
        assert config.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP should be non-negative"
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE, "CHUNK_OVERLAP should be less than CHUNK_SIZE"
    
    def test_history_settings(self):
        """Test that history settings are valid"""
        assert config.MAX_HISTORY >= 0, "MAX_HISTORY should be non-negative"
        assert isinstance(config.MAX_HISTORY, int), "MAX_HISTORY should be an integer"
    
    def test_database_path(self):
        """Test that database path is configured"""
        assert config.CHROMA_PATH, "CHROMA_PATH should not be empty"
        assert isinstance(config.CHROMA_PATH, str), "CHROMA_PATH should be a string"

class TestEnvironmentSetup:
    """Test environment variable handling"""
    
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_handling(self):
        """Test behavior when API key is missing"""
        # Reload config without API key
        new_config = Config()
        assert new_config.ANTHROPIC_API_KEY == "", "Should default to empty string when API key missing"
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key-123'})
    def test_api_key_from_environment(self):
        """Test that API key is loaded from environment"""
        new_config = Config()
        assert new_config.ANTHROPIC_API_KEY == 'test-key-123', "Should load API key from environment"
    
    def test_config_dataclass_structure(self):
        """Test that config is properly structured as dataclass"""
        assert hasattr(config, 'ANTHROPIC_API_KEY'), "Config should have ANTHROPIC_API_KEY"
        assert hasattr(config, 'MAX_RESULTS'), "Config should have MAX_RESULTS"
        assert hasattr(config, 'CHROMA_PATH'), "Config should have CHROMA_PATH"

class TestConfigBugIdentification:
    """Specific tests to identify the configuration bugs causing query failures"""
    
    def test_max_results_zero_bug(self):
        """Test specifically for the MAX_RESULTS = 0 bug"""
        # This test documents the exact bug we found
        if config.MAX_RESULTS == 0:
            pytest.fail(f"CRITICAL BUG: MAX_RESULTS is set to {config.MAX_RESULTS}, which will cause all searches to return 0 results")
    
    def test_search_would_return_results(self, test_config):
        """Test that a proper configuration would allow search results"""
        assert test_config.MAX_RESULTS > 0, "Test config should have MAX_RESULTS > 0"
        assert test_config.MAX_RESULTS >= 1, "Should allow at least 1 search result"
    
    def test_configuration_comparison(self, test_config):
        """Compare current config with test config to highlight issues"""
        issues = []
        
        if config.MAX_RESULTS <= 0:
            issues.append(f"MAX_RESULTS is {config.MAX_RESULTS}, should be > 0")
        
        if not config.ANTHROPIC_API_KEY:
            issues.append("ANTHROPIC_API_KEY is empty")
        
        if config.ANTHROPIC_API_KEY == "your-anthropic-api-key-here":
            issues.append("ANTHROPIC_API_KEY is still set to placeholder value")
        
        if issues:
            pytest.fail(f"Configuration issues found: {'; '.join(issues)}")

class TestConfigImpactOnSystem:
    """Test how configuration affects system behavior"""
    
    def test_max_results_impact_on_vector_store(self, test_config):
        """Test how MAX_RESULTS affects vector store initialization"""
        # This simulates what happens in VectorStore.__init__
        max_results = config.MAX_RESULTS
        if max_results <= 0:
            pytest.fail(f"VectorStore will be initialized with max_results={max_results}, causing search failures")
    
    def test_api_key_impact_on_ai_generator(self):
        """Test how missing API key affects AI generator"""
        if not config.ANTHROPIC_API_KEY:
            pytest.fail("AIGenerator will fail to initialize or make API calls without ANTHROPIC_API_KEY")
    
    def test_chroma_path_accessibility(self):
        """Test that ChromaDB path is accessible"""
        # Check if parent directory exists or can be created
        import os
        parent_dir = os.path.dirname(config.CHROMA_PATH)
        if parent_dir and not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except Exception as e:
                pytest.fail(f"Cannot create ChromaDB directory {parent_dir}: {e}")

if __name__ == "__main__":
    # Allow running tests directly for debugging
    pytest.main([__file__, "-v"])