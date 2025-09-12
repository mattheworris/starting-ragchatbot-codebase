"""
Tests for AI Generator and tool calling functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator

class TestAIGeneratorInitialization:
    """Test AIGenerator initialization"""
    
    def test_init_with_valid_config(self, test_config):
        """Test AIGenerator initialization with valid configuration"""
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        assert generator.model == test_config.ANTHROPIC_MODEL
        assert generator.base_params["model"] == test_config.ANTHROPIC_MODEL
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
    
    def test_init_with_empty_api_key(self):
        """Test AIGenerator initialization with empty API key"""
        # This should not fail during init, but will fail during API calls
        generator = AIGenerator("", "claude-sonnet-4-20250514")
        assert generator.model == "claude-sonnet-4-20250514"
    
    def test_system_prompt_content(self):
        """Test that system prompt includes tool guidance"""
        assert "Content Search Tool" in AIGenerator.SYSTEM_PROMPT
        assert "Course Outline Tool" in AIGenerator.SYSTEM_PROMPT
        assert "Tool Usage Guidelines" in AIGenerator.SYSTEM_PROMPT

class TestAIGeneratorWithoutTools:
    """Test AI generation without tools"""
    
    @patch('anthropic.Anthropic')
    def test_generate_response_simple_query(self, mock_anthropic_class, mock_anthropic_client):
        """Test simple response generation without tools"""
        mock_anthropic_class.return_value = mock_anthropic_client
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response("What is Python?")
        
        assert response == "This is a test response from Claude."
        mock_anthropic_client.messages.create.assert_called_once()
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_history(self, mock_anthropic_class, mock_anthropic_client):
        """Test response generation with conversation history"""
        mock_anthropic_class.return_value = mock_anthropic_client
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response(
            "Follow up question",
            conversation_history="Previous: What is testing?\nAnswer: Testing is..."
        )
        
        # Should include history in system prompt
        call_args = mock_anthropic_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation" in system_content
        assert "Previous: What is testing?" in system_content

class TestAIGeneratorWithTools:
    """Test AI generation with tool usage"""
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools_no_usage(self, mock_anthropic_class, mock_anthropic_client, tool_manager_with_tools):
        """Test response when tools are available but not used"""
        mock_anthropic_class.return_value = mock_anthropic_client
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response(
            "What is general programming?",
            tools=tool_manager_with_tools.get_tool_definitions(),
            tool_manager=tool_manager_with_tools
        )
        
        # Should return direct response without tool use
        assert response == "This is a test response from Claude."
        
        # Should have tools in API call
        call_args = mock_anthropic_client.messages.create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tool_choice"] == {"type": "auto"}
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tool_usage(self, mock_anthropic_class, mock_anthropic_tool_response, tool_manager_with_tools):
        """Test response when tools are used"""
        mock_anthropic_class.return_value = mock_anthropic_tool_response
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response(
            "What are testing concepts in the course?",
            tools=tool_manager_with_tools.get_tool_definitions(),
            tool_manager=tool_manager_with_tools
        )
        
        # Should return final response after tool execution
        assert response == "Based on the search results, testing concepts include..."
        
        # Should have made two API calls (initial + final)
        assert mock_anthropic_tool_response.messages.create.call_count == 2

class TestAIGeneratorErrorHandling:
    """Test error handling scenarios"""
    
    @patch('anthropic.Anthropic')
    def test_api_key_error(self, mock_anthropic_class):
        """Test behavior with invalid API key"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("Invalid API key")
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("invalid-key", "claude-sonnet-4-20250514")
        
        with pytest.raises(Exception) as exc_info:
            generator.generate_response("Test query")
        
        assert "Invalid API key" in str(exc_info.value)
    
    @patch('anthropic.Anthropic')
    def test_network_error(self, mock_anthropic_class):
        """Test behavior with network errors"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("Network timeout")
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        
        with pytest.raises(Exception) as exc_info:
            generator.generate_response("Test query")
        
        assert "Network timeout" in str(exc_info.value)
    
    @patch('anthropic.Anthropic')
    def test_malformed_tool_response(self, mock_anthropic_class, tool_manager_with_tools):
        """Test handling of malformed tool responses"""
        mock_client = Mock()
        
        # Create malformed tool response
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "nonexistent_tool"  # Tool doesn't exist
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"invalid": "parameters"}
        
        mock_response = Mock()
        mock_response.content = [mock_tool_block]
        mock_response.stop_reason = "tool_use"
        
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        
        # Should handle tool execution error gracefully
        response = generator.generate_response(
            "Test query",
            tools=tool_manager_with_tools.get_tool_definitions(),
            tool_manager=tool_manager_with_tools
        )
        
        # Tool manager should return error message for nonexistent tool
        # This will depend on how tool_manager handles errors

class TestAIGeneratorToolExecution:
    """Test tool execution flow"""
    
    def test_handle_tool_execution_successful(self, tool_manager_with_tools):
        """Test successful tool execution handling"""
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        
        # Mock initial response with tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "testing"}
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]
        
        # Mock base parameters
        base_params = {
            "messages": [{"role": "user", "content": "What is testing?"}],
            "system": "Test system prompt"
        }
        
        with patch.object(generator, 'client') as mock_client:
            # Mock final response
            mock_final_content = Mock()
            mock_final_content.text = "Testing involves verification..."
            
            mock_final_response = Mock()
            mock_final_response.content = [mock_final_content]
            
            mock_client.messages.create.return_value = mock_final_response
            
            result = generator._handle_tool_execution(
                mock_initial_response,
                base_params,
                tool_manager_with_tools
            )
            
            assert result == "Testing involves verification..."
    
    def test_handle_tool_execution_with_multiple_tools(self, tool_manager_with_tools):
        """Test handling multiple tool calls in one response"""
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        
        # Mock response with multiple tool uses
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "search_course_content"
        mock_tool_block1.id = "tool_123"
        mock_tool_block1.input = {"query": "testing"}
        
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "get_course_outline"
        mock_tool_block2.id = "tool_456"
        mock_tool_block2.input = {"course_title": "Testing Course"}
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block1, mock_tool_block2]
        
        base_params = {
            "messages": [{"role": "user", "content": "Tell me about testing"}],
            "system": "Test system prompt"
        }
        
        with patch.object(generator, 'client') as mock_client:
            mock_final_content = Mock()
            mock_final_content.text = "Here's information about testing..."
            
            mock_final_response = Mock()
            mock_final_response.content = [mock_final_content]
            
            mock_client.messages.create.return_value = mock_final_response
            
            result = generator._handle_tool_execution(
                mock_initial_response,
                base_params,
                tool_manager_with_tools
            )
            
            assert result == "Here's information about testing..."
            
            # Should have called both tools
            # This would need verification through tool_manager mock

class TestAIGeneratorConfigurationIssues:
    """Test how configuration issues affect AI generation"""
    
    def test_missing_api_key_detection(self, test_config):
        """Test that missing API key is handled appropriately"""
        # Create generator with empty API key
        generator = AIGenerator("", test_config.ANTHROPIC_MODEL)
        
        # The generator should be created, but API calls will fail
        assert generator.client is not None
        
        # API calls should fail with authentication error
        with patch.object(generator, 'client') as mock_client:
            mock_client.messages.create.side_effect = Exception("Authentication failed")
            
            with pytest.raises(Exception) as exc_info:
                generator.generate_response("Test query")
            
            assert "Authentication failed" in str(exc_info.value)

if __name__ == "__main__":
    # Allow running tests directly for debugging
    pytest.main([__file__, "-v"])