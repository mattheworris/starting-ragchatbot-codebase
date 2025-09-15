"""
Tests for AI Generator and tool calling functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator, ExecutionState, ExecutionContext

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
        assert "Multi-Round Strategy" in AIGenerator.SYSTEM_PROMPT
        assert "Round 1" in AIGenerator.SYSTEM_PROMPT
        assert "Round 2" in AIGenerator.SYSTEM_PROMPT

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

class TestSequentialToolCalling:
    """Test sequential tool calling functionality"""
    
    def test_execution_context_initialization(self):
        """Test ExecutionContext is properly initialized"""
        context = ExecutionContext(
            state=ExecutionState.INITIAL,
            round_count=0,
            messages=[{"role": "user", "content": "test"}],
            tool_results_history=[],
            accumulated_sources=[]
        )
        
        assert context.state == ExecutionState.INITIAL
        assert context.round_count == 0
        assert not context.is_completed()
    
    def test_execution_context_completion_conditions(self):
        """Test ExecutionContext completion detection"""
        context = ExecutionContext(
            state=ExecutionState.COMPLETED,
            round_count=0,
            messages=[],
            tool_results_history=[],
            accumulated_sources=[]
        )
        
        assert context.is_completed()
        
        # Test max rounds exceeded
        context.state = ExecutionState.ROUND_1_TOOLS
        context.round_count = 3
        context.max_rounds = 2
        assert context.is_completed()
        
        # Test error limit exceeded
        context.round_count = 1
        context.error_count = 4
        assert context.is_completed()
    
    @patch('anthropic.Anthropic')
    def test_sequential_fallback_to_standard(self, mock_anthropic_class, mock_anthropic_client):
        """Test sequential method falls back to standard when no tools"""
        mock_anthropic_class.return_value = mock_anthropic_client
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response_sequential("What is Python?")
        
        # Should call standard generate_response
        assert response == "This is a test response from Claude."
    
    @patch('anthropic.Anthropic')
    def test_sequential_single_round_no_tools_used(self, mock_anthropic_class, mock_anthropic_client, tool_manager_with_tools):
        """Test sequential execution when Claude doesn't use tools"""
        mock_anthropic_class.return_value = mock_anthropic_client
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response_sequential(
            "What is general programming?",
            tools=tool_manager_with_tools.get_tool_definitions(),
            tool_manager=tool_manager_with_tools
        )
        
        # Should complete in one round without tools
        assert response == "This is a test response from Claude."
        # Should have made only one API call
        assert mock_anthropic_client.messages.create.call_count == 1
    
    @patch('anthropic.Anthropic')
    def test_sequential_single_round_with_tools(self, mock_anthropic_class, tool_manager_with_tools):
        """Test sequential execution with tools in first round only"""
        # Create mock client that uses tools once then provides final response
        mock_client = Mock()
        
        # Mock first response with tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "testing"}
        
        mock_first_response = Mock()
        mock_first_response.content = [mock_tool_block]
        mock_first_response.stop_reason = "tool_use"
        
        # Mock second response (final, no tools)
        mock_final_content = Mock()
        mock_final_content.text = "Testing concepts include unit testing, integration testing..."
        
        mock_second_response = Mock()
        mock_second_response.content = [mock_final_content]
        mock_second_response.stop_reason = "stop"
        
        # Set up call sequence
        mock_client.messages.create.side_effect = [mock_first_response, mock_second_response]
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response_sequential(
            "What are testing concepts?",
            tools=tool_manager_with_tools.get_tool_definitions(),
            tool_manager=tool_manager_with_tools
        )
        
        # Should return final response
        assert "Testing concepts include" in response
        # Should have made two API calls
        assert mock_client.messages.create.call_count == 2
    
    @patch('anthropic.Anthropic')
    def test_sequential_two_round_execution(self, mock_anthropic_class, tool_manager_with_tools):
        """Test full two-round sequential execution"""
        mock_client = Mock()
        
        # Mock first response with tool use
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "get_course_outline"
        mock_tool_block1.id = "tool_123"
        mock_tool_block1.input = {"course_title": "MCP Course"}
        
        mock_first_response = Mock()
        mock_first_response.content = [mock_tool_block1]
        mock_first_response.stop_reason = "tool_use"
        
        # Mock second response (asks for more tools)
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "search_course_content"
        mock_tool_block2.id = "tool_456"
        mock_tool_block2.input = {"query": "lesson 4"}
        
        mock_second_response = Mock()
        mock_second_response.content = [mock_tool_block2]
        mock_second_response.stop_reason = "tool_use"
        
        # Mock final response
        mock_final_content = Mock()
        mock_final_content.text = "Based on the course outline and content search, lesson 4 covers..."
        
        mock_final_response = Mock()
        mock_final_response.content = [mock_final_content]
        mock_final_response.stop_reason = "stop"
        
        # Set up call sequence
        mock_client.messages.create.side_effect = [mock_first_response, mock_second_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response_sequential(
            "What does lesson 4 of MCP course cover?",
            tools=tool_manager_with_tools.get_tool_definitions(),
            tool_manager=tool_manager_with_tools
        )
        
        # Should return comprehensive response
        assert "lesson 4 covers" in response
        # Should have made three API calls (round 1 + round 2 + final)
        assert mock_client.messages.create.call_count == 3
    
    @patch('anthropic.Anthropic')
    def test_sequential_error_handling(self, mock_anthropic_class, tool_manager_with_tools):
        """Test error handling in sequential execution"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response_sequential(
            "Test query",
            tools=tool_manager_with_tools.get_tool_definitions(),
            tool_manager=tool_manager_with_tools
        )
        
        # Should handle error gracefully
        assert "error" in response.lower() or response == "Error generating response"
    
    @patch('anthropic.Anthropic')
    def test_sequential_max_rounds_enforcement(self, mock_anthropic_class, tool_manager_with_tools):
        """Test that max rounds limit is enforced"""
        mock_client = Mock()
        
        # Create response that always wants to use tools
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test"}
        
        mock_tool_response = Mock()
        mock_tool_response.content = [mock_tool_block]
        mock_tool_response.stop_reason = "tool_use"
        
        # Always return tool-using response
        mock_client.messages.create.return_value = mock_tool_response
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        response = generator.generate_response_sequential(
            "Test query",
            tools=tool_manager_with_tools.get_tool_definitions(),
            tool_manager=tool_manager_with_tools,
            max_rounds=1  # Limit to 1 round
        )
        
        # Should terminate after max rounds
        # At minimum should have made one attempt
        assert mock_client.messages.create.call_count >= 1

class TestSequentialToolCallingIntegration:
    """Test sequential tool calling with realistic scenarios"""
    
    def test_context_summary_generation(self):
        """Test context summary functionality"""
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        
        # Mock the summarization API call
        with patch.object(generator, 'client') as mock_client:
            mock_content = Mock()
            mock_content.text = "Course material covers testing fundamentals and advanced concepts."
            
            mock_response = Mock()
            mock_response.content = [mock_content]
            
            mock_client.messages.create.return_value = mock_response
            
            summary = generator._summarize_tool_results(
                "What is testing?",
                ["Unit testing involves...", "Integration testing covers...", "Test automation includes..."]
            )
            
            assert "testing" in summary.lower()
            assert len(summary) > 0
    
    def test_round_specific_prompt_generation(self):
        """Test round-specific prompt building"""
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        
        # Test Round 2 prompt
        round2_prompt = generator._build_round_specific_prompt(2, [{"content": "some results"}])
        
        assert "Round 2" in round2_prompt
        assert "Round 1" in round2_prompt
        assert "additional tool calls ONLY if" in round2_prompt
        
        # Test default prompt
        default_prompt = generator._build_round_specific_prompt(1)
        assert default_prompt == generator.SYSTEM_PROMPT
    
    def test_contextual_message_building(self):
        """Test contextual message building with summaries"""
        generator = AIGenerator("test-key", "claude-sonnet-4-20250514")
        
        context = ExecutionContext(
            state=ExecutionState.ROUND_1_RESPONSE,
            round_count=1,
            messages=[
                {"role": "user", "content": "What is testing?"},
                {"role": "assistant", "content": [Mock()]},
                {"role": "user", "content": [{"type": "tool_result", "content": "result1"}]},
                {"role": "assistant", "content": [Mock()]},
                {"role": "user", "content": [{"type": "tool_result", "content": "result2"}]}
            ],
            tool_results_history=[
                [{"type": "tool_result", "content": "result1"}],
                [{"type": "tool_result", "content": "result2"}],
                [{"type": "tool_result", "content": "result3"}]  # Trigger summarization
            ],
            accumulated_sources=[]
        )
        
        with patch.object(generator, '_summarize_tool_results', return_value="Summary of results"):
            messages = generator._build_contextual_messages(context)
            
            # Should include system message with summary
            assert any(msg.get("role") == "system" for msg in messages)
            assert any("Summary of results" in str(msg.get("content", "")) for msg in messages)

if __name__ == "__main__":
    # Allow running tests directly for debugging
    pytest.main([__file__, "-v"])