"""
Integration tests for the complete RAG system pipeline
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from config import Config
from models import Course, Lesson, CourseChunk

class TestRAGSystemInitialization:
    """Test RAGSystem initialization and component integration"""
    
    def test_rag_system_init_with_proper_config(self, test_config, temp_chroma_db):
        """Test RAGSystem initializes all components properly"""
        test_config.CHROMA_PATH = temp_chroma_db
        
        rag_system = RAGSystem(test_config)
        
        # All components should be initialized
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None
        
        # Tools should be registered
        tool_defs = rag_system.tool_manager.get_tool_definitions()
        assert len(tool_defs) == 2
        tool_names = [tool['name'] for tool in tool_defs]
        assert 'search_course_content' in tool_names
        assert 'get_course_outline' in tool_names
    
    def test_rag_system_init_with_broken_config(self, temp_chroma_db):
        """Test RAGSystem behavior with problematic configuration"""
        # Create config with MAX_RESULTS = 0 (the bug we fixed)
        broken_config = Config()
        broken_config.MAX_RESULTS = 0
        broken_config.CHROMA_PATH = temp_chroma_db
        
        rag_system = RAGSystem(broken_config)
        
        # System should still initialize
        assert rag_system.vector_store.max_results == 0
        # But searches will fail due to the bug

class TestEndToEndQueryProcessing:
    """Test complete query processing pipeline"""
    
    @patch('anthropic.Anthropic')
    def test_content_query_pipeline(self, mock_anthropic_class, populated_vector_store, test_config):
        """Test complete content query from user input to response"""
        # Set up mocks
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool use response
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "testing concepts"}
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]
        mock_initial_response.stop_reason = "tool_use"
        
        # Mock final response
        mock_final_content = Mock()
        mock_final_content.text = "Testing involves verifying that software works correctly."
        
        mock_final_response = Mock()
        mock_final_response.content = [mock_final_content]
        mock_final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        
        # Create RAG system with populated database
        test_config.CHROMA_PATH = populated_vector_store.client.path
        test_config.MAX_RESULTS = 5  # Ensure proper results
        rag_system = RAGSystem(test_config)
        rag_system.vector_store = populated_vector_store
        
        # Process query
        response, sources = rag_system.query("What are testing concepts?")
        
        # Verify response
        assert response == "Testing involves verifying that software works correctly."
        assert isinstance(sources, list)
        
        # Verify API was called twice (initial + final)
        assert mock_client.messages.create.call_count == 2
        
        # Verify tool was executed
        initial_call = mock_client.messages.create.call_args_list[0]
        assert "tools" in initial_call[1]
    
    @patch('anthropic.Anthropic')
    def test_outline_query_pipeline(self, mock_anthropic_class, populated_vector_store, test_config):
        """Test complete outline query pipeline"""
        # Set up mocks for outline tool usage
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "get_course_outline"
        mock_tool_block.id = "tool_456"
        mock_tool_block.input = {"course_title": "Testing Fundamentals"}
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]
        mock_initial_response.stop_reason = "tool_use"
        
        mock_final_content = Mock()
        mock_final_content.text = "**Testing Fundamentals**\nInstructor: Test Instructor\n\n**Lessons:**\n1. Introduction to Testing\n2. Advanced Testing Techniques"
        
        mock_final_response = Mock()
        mock_final_response.content = [mock_final_content]
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        
        # Set up RAG system
        test_config.CHROMA_PATH = populated_vector_store.client.path
        rag_system = RAGSystem(test_config)
        rag_system.vector_store = populated_vector_store
        
        # Process outline query
        response, sources = rag_system.query("What's the outline of Testing Fundamentals?")
        
        # Verify response contains course structure
        assert "Testing Fundamentals" in response
        assert "Test Instructor" in response
        assert "Lessons:" in response
        assert isinstance(sources, list)
    
    def test_query_without_tools(self, populated_vector_store, test_config):
        """Test query processing when AI doesn't use tools"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock direct response without tool use
            mock_content = Mock()
            mock_content.text = "This is a general knowledge answer."
            
            mock_response = Mock()
            mock_response.content = [mock_content]
            mock_response.stop_reason = "end_turn"
            
            mock_client.messages.create.return_value = mock_response
            
            # Set up RAG system
            test_config.CHROMA_PATH = populated_vector_store.client.path
            rag_system = RAGSystem(test_config)
            
            # Process general query
            response, sources = rag_system.query("What is programming in general?")
            
            # Should return direct response without sources
            assert response == "This is a general knowledge answer."
            assert sources == []  # No tool sources
    
    def test_query_with_session_management(self, populated_vector_store, test_config):
        """Test query processing with conversation history"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            mock_content = Mock()
            mock_content.text = "This is a follow-up response."
            
            mock_response = Mock()
            mock_response.content = [mock_content]
            mock_response.stop_reason = "end_turn"
            
            mock_client.messages.create.return_value = mock_response
            
            # Set up RAG system
            test_config.CHROMA_PATH = populated_vector_store.client.path
            rag_system = RAGSystem(test_config)
            
            # First query to establish session
            response1, _ = rag_system.query("What is testing?", session_id="test_session")
            
            # Second query in same session - should include history
            response2, _ = rag_system.query("Can you explain more?", session_id="test_session")
            
            # Verify both queries succeeded
            assert response1 == "This is a follow-up response."
            assert response2 == "This is a follow-up response."
            
            # Verify session was used in second call
            assert mock_client.messages.create.call_count == 2
            second_call = mock_client.messages.create.call_args_list[1]
            system_prompt = second_call[1]["system"]
            assert "Previous conversation" in system_prompt

class TestErrorHandlingIntegration:
    """Test error handling across the complete system"""
    
    def test_query_with_anthropic_api_error(self, populated_vector_store, test_config):
        """Test system behavior when Anthropic API fails"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock API error
            mock_client.messages.create.side_effect = Exception("API rate limit exceeded")
            
            test_config.CHROMA_PATH = populated_vector_store.client.path
            rag_system = RAGSystem(test_config)
            
            # Query should raise exception
            with pytest.raises(Exception) as exc_info:
                rag_system.query("Test query")
            
            assert "API rate limit exceeded" in str(exc_info.value)
    
    def test_query_with_vector_store_error(self, test_config, temp_chroma_db):
        """Test system behavior when vector store fails"""
        test_config.CHROMA_PATH = temp_chroma_db
        rag_system = RAGSystem(test_config)
        
        # Mock vector store to raise error
        with patch.object(rag_system.vector_store, 'search') as mock_search:
            mock_search.side_effect = Exception("ChromaDB connection failed")
            
            with patch('anthropic.Anthropic') as mock_anthropic_class:
                # Set up tool use that will trigger vector store error
                mock_client = Mock()
                mock_anthropic_class.return_value = mock_client
                
                mock_tool_block = Mock()
                mock_tool_block.type = "tool_use"
                mock_tool_block.name = "search_course_content"
                mock_tool_block.id = "tool_123"
                mock_tool_block.input = {"query": "test"}
                
                mock_initial_response = Mock()
                mock_initial_response.content = [mock_tool_block]
                mock_initial_response.stop_reason = "tool_use"
                
                mock_final_content = Mock()
                mock_final_content.text = "Error occurred during search."
                
                mock_final_response = Mock()
                mock_final_response.content = [mock_final_content]
                
                mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
                
                # Query should complete but with error result
                response, sources = rag_system.query("Test query")
                
                # Should still return a response (AI handles the error)
                assert response == "Error occurred during search."
    
    def test_query_with_max_results_zero_bug(self, populated_vector_store, test_config):
        """Test system behavior with the MAX_RESULTS = 0 bug"""
        # Simulate the bug
        test_config.MAX_RESULTS = 0
        test_config.CHROMA_PATH = populated_vector_store.client.path
        
        rag_system = RAGSystem(test_config)
        rag_system.vector_store = populated_vector_store
        rag_system.vector_store.max_results = 0  # Ensure the bug is present
        
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Set up tool use
            mock_tool_block = Mock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "search_course_content"
            mock_tool_block.id = "tool_123"
            mock_tool_block.input = {"query": "testing"}
            
            mock_initial_response = Mock()
            mock_initial_response.content = [mock_tool_block]
            mock_initial_response.stop_reason = "tool_use"
            
            # AI responds to "no content found"
            mock_final_content = Mock()
            mock_final_content.text = "I couldn't find any relevant content."
            
            mock_final_response = Mock()
            mock_final_response.content = [mock_final_content]
            
            mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
            
            # Process query
            response, sources = rag_system.query("What is testing?")
            
            # Should indicate no content found due to the bug
            assert "couldn't find" in response.lower() or "no" in response.lower()

class TestCourseManagementIntegration:
    """Test course loading and management integration"""
    
    def test_add_course_document_integration(self, test_config, temp_chroma_db, sample_course):
        """Test adding a course document to the system"""
        test_config.CHROMA_PATH = temp_chroma_db
        rag_system = RAGSystem(test_config)
        
        # Create temporary course file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"Course Title: {sample_course.title}\n")
            f.write(f"Course Link: {sample_course.course_link}\n")
            f.write(f"Course Instructor: {sample_course.instructor}\n\n")
            
            for lesson in sample_course.lessons:
                f.write(f"Lesson {lesson.lesson_number}: {lesson.title}\n")
                f.write(f"Lesson Link: {lesson.lesson_link}\n")
                f.write(f"{lesson.content}\n\n")
            
            temp_file = f.name
        
        try:
            # Add course to system
            course, chunk_count = rag_system.add_course_document(temp_file)
            
            # Verify course was added
            assert course is not None
            assert course.title == sample_course.title
            assert chunk_count > 0
            
            # Verify course appears in analytics
            analytics = rag_system.get_course_analytics()
            assert sample_course.title in analytics["course_titles"]
            
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def test_course_folder_processing_integration(self, test_config, temp_chroma_db):
        """Test processing a folder of course documents"""
        test_config.CHROMA_PATH = temp_chroma_db
        rag_system = RAGSystem(test_config)
        
        # Create temporary directory with course files
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a test course file
            course_file = os.path.join(temp_dir, "test_course.txt")
            with open(course_file, 'w') as f:
                f.write("Course Title: Integration Test Course\n")
                f.write("Course Link: https://example.com/course\n")
                f.write("Course Instructor: Test Instructor\n\n")
                f.write("Lesson 1: Introduction\n")
                f.write("Lesson Link: https://example.com/lesson1\n")
                f.write("This is the introduction lesson content.\n\n")
            
            # Process folder
            courses_added, chunks_added = rag_system.add_course_folder(temp_dir)
            
            # Verify processing
            assert courses_added == 1
            assert chunks_added > 0
            
            # Verify course is searchable
            analytics = rag_system.get_course_analytics()
            assert "Integration Test Course" in analytics["course_titles"]
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
    
    def test_incremental_course_loading(self, test_config, temp_chroma_db):
        """Test that existing courses are not reprocessed"""
        test_config.CHROMA_PATH = temp_chroma_db
        rag_system = RAGSystem(test_config)
        
        # Add course first time
        import tempfile
        temp_dir = tempfile.mkdtemp()
        try:
            course_file = os.path.join(temp_dir, "existing_course.txt")
            with open(course_file, 'w') as f:
                f.write("Course Title: Existing Course\n")
                f.write("Course Link: https://example.com/existing\n")
                f.write("Course Instructor: Existing Instructor\n\n")
                f.write("Lesson 1: Existing Lesson\n")
                f.write("This is existing content.\n\n")
            
            # First load
            courses1, chunks1 = rag_system.add_course_folder(temp_dir)
            assert courses1 == 1
            assert chunks1 > 0
            
            # Second load - should skip existing course
            courses2, chunks2 = rag_system.add_course_folder(temp_dir)
            assert courses2 == 0  # No new courses added
            assert chunks2 == 0   # No new chunks added
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Allow running tests directly for debugging
    pytest.main([__file__, "-v"])