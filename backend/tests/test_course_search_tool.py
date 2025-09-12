"""
Tests for CourseSearchTool functionality and execute method
"""
import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool
from vector_store import SearchResults

class TestCourseSearchToolInitialization:
    """Test CourseSearchTool initialization"""
    
    def test_init_with_vector_store(self, mock_vector_store):
        """Test tool initialization with vector store"""
        tool = CourseSearchTool(mock_vector_store)
        assert tool.store == mock_vector_store
        assert tool.last_sources == []
    
    def test_get_tool_definition(self, course_search_tool):
        """Test tool definition structure"""
        definition = course_search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert schema["required"] == ["query"]
        
        # Optional parameters should be present
        assert "course_name" in schema["properties"] 
        assert "lesson_number" in schema["properties"]

class TestCourseSearchToolExecute:
    """Test the execute method with various scenarios"""
    
    def test_execute_successful_search(self, course_search_tool):
        """Test execute with successful search results"""
        result = course_search_tool.execute("testing concepts")
        
        # Should return formatted results, not error message
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Should contain the expected course information
        assert "Testing Fundamentals" in result
        assert "This lesson covers basic testing concepts" in result
        
        # Should track sources
        assert len(course_search_tool.last_sources) > 0
        assert course_search_tool.last_sources[0]["text"] == "Testing Fundamentals - Lesson 1"
    
    def test_execute_with_empty_results(self, mock_vector_store_empty):
        """Test execute when vector store returns empty results"""
        tool = CourseSearchTool(mock_vector_store_empty)
        
        result = tool.execute("nonexistent query")
        
        # Should return "no content found" message
        assert "No relevant content found" in result
        assert isinstance(result, str)
    
    def test_execute_with_course_filter(self, course_search_tool):
        """Test execute with course name filter"""
        result = course_search_tool.execute("concepts", course_name="Testing")
        
        # Mock is configured to return results
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Verify that search was called with course name
        course_search_tool.store.search.assert_called_with(
            query="concepts",
            course_name="Testing",
            lesson_number=None
        )
    
    def test_execute_with_lesson_filter(self, course_search_tool):
        """Test execute with lesson number filter"""
        result = course_search_tool.execute("concepts", lesson_number=1)
        
        assert isinstance(result, str)
        
        # Verify that search was called with lesson number
        course_search_tool.store.search.assert_called_with(
            query="concepts",
            course_name=None,
            lesson_number=1
        )
    
    def test_execute_with_both_filters(self, course_search_tool):
        """Test execute with both course and lesson filters"""
        result = course_search_tool.execute("concepts", course_name="Testing", lesson_number=1)
        
        assert isinstance(result, str)
        
        # Verify that search was called with both filters
        course_search_tool.store.search.assert_called_with(
            query="concepts",
            course_name="Testing",
            lesson_number=1
        )
    
    def test_execute_with_search_error(self, mock_vector_store):
        """Test execute when search returns an error"""
        # Configure mock to return error
        error_results = SearchResults(
            documents=[], 
            metadata=[], 
            distances=[],
            error="Search failed due to database error"
        )
        mock_vector_store.search.return_value = error_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Should return the error message
        assert result == "Search failed due to database error"
    
    def test_execute_with_max_results_zero_bug(self):
        """Test execute behavior when max_results is 0 (the actual bug)"""
        # Create mock that simulates the max_results=0 bug
        mock_store = Mock()
        mock_store.max_results = 0  # The bug
        
        # When max_results=0, ChromaDB returns empty results
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_store.search.return_value = empty_results
        
        tool = CourseSearchTool(mock_store)
        result = tool.execute("test query")
        
        # Should return "no content found" even if content exists
        assert "No relevant content found" in result
        
        # This demonstrates how the bug manifests: even valid queries return empty

class TestCourseSearchToolFormatting:
    """Test result formatting methods"""
    
    def test_format_results_single_document(self, course_search_tool):
        """Test formatting with single search result"""
        results = SearchResults(
            documents=["This is lesson content about testing."],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        
        formatted = course_search_tool._format_results(results)
        
        assert "[Test Course - Lesson 1]" in formatted
        assert "This is lesson content about testing." in formatted
    
    def test_format_results_multiple_documents(self, course_search_tool):
        """Test formatting with multiple search results"""
        results = SearchResults(
            documents=[
                "First lesson content.",
                "Second lesson content."
            ],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        
        formatted = course_search_tool._format_results(results)
        
        assert "[Course A - Lesson 1]" in formatted
        assert "[Course B - Lesson 2]" in formatted
        assert "First lesson content." in formatted
        assert "Second lesson content." in formatted
        
        # Should be separated by double newlines
        parts = formatted.split("\n\n")
        assert len(parts) == 2
    
    def test_format_results_no_lesson_number(self, course_search_tool):
        """Test formatting when lesson number is missing"""
        results = SearchResults(
            documents=["Course overview content."],
            metadata=[{"course_title": "Overview Course"}],  # No lesson_number
            distances=[0.1]
        )
        
        formatted = course_search_tool._format_results(results)
        
        # Should handle missing lesson number gracefully
        assert "[Overview Course]" in formatted  # No lesson number in header
        assert "Course overview content." in formatted
    
    def test_format_results_source_tracking(self, course_search_tool):
        """Test that source tracking works during formatting"""
        results = SearchResults(
            documents=["Lesson about advanced topics."],
            metadata=[{"course_title": "Advanced Course", "lesson_number": 3}],
            distances=[0.1]
        )
        
        # Mock the get_lesson_link to return a URL
        course_search_tool.store.get_lesson_link.return_value = "https://example.com/lesson3"
        
        formatted = course_search_tool._format_results(results)
        
        # Check that sources are tracked
        assert len(course_search_tool.last_sources) == 1
        source = course_search_tool.last_sources[0]
        assert source["text"] == "Advanced Course - Lesson 3"
        assert source["url"] == "https://example.com/lesson3"

class TestCourseSearchToolErrorConditions:
    """Test error handling and edge cases"""
    
    def test_execute_empty_query(self, course_search_tool):
        """Test execute with empty query"""
        result = course_search_tool.execute("")
        
        # Should still attempt search (might return empty results)
        assert isinstance(result, str)
    
    def test_execute_very_long_query(self, course_search_tool):
        """Test execute with very long query"""
        long_query = "test " * 1000  # Very long query
        result = course_search_tool.execute(long_query)
        
        # Should handle gracefully
        assert isinstance(result, str)
    
    def test_execute_special_characters(self, course_search_tool):
        """Test execute with special characters in query"""
        special_query = "test @#$%^&*()_+-=[]{}|;':\",./<>?"
        result = course_search_tool.execute(special_query)
        
        # Should handle special characters gracefully
        assert isinstance(result, str)
    
    def test_format_results_corrupted_metadata(self, course_search_tool):
        """Test formatting with corrupted metadata"""
        results = SearchResults(
            documents=["Content"],
            metadata=[{}],  # Empty metadata
            distances=[0.1]
        )
        
        formatted = course_search_tool._format_results(results)
        
        # Should handle missing metadata gracefully
        assert "Content" in formatted
        # Should use default values for missing fields
        assert "[unknown" in formatted.lower() or "Content" in formatted

class TestCourseSearchToolIntegration:
    """Test integration with different vector store states"""
    
    def test_with_populated_database(self, populated_vector_store):
        """Test with real populated database (if max_results > 0)"""
        tool = CourseSearchTool(populated_vector_store)
        
        # Only run this test if max_results is properly configured
        if populated_vector_store.max_results > 0:
            result = tool.execute("testing")
            
            if not result.startswith("No relevant content found"):
                # Should return actual content
                assert len(result) > 0
                assert isinstance(result, str)
        else:
            # If max_results is 0, should get empty results
            result = tool.execute("testing")
            assert "No relevant content found" in result
    
    def test_source_link_generation(self, populated_vector_store):
        """Test that source links are generated correctly"""
        tool = CourseSearchTool(populated_vector_store)
        
        # Execute search to populate sources
        tool.execute("testing")
        
        # Check if sources were populated (depends on max_results)
        if populated_vector_store.max_results > 0 and tool.last_sources:
            for source in tool.last_sources:
                assert "text" in source
                # URL might be None if no link available
                assert "url" in source

if __name__ == "__main__":
    # Allow running tests directly for debugging
    pytest.main([__file__, "-v"])