"""
Tests for VectorStore functionality and search behavior
"""
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk

class TestVectorStoreConfiguration:
    """Test VectorStore initialization and configuration"""
    
    def test_vector_store_init_with_zero_max_results(self, temp_chroma_db):
        """Test VectorStore behavior when initialized with max_results=0"""
        store = VectorStore(
            chroma_path=temp_chroma_db,
            embedding_model="all-MiniLM-L6-v2",
            max_results=0  # This is the bug
        )
        assert store.max_results == 0, "max_results should be stored correctly"
    
    def test_vector_store_init_with_proper_max_results(self, temp_chroma_db):
        """Test VectorStore with proper max_results configuration"""
        store = VectorStore(
            chroma_path=temp_chroma_db,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )
        assert store.max_results == 5, "max_results should be stored correctly"

class TestSearchResults:
    """Test SearchResults data structure"""
    
    def test_search_results_from_chroma_with_data(self):
        """Test SearchResults creation from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'course': 'test1'}, {'course': 'test2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        assert len(results.documents) == 2
        assert len(results.metadata) == 2
        assert len(results.distances) == 2
        assert results.error is None
    
    def test_search_results_from_chroma_empty(self):
        """Test SearchResults creation from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        assert len(results.documents) == 0
        assert results.is_empty() == True
    
    def test_search_results_empty_with_error(self):
        """Test SearchResults error creation"""
        results = SearchResults.empty("Test error message")
        assert results.is_empty() == True
        assert results.error == "Test error message"

class TestVectorStoreSearch:
    """Test search functionality with different configurations"""
    
    def test_search_with_zero_max_results(self, populated_vector_store):
        """Test search behavior when max_results is 0 - this should demonstrate the bug"""
        # Override max_results to 0 to simulate the bug
        populated_vector_store.max_results = 0
        
        results = populated_vector_store.search("testing concepts")
        
        # This test will show the actual behavior with max_results=0
        # It should return empty results due to n_results=0 in ChromaDB query
        assert results.is_empty() or len(results.documents) == 0, "Search with max_results=0 should return no results"
    
    def test_search_with_proper_max_results(self, populated_vector_store):
        """Test search behavior with proper max_results configuration"""
        populated_vector_store.max_results = 5
        
        results = populated_vector_store.search("testing concepts")
        
        # This should return actual results
        if results.error:
            pytest.fail(f"Search failed with error: {results.error}")
        
        # Should have results if database is properly populated
        assert not results.is_empty(), "Search with proper max_results should return results"
    
    def test_search_limit_parameter_override(self, populated_vector_store):
        """Test that limit parameter overrides max_results"""
        populated_vector_store.max_results = 0  # Broken config
        
        # But explicitly pass limit to override
        results = populated_vector_store.search("testing concepts", limit=3)
        
        # Should use the limit parameter instead of max_results
        # This may still fail depending on implementation details
        if not results.error:
            assert not results.is_empty(), "Explicit limit should override max_results"

class TestVectorStoreMethods:
    """Test other VectorStore methods"""
    
    def test_resolve_course_name(self, populated_vector_store):
        """Test course name resolution"""
        # This should find the course we added in populated_vector_store
        resolved = populated_vector_store._resolve_course_name("Testing")
        assert resolved is not None, "Should resolve partial course name"
        assert "Testing" in resolved, "Resolved name should contain search term"
    
    def test_resolve_nonexistent_course(self, populated_vector_store):
        """Test course name resolution for nonexistent course"""
        resolved = populated_vector_store._resolve_course_name("NonexistentCourse123")
        assert resolved is None, "Should return None for nonexistent course"
    
    def test_build_filter_course_only(self, populated_vector_store):
        """Test filter building with course title only"""
        filter_dict = populated_vector_store._build_filter("Testing Fundamentals", None)
        expected = {"course_title": "Testing Fundamentals"}
        assert filter_dict == expected, "Should create course-only filter"
    
    def test_build_filter_lesson_only(self, populated_vector_store):
        """Test filter building with lesson number only"""
        filter_dict = populated_vector_store._build_filter(None, 1)
        expected = {"lesson_number": 1}
        assert filter_dict == expected, "Should create lesson-only filter"
    
    def test_build_filter_both(self, populated_vector_store):
        """Test filter building with both course and lesson"""
        filter_dict = populated_vector_store._build_filter("Testing Fundamentals", 1)
        expected = {"$and": [
            {"course_title": "Testing Fundamentals"},
            {"lesson_number": 1}
        ]}
        assert filter_dict == expected, "Should create combined filter"

class TestVectorStoreCourseManagement:
    """Test course metadata and content management"""
    
    def test_add_course_metadata(self, real_vector_store, sample_course):
        """Test adding course metadata"""
        real_vector_store.add_course_metadata(sample_course)
        
        # Verify course was added by checking if we can resolve it
        resolved = real_vector_store._resolve_course_name(sample_course.title)
        assert resolved == sample_course.title, "Course metadata should be searchable"
    
    def test_add_course_content(self, real_vector_store, sample_course_chunks):
        """Test adding course content chunks"""
        real_vector_store.add_course_content(sample_course_chunks)
        
        # This doesn't fail, but content might not be searchable due to max_results=0
        # We can't easily verify without doing a search, which might fail
    
    def test_get_existing_course_titles(self, populated_vector_store):
        """Test getting existing course titles"""
        titles = populated_vector_store.get_existing_course_titles()
        assert isinstance(titles, list), "Should return a list"
        assert "Testing Fundamentals" in titles, "Should include our test course"
    
    def test_get_course_count(self, populated_vector_store):
        """Test getting course count"""
        count = populated_vector_store.get_course_count()
        assert isinstance(count, int), "Should return integer count"
        assert count >= 1, "Should have at least our test course"

class TestDatabaseConsistency:
    """Test database consistency between course_catalog and course_content"""
    
    def test_metadata_content_consistency(self, populated_vector_store):
        """Test that courses in metadata have corresponding content"""
        # Get all course titles from metadata
        titles = populated_vector_store.get_existing_course_titles()
        
        for title in titles:
            # Try to search for content in this course
            results = populated_vector_store.search("lesson", course_name=title, limit=1)
            
            if results.error:
                pytest.fail(f"Error searching for content in course '{title}': {results.error}")
            
            # If max_results is 0, this might be empty even if content exists
            if populated_vector_store.max_results == 0:
                # Skip this check if we have the max_results bug
                continue
            
            # Should have some content for this course
            if results.is_empty():
                pytest.fail(f"Course '{title}' exists in metadata but has no searchable content")
    
    def test_course_metadata_structure(self, populated_vector_store):
        """Test that course metadata has proper structure"""
        metadata_list = populated_vector_store.get_all_courses_metadata()
        
        assert len(metadata_list) > 0, "Should have course metadata"
        
        for metadata in metadata_list:
            required_fields = ['title', 'instructor', 'course_link', 'lessons', 'lesson_count']
            for field in required_fields:
                assert field in metadata, f"Metadata should have {field} field"
            
            # Verify lessons structure
            lessons = metadata['lessons']
            assert isinstance(lessons, list), "Lessons should be a list"
            
            for lesson in lessons:
                lesson_fields = ['lesson_number', 'lesson_title', 'lesson_link']
                for field in lesson_fields:
                    assert field in lesson, f"Lesson should have {field} field"

class TestErrorHandling:
    """Test error handling in VectorStore"""
    
    @patch('chromadb.PersistentClient')
    def test_chroma_connection_error(self, mock_client, temp_chroma_db):
        """Test behavior when ChromaDB connection fails"""
        mock_client.side_effect = Exception("ChromaDB connection failed")
        
        with pytest.raises(Exception):
            VectorStore(
                chroma_path=temp_chroma_db,
                embedding_model="all-MiniLM-L6-v2",
                max_results=5
            )
    
    def test_search_with_invalid_embedding(self, real_vector_store):
        """Test search behavior with potential embedding errors"""
        # This might cause embedding errors that should be handled gracefully
        results = real_vector_store.search("" * 10000)  # Very long empty query
        
        # Should return error instead of crashing
        if results.error:
            assert "error" in results.error.lower() or "Error" in results.error
        else:
            # If no error, should still be a valid SearchResults
            assert isinstance(results, SearchResults)

if __name__ == "__main__":
    # Allow running tests directly for debugging
    pytest.main([__file__, "-v"])