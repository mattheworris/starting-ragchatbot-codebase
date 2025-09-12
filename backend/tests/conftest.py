"""
Test configuration and fixtures for RAG system testing
"""
import pytest
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Any

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from config import Config
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from session_manager import SessionManager

@dataclass
class TestConfig:
    """Test-specific configuration with safe defaults"""
    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5  # Fixed: should not be 0
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"

@pytest.fixture
def test_config():
    """Provide test configuration with proper MAX_RESULTS"""
    return TestConfig()

@pytest.fixture
def temp_chroma_db():
    """Create temporary ChromaDB for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sample_course():
    """Create a sample course with lessons for testing"""
    lessons = [
        Lesson(
            lesson_number=1,
            title="Introduction to Testing",
            content="This lesson covers basic testing concepts.",
            lesson_link="https://example.com/lesson1"
        ),
        Lesson(
            lesson_number=2,
            title="Advanced Testing Techniques",
            content="This lesson covers advanced testing strategies.",
            lesson_link="https://example.com/lesson2"
        )
    ]
    
    return Course(
        title="Testing Fundamentals",
        instructor="Test Instructor",
        course_link="https://example.com/course",
        lessons=lessons
    )

@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    chunks = []
    for i, lesson in enumerate(sample_course.lessons):
        # Split lesson content into chunks
        chunk = CourseChunk(
            course_title=sample_course.title,
            lesson_number=lesson.lesson_number,
            chunk_index=i,
            content=lesson.content
        )
        chunks.append(chunk)
    return chunks

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock(spec=VectorStore)
    
    # Configure common mock behaviors
    mock_store.max_results = 5
    mock_store._resolve_course_name.return_value = "Testing Fundamentals"
    
    # Mock successful search results
    mock_results = SearchResults(
        documents=["This lesson covers basic testing concepts."],
        metadata=[{"course_title": "Testing Fundamentals", "lesson_number": 1}],
        distances=[0.1]
    )
    mock_store.search.return_value = mock_results
    
    # Mock course catalog get method
    mock_store.course_catalog = Mock()
    mock_store.course_catalog.get.return_value = {
        'metadatas': [{
            'title': 'Testing Fundamentals',
            'instructor': 'Test Instructor',
            'course_link': 'https://example.com/course',
            'lessons_json': '[{"lesson_number": 1, "lesson_title": "Introduction to Testing", "lesson_link": "https://example.com/lesson1"}]',
            'lesson_count': 1
        }]
    }
    
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
    
    return mock_store

@pytest.fixture
def mock_vector_store_empty():
    """Create a mock vector store that returns empty results"""
    mock_store = Mock(spec=VectorStore)
    mock_store.max_results = 0  # This simulates the bug
    mock_store._resolve_course_name.return_value = "Testing Fundamentals"
    
    # Mock empty search results
    empty_results = SearchResults(documents=[], metadata=[], distances=[])
    mock_store.search.return_value = empty_results
    
    return mock_store

@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()
    
    # Mock successful response
    mock_content_block = Mock()
    mock_content_block.text = "This is a test response from Claude."
    
    mock_response = Mock()
    mock_response.content = [mock_content_block]
    mock_response.stop_reason = "end_turn"
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client

@pytest.fixture
def mock_anthropic_tool_response():
    """Create a mock Anthropic response with tool usage"""
    mock_client = Mock()
    
    # Mock tool use response
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.id = "tool_12345"
    mock_tool_block.input = {"query": "testing concepts"}
    
    mock_initial_response = Mock()
    mock_initial_response.content = [mock_tool_block]
    mock_initial_response.stop_reason = "tool_use"
    
    # Mock final response after tool execution
    mock_final_content = Mock()
    mock_final_content.text = "Based on the search results, testing concepts include..."
    
    mock_final_response = Mock()
    mock_final_response.content = [mock_final_content]
    mock_final_response.stop_reason = "end_turn"
    
    mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
    
    return mock_client

@pytest.fixture
def real_vector_store(temp_chroma_db, test_config):
    """Create a real vector store with temporary database"""
    test_config.CHROMA_PATH = temp_chroma_db
    store = VectorStore(
        chroma_path=temp_chroma_db,
        embedding_model=test_config.EMBEDDING_MODEL,
        max_results=test_config.MAX_RESULTS
    )
    return store

@pytest.fixture
def populated_vector_store(real_vector_store, sample_course, sample_course_chunks):
    """Create a vector store populated with test data"""
    real_vector_store.add_course_metadata(sample_course)
    real_vector_store.add_course_content(sample_course_chunks)
    return real_vector_store

@pytest.fixture
def course_search_tool(mock_vector_store):
    """Create CourseSearchTool with mocked dependencies"""
    return CourseSearchTool(mock_vector_store)

@pytest.fixture
def course_outline_tool(mock_vector_store):
    """Create CourseOutlineTool with mocked dependencies"""
    return CourseOutlineTool(mock_vector_store)

@pytest.fixture
def tool_manager_with_tools(course_search_tool, course_outline_tool):
    """Create ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager

@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables for testing"""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)