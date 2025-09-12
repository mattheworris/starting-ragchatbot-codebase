# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Important**: This project uses `uv` as the package manager. Always use `uv` to manage all dependencies and execute Python commands. Never use `pip`, `pip install`, or direct Python execution.

### Running the Application
```bash
# Quick start using the provided script
./run.sh

# Manual start (from project root)
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Dependencies
```bash
# Install all dependencies from lock file
uv sync

# Add new runtime dependencies
uv add package-name

# Add development dependencies
uv add --dev package-name

# Remove dependencies
uv remove package-name

# Update dependencies
uv lock --upgrade

# Run Python scripts/commands
uv run python script.py
uv run pytest
uv run black .
```

### Environment Setup
```bash
# Copy environment template and add your API key
cp .env.example .env
# Edit .env to add: ANTHROPIC_API_KEY=your-api-key-here
```

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) system** for querying course materials. The architecture follows a tool-based AI approach where the Claude model can intelligently search and retrieve information.

### Core Components

**RAG System Flow (`rag_system.py`)**:
- Main orchestrator that coordinates all components
- Manages the query → search → AI generation → response pipeline
- Handles session management and conversation context

**AI-Tool Integration**:
- AI model (`ai_generator.py`) has access to search tools
- `CourseSearchTool` performs semantic search when AI needs information
- AI can make multiple tool calls per query for comprehensive answers
- Tool results are automatically tracked for source attribution

**Document Processing Pipeline**:
- `DocumentProcessor` parses structured course files with format:
  ```
  Course Title: [title]
  Course Link: [url]
  Course Instructor: [instructor]
  
  Lesson 0: [title]
  Lesson Link: [url]
  [content...]
  ```
- Text is chunked with sentence-boundary awareness and overlap
- Each chunk gets contextual metadata (course title, lesson number)

**Vector Storage (`vector_store.py`)**:
- ChromaDB for semantic search using embeddings
- Stores both course metadata and content chunks
- Supports incremental updates (avoids re-processing existing courses)

### Key Architectural Patterns

**Tool-Based AI**: Unlike traditional RAG, the AI decides when and how to search rather than automatically retrieving context for every query.

**Session-Aware**: Conversation history is maintained per session for context-aware responses.

**Source Attribution**: All search results are tracked and returned to users for transparency.

**Structured Course Data**: Course documents follow a specific format that preserves lesson structure and metadata.

## Configuration

Key settings in `backend/config.py`:
- `CHUNK_SIZE`: 800 characters per text chunk
- `CHUNK_OVERLAP`: 100 characters overlap between chunks  
- `MAX_RESULTS`: 5 search results per query
- `MAX_HISTORY`: 2 conversation turns remembered
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514

## File Structure Context

- `/backend/`: FastAPI server with all RAG logic
- `/frontend/`: Vanilla HTML/CSS/JS web interface
- `/docs/`: Course material files (course1_script.txt, etc.)
- `run.sh`: Start script that launches backend server
- No test suite or linting tools configured

## Adding New Courses

Place text files in `/docs/` following the structured format. The system automatically processes new courses on startup and avoids re-processing existing ones.