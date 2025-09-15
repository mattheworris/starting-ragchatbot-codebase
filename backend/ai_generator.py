import anthropic
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field

class ExecutionState(Enum):
    """States for sequential tool calling state machine"""
    INITIAL = "initial"
    ROUND_1_TOOLS = "round_1_tools"
    ROUND_1_RESPONSE = "round_1_response"
    ROUND_2_TOOLS = "round_2_tools"
    ROUND_2_RESPONSE = "round_2_response"
    COMPLETED = "completed"

@dataclass
class ExecutionContext:
    """Tracks state and context across multiple rounds of tool calling"""
    state: ExecutionState
    round_count: int
    messages: List[Dict[str, Any]]
    tool_results_history: List[Dict[str, Any]]
    accumulated_sources: List[Dict[str, str]]
    error_count: int = 0
    max_rounds: int = 2
    final_response: Optional[str] = None
    
    def add_message(self, role: str, content: Any):
        """Add a message to the conversation"""
        self.messages.append({"role": role, "content": content})
    
    def add_tool_results(self, results: List[Dict[str, Any]]):
        """Add tool results to history"""
        self.tool_results_history.extend(results)
    
    def increment_round(self):
        """Move to next round"""
        self.round_count += 1
    
    def increment_error(self):
        """Increment error count"""
        self.error_count += 1
    
    def is_completed(self) -> bool:
        """Check if execution should terminate"""
        return (self.state == ExecutionState.COMPLETED or 
                self.round_count > self.max_rounds or 
                self.error_count > 3)

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

**Multi-Round Strategy:**
You can make tool calls across up to 2 rounds to thoroughly answer questions:
- **Round 1**: Make initial tool calls based on the user's query
- **Round 2** (if needed): Make additional tool calls to fill gaps or get more specific information

**Tool Usage Guidelines:**
- **Content Search Tool**: Use for specific course content, concepts, or detailed educational materials
- **Course Outline Tool**: Use for course structure, lesson lists, course overviews, or when users ask "what's in this course"
- **Multiple tools per round allowed**: You can call both tools in one round if beneficial
- **Cross-round learning**: Use results from Round 1 to inform Round 2 tool calls
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

**Decision Making:**
- After Round 1: Assess if you have sufficient information
- Continue to Round 2 if: gaps remain, query has multiple parts, or deeper investigation would help
- Stop after Round 1 if: comprehensive answer is possible with current information

**Response Protocol:**
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use content search tool first, then answer
- **Course outline/structure questions**: Use outline tool to get course title, instructor, course link, and complete lesson list
- **Complex questions**: May require both tools across multiple rounds
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"
 - Do not mention rounds or tool usage strategy

For outline queries, always include:
- Course title
- Course link (if available)
- Instructor (if available)
- Complete numbered lesson list with titles

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.

Remember: Quality over quantity - only proceed to additional rounds if it genuinely improves the answer.
"""

    SUMMARY_PROMPT = """Summarize the key information from this tool-based conversation:

Original Query: {query}
Tool Results: {results}

Provide a concise summary of the factual information found and any insights discovered. Focus on information that would be relevant for follow-up questions. Be brief and factual."""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
    
    def generate_response_sequential(self, query: str,
                                   conversation_history: Optional[str] = None,
                                   tools: Optional[List] = None,
                                   tool_manager=None,
                                   max_rounds: int = 2) -> str:
        """
        Generate AI response with sequential tool calling across up to 2 rounds.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of rounds (default 2)
            
        Returns:
            Generated response as string
        """
        if not tools or not tool_manager:
            # No tools available, use standard generation
            return self.generate_response(query, conversation_history)
        
        # Initialize execution context
        context = ExecutionContext(
            state=ExecutionState.INITIAL,
            round_count=0,
            messages=[{"role": "user", "content": query}],
            tool_results_history=[],
            accumulated_sources=[],
            max_rounds=max_rounds
        )
        
        # Build initial system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Execute state machine
        while not context.is_completed():
            context = self._process_state(context, system_content, tools, tool_manager)
            
            # Safety check for infinite loops
            if context.round_count > max_rounds or context.error_count > 3:
                break
        
        return context.final_response or "Error generating response"
    
    def _process_state(self, context: ExecutionContext, system_content: str,
                      tools: List, tool_manager) -> ExecutionContext:
        """Process current state and transition to next state"""
        
        try:
            if context.state == ExecutionState.INITIAL:
                return self._handle_initial_state(context, system_content, tools)
            elif context.state == ExecutionState.ROUND_1_TOOLS:
                return self._handle_tool_execution_state(context, system_content, tools, tool_manager, 1)
            elif context.state == ExecutionState.ROUND_1_RESPONSE:
                return self._handle_response_analysis_state(context, tools, 1)
            elif context.state == ExecutionState.ROUND_2_TOOLS:
                return self._handle_tool_execution_state(context, system_content, tools, tool_manager, 2)
            elif context.state == ExecutionState.ROUND_2_RESPONSE:
                return self._handle_final_response_state(context, system_content)
            
        except Exception as e:
            print(f"State processing error: {e}")
            context.increment_error()
            if context.error_count > 3:
                context.state = ExecutionState.COMPLETED
                context.final_response = "I apologize, but I encountered an error while processing your request."
        
        return context
    
    def _handle_initial_state(self, context: ExecutionContext, system_content: str,
                             tools: List) -> ExecutionContext:
        """Handle initial state - make first API call"""
        
        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": context.messages.copy(),
            "system": system_content,
            "tools": tools,
            "tool_choice": {"type": "auto"}
        }
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Add AI's response to messages
        context.add_message("assistant", response.content)
        
        # Check if tools were used
        if response.stop_reason == "tool_use":
            context.state = ExecutionState.ROUND_1_TOOLS
            context.increment_round()
        else:
            # No tools used, we're done
            context.state = ExecutionState.COMPLETED
            context.final_response = response.content[0].text
        
        return context
    
    def _handle_tool_execution_state(self, context: ExecutionContext, system_content: str,
                                   tools: List, tool_manager, round_num: int) -> ExecutionContext:
        """Handle tool execution and add results to context"""
        
        # Get the last assistant message (contains tool calls)
        last_message = context.messages[-1]
        
        # Execute tools and collect results
        tool_results = []
        for content_block in last_message["content"]:
            if hasattr(content_block, 'type') and content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                    
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution error: {str(e)}"
                    })
        
        # Add tool results to context
        if tool_results:
            context.add_message("user", tool_results)
            context.add_tool_results(tool_results)
        
        # Transition to response analysis
        if round_num == 1:
            context.state = ExecutionState.ROUND_1_RESPONSE
        else:
            context.state = ExecutionState.ROUND_2_RESPONSE
        
        return context
    
    def _handle_response_analysis_state(self, context: ExecutionContext,
                                      tools: List, round_num: int) -> ExecutionContext:
        """Analyze if we need another round or can provide final response"""
        
        # Build messages with context summary for efficiency
        messages = self._build_contextual_messages(context)
        
        # Make API call without tools to get response or check if more tools needed
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": self._build_round_specific_prompt(round_num + 1, context.tool_results_history),
            "tools": tools if round_num < context.max_rounds else None,
            "tool_choice": {"type": "auto"} if round_num < context.max_rounds else None
        }
        
        response = self.client.messages.create(**api_params)
        
        # Add response to messages
        context.add_message("assistant", response.content)
        
        # Check if tools were used and we haven't exceeded max rounds
        if (response.stop_reason == "tool_use" and 
            round_num < context.max_rounds):
            # Continue to next round
            if round_num == 1:
                context.state = ExecutionState.ROUND_2_TOOLS
                context.increment_round()
            else:
                context.state = ExecutionState.COMPLETED
                context.final_response = response.content[0].text
        else:
            # Done - provide final response
            context.state = ExecutionState.COMPLETED
            context.final_response = response.content[0].text
        
        return context
    
    def _handle_final_response_state(self, context: ExecutionContext,
                                   system_content: str) -> ExecutionContext:
        """Generate final response after all tool rounds"""
        
        # Build final messages with all context
        messages = self._build_contextual_messages(context)
        
        # Make final API call without tools
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content + "\n\nProvide your final comprehensive response based on all the information gathered."
        }
        
        response = self.client.messages.create(**api_params)
        
        context.state = ExecutionState.COMPLETED
        context.final_response = response.content[0].text
        
        return context
    
    def _build_contextual_messages(self, context: ExecutionContext) -> List[Dict[str, Any]]:
        """Build messages with summarized context for efficiency"""
        
        if len(context.tool_results_history) <= 2:
            # Not much history, use full messages
            return context.messages.copy()
        
        # Summarize previous tool results
        all_results = []
        for result_batch in context.tool_results_history:
            if isinstance(result_batch, list):
                for result in result_batch:
                    if isinstance(result, dict) and "content" in result:
                        all_results.append(str(result["content"]))
            else:
                all_results.append(str(result_batch))
        
        if all_results:
            summary = self._summarize_tool_results(context.messages[0]["content"], all_results)
            
            # Build new message list with summary
            messages = [
                {"role": "system", "content": f"Context from previous analysis: {summary}"},
                context.messages[0]  # Original user query
            ]
            
            # Add only the most recent exchanges
            if len(context.messages) > 3:
                messages.extend(context.messages[-3:])
            else:
                messages.extend(context.messages[1:])
            
            return messages
        
        return context.messages.copy()
    
    def _summarize_tool_results(self, original_query: str, tool_results: List[str]) -> str:
        """Generate summary of previous tool results"""
        
        summary_prompt = self.SUMMARY_PROMPT.format(
            query=original_query,
            results="\n".join(tool_results[:5])  # Limit to prevent token overflow
        )
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0,
                messages=[{"role": "user", "content": summary_prompt}],
                system="Provide concise, factual summaries."
            )
            return response.content[0].text
        except Exception:
            # Fallback if summarization fails
            return "Previous searches found relevant course information."
    
    def _build_round_specific_prompt(self, round_number: int, 
                                   previous_results: List[Dict[str, Any]] = None) -> str:
        """Build prompts tailored to specific rounds"""
        
        base_prompt = self.SYSTEM_PROMPT
        
        if round_number == 2 and previous_results:
            return base_prompt + f"""

This is Round 2 of your analysis. You have already gathered some information in Round 1.

Based on the information from Round 1, make additional tool calls ONLY if they would meaningfully improve your answer. Consider:
- Are there gaps in the information?
- Would different search terms or filters help?
- Does the query have multiple parts not yet addressed?

If the Round 1 results are sufficient, you may choose not to make additional tool calls and provide your final response."""
        
        return base_prompt