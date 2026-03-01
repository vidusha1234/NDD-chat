import json
from groq import Groq
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Groq's API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are a compassionate AI assistant supporting parents of children with neurodevelopmental delays.

CRITICAL RULE: For ANY question where a parent is asking for help, advice, or strategies — you MUST search the guidance documents first using the search tool. This includes questions about challenging behaviour, tantrums, crying, meltdowns, emotional regulation, communication, sensory issues, routines, school, therapy, milestones, and any parenting concern. Do NOT answer from general knowledge alone.

Search Tool Usage:
- ALWAYS search before answering a parent question
- Translate the parent's situation into search keywords that describe the underlying challenge (e.g. if a child cries over not getting a toy, search "challenging behaviour strategies" or "emotional regulation tantrum")
- One search per query maximum
- Base your answer primarily on what the documents contain

Response Protocol:
- List specific strategies or steps found in the documents, clearly numbered or bulleted
- Be compassionate — briefly acknowledge the parent's difficulty before listing strategies
- Use plain language; avoid jargon
- No meta-commentary — do not say "based on the search results" or describe your process

Always remind parents to consult qualified professionals (therapists, pediatricians, developmental specialists) for advice specific to their child.
"""

    def __init__(self, api_key: str, model: str):
        self.client = Groq(api_key=api_key)
        self.model = model

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert Anthropic-format tool definitions to OpenAI/Groq format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]  # rename input_schema -> parameters
                }
            }
            for tool in tools
        ]

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

        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]

        api_params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 800,
        }

        if tools:
            api_params["tools"] = self._convert_tools(tools)
            api_params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**api_params)

        # Handle tool execution if needed
        if response.choices[0].finish_reason == "tool_calls" and tool_manager:
            return self._handle_tool_execution(response, messages, tool_manager)

        return response.choices[0].message.content

    def _handle_tool_execution(self, initial_response, messages: List[Dict], tool_manager) -> str:
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool call requests
            messages: Current message history
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        assistant_message = initial_response.choices[0].message

        # Add assistant's tool call message to history
        messages.append(assistant_message)

        # Execute each tool call and collect results
        for tool_call in assistant_message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            tool_result = tool_manager.execute_tool(fn_name, **fn_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        # Get final response without tools
        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=800,
        )

        return final_response.choices[0].message.content
