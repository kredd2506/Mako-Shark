import os
import json
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from openai import OpenAI

class NRPModel:
    def __init__(self, client):
        self.client = client
        self.tools = []
    def bind_tools(self, tools):
        self.tools = tools
        return self
    def _convert_tool_to_openai_format(self, tool):
        """Convert LangChain tool to OpenAI tool format"""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema.model_json_schema() if tool.args_schema else {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    def invoke(self, messages):
        # Convert messages to proper format
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                if msg.__class__.__name__ == "SystemMessage":
                    formatted_messages.append({"role": "system", "content": msg.content})
                elif msg.__class__.__name__ == "HumanMessage":
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif msg.__class__.__name__ == "AIMessage":
                    formatted_messages.append({"role": "assistant", "content": msg.content})
                elif msg.__class__.__name__ == "ToolMessage":
                    # Truncate tool message content if too long
                    content = str(msg.content)
                    if len(content) > 2000:
                        content = content[:2000] + "\n[Content truncated...]"
                    formatted_messages.append({
                        "role": "tool", 
                        "content": content,
                        "tool_call_id": getattr(msg, 'tool_call_id', 'unknown')
                    })
            else:
                formatted_messages.append(msg)
        # Convert tools to OpenAI format
        openai_tools = None
        if self.tools:
            openai_tools = [self._convert_tool_to_openai_format(t) for t in self.tools]
        try:
            print(f"Sending request to OpenAI with {len(formatted_messages)} messages")
            response = self.client.chat.completions.create(
                model="gemma3",
                temperature=0,
                messages=formatted_messages,
                tool_choice="auto" if openai_tools else None,
                tools=openai_tools,
                timeout=30,  # Add timeout to prevent hanging
            )
            print("Received response from OpenAI")
            choice = response.choices[0].message
            tool_calls = []
            if hasattr(choice, "tool_calls") and choice.tool_calls:
                for t in choice.tool_calls:
                    args = t.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    tool_calls.append({
                        "name": t.function.name,
                        "args": args,
                        "id": t.id
                    })
            return AIMessage(
                content=choice.content or "",
                tool_calls=tool_calls
            )
        except Exception as e:
            print(f"Error calling OpenAI: {str(e)}")
            return AIMessage(content=f"Error calling model: {str(e)}")

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]



# Create a simplified agent class
class SimpleAgent:
    def __init__(self, model, tools, system=""):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.system = system
        self.max_iterations = 3
    
    def run(self, question):
        """Run the agent with a simple loop"""
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": question})
        
        for i in range(self.max_iterations):
            print(f"Iteration {i+1}")
            
            # Get model response
            try:
                response = self.model._make_openai_call(messages)
                print(f"Model response: {response.content}")
                messages.append({"role": "assistant", "content": response.content})
                
                # Check if there are tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_id = tool_call["id"]
                        
                        print(f"Calling tool: {tool_name} with args: {tool_args}")
                        
                        if tool_name in self.tools:
                            try:
                                tool_result = self.tools[tool_name].invoke(tool_args)
                                print(f"Tool result: {tool_result[:100]}...")
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "content": str(tool_result)
                                })
                            except Exception as e:
                                print(f"Tool error: {str(e)}")
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "content": f"Error: {str(e)}"
                                })
                        else:
                            print(f"Unknown tool: {tool_name}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": f"Unknown tool: {tool_name}"
                            })
                else:
                    # No tool calls, return the response
                    return response.content
            except Exception as e:
                print(f"Error in iteration {i+1}: {str(e)}")
                return f"Error: {str(e)}"
        
        # Return the last response if we reached max iterations
        return messages[-1]["content"]

# Add this method to NRPModel
def _make_openai_call(self, messages):
    """Make a direct OpenAI call"""
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            formatted_messages.append(msg)
        else:
            # Handle LangChain message objects
            if hasattr(msg, 'content'):
                role = "user"
                if hasattr(msg, 'type'):
                    role = msg.type
                elif msg.__class__.__name__ == "SystemMessage":
                    role = "system"
                elif msg.__class__.__name__ == "AIMessage":
                    role = "assistant"
                elif msg.__class__.__name__ == "ToolMessage":
                    role = "tool"
                
                formatted_messages.append({"role": role, "content": msg.content})
    
    # Convert tools to OpenAI format
    openai_tools = None
    if self.tools:
        openai_tools = [self._convert_tool_to_openai_format(t) for t in self.tools]
    
    try:
        response = self.client.chat.completions.create(
            model="gemma3",
            temperature=0,
            messages=formatted_messages,
            tool_choice="auto" if openai_tools else None,
            tools=openai_tools,
            timeout=30,
        )
        choice = response.choices[0].message
        tool_calls = []
        if hasattr(choice, "tool_calls") and choice.tool_calls:
            for t in choice.tool_calls:
                args = t.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append({
                    "name": t.function.name,
                    "args": args,
                    "id": t.id
                })
        return AIMessage(
            content=choice.content or "",
            tool_calls=tool_calls
        )
    except Exception as e:
        return AIMessage(content=f"Error calling model: {str(e)}")

# Add the method to the NRPModel class
NRPModel._make_openai_call = _make_openai_call

class Agent:
    def __init__(self, model, tools, system: str = ""):
        self.system = system
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        self.max_iterations = 5  # Prevent infinite loops
        self.current_iteration = 0
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.raw_graph = graph
        self.graph = graph.compile()
    
    def exists_action(self, state: AgentState) -> bool:
        """Check if the last message has tool calls and we haven't exceeded max iterations"""
        if self.current_iteration >= self.max_iterations:
            print(f"⚠️ Reached max iterations ({self.max_iterations}). Stopping.")
            return False
            
        try:
            result = state["messages"][-1]
            return (hasattr(result, "tool_calls") and 
                    result.tool_calls is not None and 
                    len(result.tool_calls) > 0)
        except (IndexError, KeyError, AttributeError):
            return False
    
    def call_openai(self, state: AgentState) -> dict:
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}
    
    def take_action(self, state: AgentState) -> dict:
        self.current_iteration += 1
        tool_calls = state["messages"][-1].tool_calls
        results = []
        
        for t in tool_calls:
            tool_name = t["name"]
            tool_args = t["args"]
            print(f"🔧 Calling tool: {tool_name} with args: {tool_args}")
            
            if tool_name not in self.tools:
                result = "❌ Tool name not recognized. Available tools: " + ", ".join(self.tools.keys())
            else:
                try:
                    result = self.tools[tool_name].invoke(tool_args)
                    # Ensure result is string and truncate if needed
                    result = str(result)
                    if len(result) > 3000:
                        result = result[:3000] + "\n\n[Output truncated to prevent loops]"
                except Exception as e:
                    result = f"❌ Tool error: {str(e)}"
            
            results.append(ToolMessage(tool_call_id=t["id"], name=tool_name, content=result))
        
        print("✅ Tool(s) executed. Returning to model.")
        return {"messages": results}
    
    def reset_iteration_counter(self):
        """Reset the iteration counter for a new query"""
        self.current_iteration = 0