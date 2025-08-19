import os
import gradio as gr
import time
import sys
import importlib
import json
from io import StringIO
from openai import OpenAI
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from dotenv import load_dotenv
from tabulate import tabulate

# Load environment variables
load_dotenv()

# Check dependencies
def check_dependencies():
    required_modules = [
        'gradio',
        'langchain_core',
        'openai',
        'requests',
        'bs4',
        'sklearn',
        'kubernetes',
        'tabulate',
        'dotenv',
        'numpy'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing required modules: {', '.join(missing_modules)}")
        print("Please install them by running:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("✅ All dependencies are installed!")

check_dependencies()

# Import our modules
from agent.knowledge_base import DocumentationKnowledgeBase
from agent.agent import NRPModel
from agent.monitoring import describe_pods, namespace_gpu_utilization, fetch_dcgm_gpu_util_data, analyze_dcgm_gpu_data

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("NRP_API_KEY"),
    base_url="https://llm.nrp-nautilus.io/"
)

# Initialize Documentation Knowledge Base
print("Initializing Documentation Knowledge Base...")
doc_kb = DocumentationKnowledgeBase(api_key=os.environ.get("NRP_API_KEY"))
kb_file = "nrp_expert_docs_kb.json"

if os.path.exists(kb_file):
    print(f"Loading knowledge base from {kb_file}...")
    doc_kb.load_knowledge_base(kb_file)
else:
    print(f"Knowledge base file {kb_file} not found. Building it now...")
    doc_kb.crawl_documentation("https://nrp.ai/documentation/", max_depth=1)
    doc_kb.build_knowledge_base()
    doc_kb.save_knowledge_base(kb_file)

# Helper to capture and truncate printed output
def capture_stdout_truncated(func, max_length=2000, *args, **kwargs):
    """Capture stdout and truncate if too long to prevent LLM loops"""
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    
    output = mystdout.getvalue()
    if len(output) > max_length:
        output = output[:max_length] + f"\n\n... [Output truncated - showing first {max_length} characters]"
    return output

# Define tools directly in app.py to avoid circular imports
@tool
def search_documentation(query: str) -> str:
    """Search the NRP.ai documentation for relevant information."""
    if not doc_kb.documents:
        return "❌ Knowledge base not loaded. Cannot search documentation."
    
    results = doc_kb.search(query, top_k=3)
    if not results:
        return "❌ No relevant documentation found."
    
    output = []
    for i, result in enumerate(results, 1):
        output.append(f"Result {i}:")
        output.append(f"Title: {result['title']}")
        output.append(f"URL: {result['url']}")
        output.append(f"Content: {result['text'][:200]}...")
        output.append("")  # Empty line
    
    return "\n".join(output)

@tool
def describe_pods_tool(namespace: str = "gsoc") -> str:
    """Describe pods in a given Kubernetes namespace. Defaults to 'gsoc'."""
    return capture_stdout_truncated(describe_pods, 1500, namespace=namespace)

@tool
def namespace_gpu_util_tool(threshold: float = 0.0) -> str:
    """Get average GPU utilization per namespace with optional threshold filter."""
    return capture_stdout_truncated(namespace_gpu_utilization, 1500, threshold=threshold)

@tool
def dcgm_gpu_inspect_tool(threshold: float = 0.0) -> str:
    """Inspect raw GPU usage with model name, host, pod, and utilization. Shows top 10 GPUs above threshold."""
    data = fetch_dcgm_gpu_util_data()
    if not data:
        return "⚠️ No GPU data available."
    filtered = [d for d in data if d["utilization"] >= threshold]
    if not filtered:
        return f"✅ No GPUs over {threshold}% utilization."
    top = sorted(filtered, key=lambda x: x["utilization"], reverse=True)[:10]
    rows = [
        [d["hostname"][:20], d["gpu_id"], d["model"][:25], f"{d['utilization']:.2f}%", d["namespace"], d["pod"][:20]]
        for d in top
    ]
    result = tabulate(rows, headers=["Host", "GPU", "Model", "Util%", "Namespace", "Pod"], tablefmt="grid")
    result += f"\n\nShowing top 10 of {len(filtered)} GPUs above {threshold}% threshold."
    return result

@tool
def calculate_dcgm_gpu_stats(threshold: float = 0.0) -> str:
    """Analyze GPU utilization across nodes and return statistical breakdown."""
    data = fetch_dcgm_gpu_util_data()
    if not data:
        return "⚠️ No GPU data available."
    filtered = [d for d in data if d["utilization"] >= threshold]
    total = len(filtered)
    if total == 0:
        return f"✅ No GPUs over the threshold of {threshold}% utilization."
    avg_util = sum(d["utilization"] for d in filtered) / total
    maxed = [d for d in filtered if d["utilization"] >= 99.0]
    idle = [d for d in filtered if d["utilization"] < 1.0]
    moderate = [d for d in filtered if 1.0 <= d["utilization"] < 70.0]
    available = [d for d in filtered if d["utilization"] < 100.0]
    unique_models = set(d["model"] for d in filtered)
    unique_hosts = set(d["hostname"] for d in filtered)
    return f"""📊 GPU Utilization Stats (threshold: {threshold}%):
🔍 Total GPUs: {total}
📈 Average Utilization: {avg_util:.2f}%
🔴 Fully Utilized (>=99%): {len(maxed)}
🟢 Idle (<1%): {len(idle)}
⚙️ Moderate (1-70%): {len(moderate)}
💻 Unique Hosts: {len(unique_hosts)}
🧠 Unique Models: {len(unique_models)}
🧮 Available (<100%): {len(available)}"""

# System prompt combining documentation search and monitoring
system_prompt = """You are a Kubernetes monitoring assistant with access to NRP.ai documentation. 
Use these tools to answer questions:
- 'search_documentation': Search NRP.ai documentation for conceptual information
- 'describe_pods_tool': View pod/container info in a namespace
- 'namespace_gpu_util_tool': View average GPU utilization per namespace  
- 'dcgm_gpu_inspect_tool': View detailed GPU metrics (top 10 results)
- 'calculate_dcgm_gpu_stats': Get statistical breakdown of GPU usage

Guidelines:
1. For conceptual questions about Kubernetes, GPU usage, or cloud computing, use 'search_documentation'
2. For current cluster status or resource utilization questions, use the monitoring tools
3. Only call each tool ONCE per question
4. Use the tool output to provide a direct answer
5. Do not repeat tool calls
6. For complex queries, break them down into multiple tool calls if needed"""

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

# Add method to NRPModel
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

# Create model and agent
model = NRPModel(client)
tools = [
    search_documentation,
    describe_pods_tool,
    namespace_gpu_util_tool,
    dcgm_gpu_inspect_tool,
    calculate_dcgm_gpu_stats
]
simple_agent = SimpleAgent(model, tools, system_prompt)

# Define the query function
def ask_agent(question):
    """Ask the agent a question and get a response"""
    return simple_agent.run(question)

# Create the Gradio interface
with gr.Blocks(title="Kubernetes Monitoring Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Kubernetes Monitoring Assistant")
    gr.Markdown("Ask questions about your Kubernetes cluster, GPU utilization, or search the NRP.ai documentation.")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                elem_id="chatbot",
                type="messages",  # Fix deprecation warning
                avatar_images=("👤", "🤖")
            )
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask about Kubernetes, GPU utilization, or search documentation...",
                lines=2,
                max_lines=5
            )
            with gr.Row():
                submit = gr.Button("Submit", variant="primary")
                clear = gr.Button("Clear")
        
        with gr.Column(scale=1):
            gr.Markdown("### Example Questions")
            gr.Examples(
                examples=[
                    "How does GPU scheduling work in Kubernetes?",
                    "What's the current GPU utilization across all namespaces?",
                    "Show me detailed GPU statistics and identify any idle GPUs",
                    "I'm having issues with GPU scheduling. Can you check current utilization and explain best practices?",
                    "Analyze all A100 GPUs in the cluster. Show utilization, availability, and which namespaces are using them"
                ],
                inputs=msg
            )
            
            gr.Markdown("### Agent Status")
            status = gr.Textbox(
                label="Status",
                value="Ready",
                interactive=False
            )
    
    def user(user_message, history):
        """Add user message to chat history"""
        # Convert to new format if needed
        if history and isinstance(history[0], list):
            # Convert old format to new
            new_history = []
            for item in history:
                if item[0]:  # User message
                    new_history.append({"role": "user", "content": item[0]})
                if item[1]:  # Assistant message
                    new_history.append({"role": "assistant", "content": item[1]})
            history = new_history
        
        # Add new user message
        history.append({"role": "user", "content": user_message})
        return "", history
    
    def bot(history):
        """Process user message and generate bot response"""
        user_message = history[-1]["content"]
        
        # Update status
        status.value = "Thinking..."
        
        # Add a temporary thinking message
        history.append({"role": "assistant", "content": "Thinking..."})
        yield history
        
        try:
            print(f"Processing user message: {user_message}")
            
            # Get response from agent
            response = ask_agent(user_message)
            print(f"Agent response: {response}")
            
            # Replace the thinking message with the actual response
            history[-1] = {"role": "assistant", "content": response}
            
            yield history
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            history[-1] = {"role": "assistant", "content": error_msg}
            yield history
        finally:
            # Reset status
            status.value = "Ready"
    
    def clear_history():
        """Clear chat history"""
        return []
    
    # Event handlers
    msg.submit(user, [msg, chatbot], [msg, chatbot]).then(bot, chatbot, chatbot)
    submit.click(user, [msg, chatbot], [msg, chatbot]).then(bot, chatbot, chatbot)
    clear.click(clear_history, outputs=[chatbot])

# Launch the app
if __name__ == "__main__":
    demo.launch()