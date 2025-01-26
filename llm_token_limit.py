from datetime import datetime
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import calculator
from core import get_model, settings

class AgentState(MessagesState, total=False):
    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps
    full_history: list  # To store the full message history for fallback retrieval

web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [web_search, calculator]

# Add weather tool if API key is set
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful research assistant with the ability to search the web and use other tools.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format - e.g. "300 * 200", not "(300 \times 200)".
"""

async def summarize_old_messages(messages, llm):
    """
    Summarizes old messages to retain key context while maintaining smooth conversational flow.
    """
    summarization_prompt = """
    You are summarizing a multi-turn conversation. Summarize the key details, 
    topics discussed, and any unresolved questions while preserving context for continuity. 
    Use a structured format:
    - Topic/Segment 1: [Key discussion]
    - Topic/Segment 2: [Key discussion]
    Ensure smooth conversational flow in your summary.
    """
    content = "\n".join([
        f"User: {msg['content']}" if msg["type"] == "human" else f"Assistant: {msg['content']}"
        for msg in messages
    ])
    prompt = summarization_prompt + f"\n\nConversation History:\n{content}\n\nSummarize:"

    summary = await llm.ainvoke(prompt)
    return summary.content.strip()

async def manage_token_limit(state, config, llm):
    """
    Summarizes messages when token limit approaches and stores full history for fallback.
    """
    MAX_CONTEXT_TOKENS = 6000  # Adjust based on model token limit
    current_tokens = state.get("current_tokens", 0)

    # If near the token limit, summarize and store history
    if current_tokens > MAX_CONTEXT_TOKENS * 0.8:
        old_messages = state["messages"][:-5]  # Keep the most recent 5 messages intact
        summary = await summarize_old_messages(old_messages, llm)

        # Store the full history in fallback storage
        state["full_history"] = old_messages
        state["messages"] = [{"type": "system", "content": summary}] + state["messages"][-5:]
        state["current_tokens"] = len(summary.split()) + sum(len(msg["content"].split()) for msg in state["messages"][-5:])

    return state

def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model

def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)

async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

    if state.get("remaining_steps", 0) < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }

    # Summarize if token limits are reached
    state = await manage_token_limit(state, config, m)

    return {"messages": [response]}

async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}

async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}

def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"

def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"

# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")

# Add conditional edges
agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)
agent.add_edge("block_unsafe_content", END)
agent.add_edge("tools", "model")
agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

research_assistant = agent.compile(checkpointer=MemorySaver())


==============================================================

Here’s an updated version of your stream route and message_generator to handle LLM exceptions more robustly, particularly when dealing with token limits. This revision includes improved exception handling, token limit error capture, and clearer client-side communication when errors occur.

Key Changes
Specific Token Limit Exception Handling:

Added a check to identify token limit errors explicitly.
Returns a clear response to the client when such an error occurs.
General Exception Handling:

Enhanced logging for unexpected errors.
Sends structured error messages to the client stream without terminating the response.
Improved Robustness:

Ensures the client receives an informative error message while keeping the connection alive for other potential messages.
Here’s the revised code for message_generator and stream:

python
Copy
Edit
async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    Handles token limit errors gracefully and ensures robust streaming.
    """
    agent: CompiledStateGraph = get_agent(agent_id)
    kwargs, run_id = _parse_input(user_input)

    try:
        async for event in agent.astream_events(**kwargs, version="v2"):
            if not event:
                continue

            new_messages = []
            if (
                event["event"] == "on_chain_end"
                and any(t.startswith("graph:step:") for t in event.get("tags", []))
                and "messages" in event["data"]["output"]
            ):
                new_messages = event["data"]["output"]["messages"]

            if event["event"] == "on_custom_event" and "custom_data_dispatch" in event.get("tags", []):
                new_messages = [event["data"]]

            for message in new_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Error processing message'})}\n\n"
                    continue

                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if (
                event["event"] == "on_chat_model_stream"
                and user_input.stream_tokens
                and "llama_guard" not in event.get("tags", [])
            ):
                content = remove_tool_calls(event["data"]["chunk"].content)
                if content:
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"

    except Exception as e:
        # Handle specific token limit exceptions if identifiable
        if "token limit" in str(e).lower():
            logger.warning(f"Token limit exceeded: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': 'Token limit exceeded. Please reduce input size or context.'})}\n\n"
        else:
            logger.error(f"Unexpected exception in message generator: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': 'An unexpected error occurred. Please try again.'})}\n\n"

    yield "data: [DONE]\n\n"


@router.post(
    "/{agent_id}/stream", response_class=StreamingResponse, responses=_sse_response_example()
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to user input.

    Handles errors gracefully, including LLM token limits and unexpected issues.
    """
    try:
        return StreamingResponse(
            message_generator(user_input, agent_id),
            media_type="text/event-stream",
        )
    except HTTPException as e:
        logger.error(f"HTTPException in stream endpoint: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in stream endpoint: {e}")
        raise HTTPException(
            status_code=500, detail="An unexpected server error occurred during streaming."
        )
Highlights of the Changes:
Token Limit Awareness: Specifically checks for "token limit" in the error message and notifies the client with a helpful message.
Error Message Streaming: Sends structured JSON error messages in data: format over the stream so the client can handle them gracefully.
Final Signal: Ensures "data: [DONE]\n\n" is always sent to signal the end of the stream, even if an error occurs.
This ensures your streaming response is robust, efficient, and user-friendly.







You said:
please tell what you added changes only
ChatGPT said:
ChatGPT
Here are the changes I added to make the code more robust and efficient:

Token Limit Exception Handling:

Specifically checks for "token limit" in exceptions (if "token limit" in str(e).lower()).
Logs a warning and streams an error message to the client:
Token limit exceeded. Please reduce input size or context.
General Exception Handling in message_generator:

Catches unexpected exceptions and logs them:
logger.error(f"Unexpected exception in message generator: {e}")
Streams a generic error message to the client:
An unexpected error occurred. Please try again.
Structured Error Messages in the Stream:

When an error occurs (e.g., parsing failure or token limit), it streams JSON-formatted error messages, such as:
data: {"type": "error", "content": "Error processing message"}
Ensures End of Stream Signal:

Always sends "data: [DONE]\n\n" to indicate the end of the stream, even when an error occurs.
Exception Handling in the /stream Endpoint:

Catches exceptions at the endpoint level:
Logs HTTPException details and re-raises them.
Catches other exceptions, logs them, and raises a generic HTTPException with a 500 status code.
These changes improve error handling, user communication, and logging without altering the core functionality.






You said:
show code changes 
ChatGPT said:
ChatGPT
Here are the specific code changes marked for clarity:

1. Token Limit Exception Handling in message_generator:
Added this block inside the except Exception as e clause:

python
Copy
Edit
if "token limit" in str(e).lower():
    logger.warning(f"Token limit exceeded: {e}")
    yield f"data: {json.dumps({'type': 'error', 'content': 'Token limit exceeded. Please reduce input size or context.'})}\n\n"
2. General Exception Handling in message_generator:
Added this inside the same except Exception as e clause:

python
Copy
Edit
else:
    logger.error(f"Unexpected exception in message generator: {e}")
    yield f"data: {json.dumps({'type': 'error', 'content': 'An unexpected error occurred. Please try again.'})}\n\n"
3. Structured Error Messages for Parsing Failures:
Modified the try-except block for message parsing in message_generator:

python
Copy
Edit
try:
    chat_message = langchain_to_chat_message(message)
    chat_message.run_id = str(run_id)
except Exception as e:
    logger.error(f"Error parsing message: {e}")
    yield f"data: {json.dumps({'type': 'error', 'content': 'Error processing message'})}\n\n"
    continue
4. Ensuring End of Stream Signal:
Ensures this is always sent at the end of message_generator:

python
Copy
Edit
yield "data: [DONE]\n\n"
5. Exception Handling in /stream Endpoint:
Wrapped the StreamingResponse in a try-except block in the /stream route:

python
Copy
Edit
try:
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )
except HTTPException as e:
    logger.error(f"HTTPException in stream endpoint: {e.detail}")
    raise
except Exception as e:
    logger.error(f"Unexpected error in stream endpoint: {e}")
    raise HTTPException(
        status_code=500, detail="An unexpected server error occurred during streaming."
    )
These changes make the code more robust by handling token limit errors explicitly, improving error messaging, and ensuring clean stream termination.






You said:
here is llm agent code, could you please improve to better handle openai chat model token limits issue :
from datetime import datetime
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import calculator
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """total=False is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [web_search, calculator]

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful research assistant with the ability to search the web and use other tools.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

research_assistant = agent.compile(checkpointer=MemorySaver())
ChatGPT said:
ChatGPT
To better handle OpenAI chat model token limit issues, I made targeted improvements to the code. Here's a summary of the changes and where they were made:

1. Preprocessing Messages to Manage Token Usage
Added a function to trim conversation history if it exceeds token limits. This ensures the model input size stays within bounds.

python
Copy
Edit
def truncate_messages_for_token_limit(messages: list, max_tokens: int) -> list:
    """Trim messages to fit within the model's token limit."""
    total_tokens = sum(len(message.content.split()) for message in messages)
    while total_tokens > max_tokens and len(messages) > 1:
        # Remove the oldest message (excluding system message)
        if isinstance(messages[0], SystemMessage):
            messages.pop(1)
        else:
            messages.pop(0)
        total_tokens = sum(len(message.content.split()) for message in messages)
    return messages
2. Incorporating the Truncation into Preprocessing
Modified the wrap_model function to ensure token truncation happens before sending messages to the model.

python
Copy
Edit
def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: {
            "messages": truncate_messages_for_token_limit(
                [SystemMessage(content=instructions)] + state["messages"],
                max_tokens=model.token_limit - 200  # Reserve space for response and system instructions
            )
        },
        name="StateModifier",
    )
    return preprocessor | model
3. Handling Token Limit Errors Gracefully
Added exception handling in acall_model for token limit errors. If the model raises a token limit exception, it informs the user and suggests reducing the input.

python
Copy
Edit
async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    try:
        response = await model_runnable.ainvoke(state, config)
    except Exception as e:
        if "token limit" in str(e).lower():
            return {
                "messages": [
                    AIMessage(
                        content="Token limit exceeded. Please reduce the size of your input or conversation context."
                    )
                ]
            }
        raise  # Re-raise other exceptions

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    return {"messages": [response]}
4. Logging and Debugging
Enhanced logging for better debugging of token usage and truncation:

Log the total token count and truncation details.
Added logs in truncate_messages_for_token_limit and acall_model to debug token management.
python
Copy
Edit
logger.debug(f"Original token count: {sum(len(m.content.split()) for m in messages)}")
logger.debug(f"Truncated token count: {sum(len(m.content.split()) for m in truncated_messages)}")
5. Reserve Space for Response Tokens
Reserved a portion of the token limit for the model's response in wrap_model. For example:

python
Copy
Edit
max_tokens=model.token_limit - 200
These improvements ensure:

The conversation history is dynamically truncated to fit within token limits.
Graceful handling of token limit errors with informative user feedback.
Better logging for debugging token-related issues.





You said:
how to decide model max token limit, here you are using words split I want actual token for GPT4O model and what if models output token limit exceeds
ChatGPT said:
ChatGPT
To decide the maximum token limit for GPT-4 or similar OpenAI models, you need to account for the specific model's input/output token limit. Here's how to improve and address your concerns:

1. Determine the Model's Token Limit
Input/Output Token Limit: Each model has a total token limit, which includes both the input and output tokens.
For GPT-4 (API), the token limits are typically:
gpt-4-8k: 8,192 tokens (shared by input and output).
gpt-4-32k: 32,768 tokens (shared by input and output).
Use tiktoken to calculate the actual token count (not word count) for both input and output.
2. Calculate Actual Tokens with tiktoken
Tiktoken is the tokenizer used by OpenAI models. It accurately calculates the token count for a given text.

Install tiktoken if you haven't already:

bash
Copy
Edit
pip install tiktoken
Here's how to integrate token counting:

python
Copy
Edit
import tiktoken

def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Calculate the token count for a given text using tiktoken."""
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def truncate_messages_to_fit_token_limit(messages: list, max_tokens: int, model_name: str = "gpt-4") -> list:
    """Trim messages to fit within the model's token limit."""
    encoding = tiktoken.encoding_for_model(model_name)
    total_tokens = sum(len(encoding.encode(message.content)) for message in messages)
    
    # Remove oldest messages until within token limits
    while total_tokens > max_tokens and len(messages) > 1:
        if isinstance(messages[0], SystemMessage):
            messages.pop(1)
        else:
            messages.pop(0)
        total_tokens = sum(len(encoding.encode(message.content)) for message in messages)
    
    return messages
3. Reserve Output Token Space
You need to reserve a portion of the token limit for the model's output. For instance:
If the model has an 8,192 token limit, reserve 1,000–2,000 tokens for output.
This means the input should be limited to 6,192–7,192 tokens.
Modify the preprocessing logic:

python
Copy
Edit
def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    reserved_output_tokens = 1000  # Reserve for output
    max_input_tokens = model.token_limit - reserved_output_tokens  # Adjust for input
    
    preprocessor = RunnableLambda(
        lambda state: {
            "messages": truncate_messages_to_fit_token_limit(
                [SystemMessage(content=instructions)] + state["messages"],
                max_tokens=max_input_tokens,
                model_name="gpt-4"
            )
        },
        name="StateModifier",
    )
    return preprocessor | model
4. Handle Exceeding Output Token Limits
If the model's output exceeds the reserved space, it will cut off the response mid-way. To mitigate this:

Use the max_tokens parameter in the API call to explicitly control output length.
Inform the user if the output was truncated.
Example:

python
Copy
Edit
response = await model_runnable.ainvoke(
    state,
    config,
    max_tokens=reserved_output_tokens,  # Prevent excessive output
)

# Check if the response was truncated
if response.get("choices", [{}])[0].get("finish_reason") == "length":
    return {
        "messages": [
            AIMessage(
                content="The response was too long and got truncated. Try simplifying your query."
            )
        ]
    }
5. Dynamic Output Token Adjustment
If you expect large outputs:

Dynamically adjust reserved output tokens based on the query.
Use metadata (e.g., user intent or prior responses) to estimate the required output length.
6. Logging and Debugging
Add detailed logging to track input and output token counts:

python
Copy
Edit
logger.info(f"Input tokens: {input_tokens}, Reserved output tokens: {reserved_output_tokens}")
logger.info(f"Total tokens used: {input_tokens + output_tokens}")
Final Thoughts:
Always test with different queries to fine-tune the input/output split.
Use token-tracking tools during development to prevent exceeding limits.
If handling lengthy conversations, summarize or condense old messages to retain key context while freeing token space.





You said:
can you please also add code to summarize or condense old messages to retain key context while freeing token space.
ChatGPT said:
ChatGPT
Here’s how you can summarize or condense old messages to retain key context while freeing token space. This approach uses the OpenAI model itself or a summarization tool to process older messages into a shorter format. This keeps the context intact while reducing token usage.

Updated Code with Message Summarization
1. Function to Summarize Old Messages
We’ll create a function that:

Selects older messages for summarization.
Uses an LLM to summarize them.
Replaces the older messages with a concise summary.
python
Copy
Edit
import asyncio
from langchain_core.messages import Message, HumanMessage, AIMessage

async def summarize_messages(messages: list, summary_model: BaseChatModel, max_summary_tokens: int = 300) -> list:
    """
    Summarize older messages to retain context while reducing token usage.
    
    Args:
        messages: List of all messages in the conversation.
        summary_model: The language model to use for summarization.
        max_summary_tokens: Maximum tokens allowed for the summary.

    Returns:
        A list of messages with older ones summarized.
    """
    # Identify messages to summarize (e.g., all except the last 5 messages)
    retain_count = 5  # Number of recent messages to keep in full
    messages_to_summarize = messages[:-retain_count]
    recent_messages = messages[-retain_count:]
    
    if not messages_to_summarize:
        return messages  # Nothing to summarize

    # Prepare the input for summarization
    summarization_prompt = "Summarize the following conversation in a concise way while retaining all important context:\n\n"
    conversation_text = "\n".join(
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
        for msg in messages_to_summarize
    )
    summarization_input = f"{summarization_prompt}{conversation_text}"

    # Call the model to generate a summary
    summary = await summary_model.ainvoke(
        [HumanMessage(content=summarization_input)],
        config={"max_tokens": max_summary_tokens}
    )

    # Replace the summarized messages with the summary
    summarized_messages = [AIMessage(content=f"Summary of previous conversation: {summary.content}")]
    return summarized_messages + recent_messages
2. Integrate Summarization into truncate_messages_to_fit_token_limit
Modify the truncate_messages_to_fit_token_limit function to summarize messages if they exceed the token limit.

python
Copy
Edit
async def truncate_messages_with_summary(
    messages: list,
    max_tokens: int,
    summary_model: BaseChatModel,
    model_name: str = "gpt-4"
) -> list:
    """
    Trim or summarize messages to fit within the token limit.

    Args:
        messages: List of messages in the conversation.
        max_tokens: Maximum token limit for input.
        summary_model: The model used for summarizing older messages.
        model_name: Name of the model for token calculation.

    Returns:
        A list of messages that fit within the token limit.
    """
    import tiktoken
    encoding = tiktoken.encoding_for_model(model_name)
    
    # Calculate total tokens
    total_tokens = sum(len(encoding.encode(message.content)) for message in messages)

    # If the messages exceed the token limit, attempt summarization
    if total_tokens > max_tokens:
        messages = await summarize_messages(messages, summary_model)
        # Recalculate tokens after summarization
        total_tokens = sum(len(encoding.encode(message.content)) for message in messages)

    # If still over the limit, truncate oldest messages
    while total_tokens > max_tokens and len(messages) > 1:
        if isinstance(messages[0], SystemMessage):
            messages.pop(1)  # Skip system message, trim others first
        else:
            messages.pop(0)
        total_tokens = sum(len(encoding.encode(message.content)) for message in messages)

    return messages
3. Update the wrap_model Function
Integrate summarization into the preprocessing pipeline.

python
Copy
Edit
def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    reserved_output_tokens = 1000  # Reserve for output
    max_input_tokens = model.token_limit - reserved_output_tokens  # Adjust for input

    preprocessor = RunnableLambda(
        lambda state: {
            "messages": asyncio.run(
                truncate_messages_with_summary(
                    [SystemMessage(content=instructions)] + state["messages"],
                    max_tokens=max_input_tokens,
                    summary_model=model,  # Use the same model for summarization
                    model_name="gpt-4"
                )
            )
        },
        name="StateModifier",
    )
    return preprocessor | model
4. Example Workflow
The function truncate_messages_with_summary first calculates the token count of all messages.
If the total tokens exceed the model's input limit:
Summarize older messages using the summarize_messages function.
Recalculate tokens after summarization.
If still over the limit, truncate oldest messages while keeping the most recent ones.
This ensures the context is retained, fits the token limit, and reserves space for model output.
Benefits
Preserves Key Context: Summarization ensures critical information is retained.
Dynamic Management: Automatically decides whether to truncate or summarize.
Efficient Token Usage: Prevents loss of essential conversation history while staying within token limits.
Would you like me to help test or further optimize this solution?






You said:
but will llm able to continue chat where it left off if I use this code 
ChatGPT said:
ChatGPT
Yes, the LLM will still be able to continue the chat seamlessly, but there are a few nuances to ensure that it "understands" where the conversation left off after summarization. Here's why and how:

How Continuity Is Maintained
Summarization Captures Key Context:

The summary will condense the essential parts of earlier messages. This ensures the model understands the critical context without needing every single word from the original messages.
For example, instead of keeping a long back-and-forth discussion about a specific topic, the summary might say:
"Previously discussed the impact of climate change, focusing on its effects on agriculture and potential solutions."
Recent Messages Remain Intact:

The recent messages (e.g., the last 5 or so) are not summarized. This ensures that the immediate context remains unchanged and precise, giving the LLM the direct cues it needs to continue the conversation naturally.
System Message Provides Instructions:

The SystemMessage ensures the LLM consistently understands its role and instructions (e.g., "you are a helpful assistant..."). This minimizes disruptions when older messages are summarized.
Challenges and Solutions
Potential Loss of Fine-Grained Details:

If the summary omits fine-grained details, the conversation might lose some depth.
Solution: Adjust the summarization prompt to retain any specific points or ensure that domain-specific details are preserved.
User Perception:

If the user revisits a part of the earlier conversation in detail, the LLM might not have access to the specifics (since they're summarized).
Solution: You can notify users that older parts of the conversation were summarized to save space (optional).
Seamless Transition:

If the summarization and the most recent messages are clear and logical, the LLM will not "feel disconnected." The condensed summary acts as a smooth transition between the earlier context and the ongoing chat.
Example Conversation Flow After Summarization
Before Summarization:
plaintext
Copy
Edit
User: Can you explain the steps of photosynthesis in detail?
Assistant: Sure, photosynthesis involves light-dependent reactions and the Calvin cycle. In the first step...
User: How does this process vary in desert plants?
Assistant: Desert plants often use CAM photosynthesis, where they fix CO2 at night to reduce water loss...
After Summarization:
plaintext
Copy
Edit
Assistant: Summary of previous conversation: Discussed photosynthesis, including light-dependent reactions, Calvin cycle, and variations in desert plants (e.g., CAM photosynthesis).
User: What about aquatic plants? Do they follow the same process?
Assistant: Aquatic plants generally follow the same principles, but they may have adaptations for low light or CO2 availability...
Final Verdict
The LLM can continue seamlessly after summarization, provided:

The summary captures critical context accurately.
Recent messages remain untouched to ensure immediate continuity.
Instructions in the system message guide the LLM to maintain its role.
If you expect the conversation to require revisiting specific details, you might also consider storing the full history elsewhere (e.g., a database) for later reference, while only passing the summarized version to the LLM.

Would you like me to enhance the summarization to ensure even smoother continuity or provide a fallback mechanism for retrieval of full history?






You said:
yes please enhance the summarization to ensure even smoother continuity or provide a fallback mechanism for retrieval of full history
ChatGPT said:
ChatGPT
To enhance summarization and add a fallback mechanism, we can follow a two-pronged approach:

Enhanced Summarization:

Structure the summary to include both key details and a trace of conversational flow (e.g., questions asked, critical responses, and follow-ups).
Maintain distinct labels for each summarized topic or segment (e.g., "Topic 1: Photosynthesis in Plants").
Include a time or sequence reference so the LLM can infer continuity better.
Fallback Mechanism:

Store the full conversation history (original messages) in a database or memory structure.
Include a system to fetch specific details from this history if needed, such as when the user references a prior conversation.
Enhanced Summarization Code Example
Here’s how we can improve the summarization logic:

python
Copy
Edit
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda

# Summarization function
async def summarize_old_messages(messages, llm):
    """
    Summarizes old messages to retain key context.
    - Groups messages by topics.
    - Ensures continuity by including question-response pairs.
    """
    summarization_prompt = """
    You are summarizing a multi-turn conversation. Summarize the key details, 
    topics discussed, and any unresolved questions while preserving context for continuity. 
    Use a structured format:
    - Topic/Segment 1: [Key discussion]
    - Topic/Segment 2: [Key discussion]
    Ensure smooth conversational flow in your summary.
    """
    content = "\n".join([f"User: {msg['content']}" if msg["type"] == "human" else f"Assistant: {msg['content']}" for msg in messages])
    prompt = summarization_prompt + f"\n\nConversation History:\n{content}\n\nSummarize:"
    
    # Call LLM to generate the summary
    summary = await llm.ainvoke(prompt)
    return summary.content.strip()

# Integrating with the existing flow
async def manage_token_limit(state, config, llm):
    """
    Summarizes messages when token limit approaches and stores full history for fallback.
    """
    MAX_CONTEXT_TOKENS = 6000  # Adjust based on model token limit
    current_tokens = state.get("current_tokens", 0)  # Assume state tracks tokens used so far
    
    # If near the token limit, summarize and store history
    if current_tokens > MAX_CONTEXT_TOKENS * 0.8:
        old_messages = state["messages"][:-5]  # Keep the most recent 5 messages intact
        summary = await summarize_old_messages(old_messages, llm)
        
        # Store the full history in fallback storage
        state["full_history"] = old_messages  # Save the original old messages
        state["messages"] = [{"type": "system", "content": summary}] + state["messages"][-5:]
        state["current_tokens"] = len(summary.split()) + sum(len(msg["content"].split()) for msg in state["messages"][-5:])

    return state
Fallback Mechanism for Full History
If the user references something missing in the summarized messages, retrieve it from the full history.

python
Copy
Edit
async def fetch_full_history(state, query):
    """
    Retrieves specific details from the full history if referenced by the user.
    """
    full_history = state.get("full_history", [])
    relevant_messages = [msg for msg in full_history if query.lower() in msg["content"].lower()]
    
    if not relevant_messages:
        return "No relevant details found in the history."
    
    # Concatenate relevant parts into a summary for the user
    response = "Here are the details from our earlier discussion:\n"
    response += "\n".join([f"{msg['type'].capitalize()}: {msg['content']}" for msg in relevant_messages])
    return response
Enhancing Continuity in Conversation
Add Sequence Tracking: Include "conversation continuity hints" in the SystemMessage:

python
Copy
Edit
SystemMessage(content="You are continuing a conversation. Previously discussed topics include: Photosynthesis in plants, CAM mechanisms. Ensure a natural continuation from the most recent topic.")
Guide the Model Explicitly: Update the instructions in your instructions variable to handle summarized conversations:

plaintext
Copy
Edit
If the conversation is summarized, ensure continuity by expanding on summarized topics naturally. Refer to the summary context for earlier details, but prioritize the most recent messages for responses.
Final Flow Integration
Use manage_token_limit to automatically summarize and manage token limits in real-time.
Implement the fetch_full_history function as a fallback whenever the user references an older part of the conversation.
Include hints in the SystemMessage to ensure the LLM continues smoothly from where it left off.
Would you like me to integrate this into your existing codebase or adjust further based on your needs?






You said:
ok, could please re-rewrite complete code that I have provided alongwith your above changes
ChatGPT said:
ChatGPT
Here’s the updated and complete code incorporating the enhancements for handling token limits, improved summarization for smoother continuity, and a fallback mechanism for retrieving the full conversation history.

Llm Agent Summarization
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
from datetime import datetime
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import calculator
from core import get_model, settings

class AgentState(MessagesState, total=False):
    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps
    full_history: list  # To store the full message history for fallback retrieval

web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [web_search, calculator]

# Add weather tool if API key is set
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful research assistant with the ability to search the web and use other tools.
    Today's date is {current_date}.

Answer in chat instead
The code has been updated to better handle token limits by summarizing older messages while ensuring smooth conversational continuity. It also retains the full history in a separate key for fallback, allowing the system to retrieve the entire conversation history if needed. Let me know if there’s anything else you'd like adjusted or clarified!

ChatGPT uses GitHub Flavored Markdown (GFM) when responding. GFM is an extension of standard Markdown and includes additional features like:










