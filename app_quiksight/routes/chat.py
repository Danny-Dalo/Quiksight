

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import datetime
import logging
import os
import pandas as pd
import json
import requests
import traceback


from app_quiksight.storage.redis import redis_client, save_chat_message, get_chat_history_openrouter
from api_training2.config import OPENROUTER_API_KEY, OPENROUTER_MODEL
from .upload import SYSTEM_INSTRUCTION, make_ai_context

logger = logging.getLogger("CHAT")


def log_section(title: str, char: str = "━"):
    line = char * 50
    logger.info(f"\n{line}\n  {title}\n{line}")

templates = Jinja2Templates(directory="app_quiksight/templates")
router = APIRouter()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ==============================================================================
#  TOOL DEFINITION
#  This is the spec the model sees. It decides when to call it — you execute it.
# ==============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code against the user's dataset to answer questions that require "
                "real computation: counts, sums, averages, filters, groupbys, comparisons, rankings, "
                "correlations, or anything you cannot determine from the context summary alone. "
                "Use this whenever the user asks a question that needs actual numbers from the data. "
                "Do NOT guess or estimate — run code and use real output. "
                "The dataframe is already loaded as `df`. The result of the last expression is returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Valid Python code. `df` (pandas DataFrame) and `pd` are pre-loaded. "
                            "Make the last line an expression whose value is the answer "
                            "(e.g. `df['sales'].sum()` or `df.groupby('region')['revenue'].mean()`). "
                            "For multi-line code, store your final answer in a variable called `result` "
                            "and make the last line just `result`."
                        )
                    }
                },
                "required": ["code"]
            }
        }
    }
]


# ==============================================================================
#  CODE EXECUTION SANDBOX
#  Runs model-generated code safely against the user's dataframe.
# ==============================================================================

def execute_python(code: str, df: pd.DataFrame) -> str:
    """
    Execute model-generated Python code against the user's dataframe.
    Returns the result as a string, or a formatted error message.
    """
    logger.info(f"\n   [TOOL] run_python called:\n{code}")

    # Give the model access to df and pandas only — nothing else
    local_env = {
        "df": df.copy(),  # copy so the model can't mutate the real df
        "pd": pd,
        "json": json,
        "result": None,
    }

    try:
        lines = code.strip().split("\n")

        # Split code into everything-but-last and the last line
        body = "\n".join(lines[:-1])
        last_line = lines[-1].strip()

        # Execute the body (if any)
        if body:
            exec(body, local_env)

        # Try to eval the last line to capture its return value
        try:
            output = eval(last_line, local_env)
        except SyntaxError:
            # Last line is a statement (e.g. assignment), not an expression — exec it
            exec(last_line, local_env)
            output = local_env.get("result", "Code executed. No return value.")

        # Convert output to a readable string for the model
        if isinstance(output, pd.DataFrame):
            return output.to_string(index=True, max_rows=50)
        elif isinstance(output, pd.Series):
            return output.to_string(max_rows=50)
        else:
            return str(output)

    except Exception as e:
        error_msg = f"ExecutionError: {type(e).__name__}: {str(e)}"
        logger.warning(f"   [TOOL] Code execution failed: {error_msg}")
        # Return the error to the model — it can try to fix or explain it
        return error_msg


# ==============================================================================
#  OPENROUTER CALL (single helper so the agentic loop stays clean)
# ==============================================================================

def call_openrouter(messages: list) -> dict:
    response = requests.post(
        url=OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://127.0.0.1:8000",
            "X-Title": "Quiksight"
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "tools": TOOLS,          # Must be included in every request per OpenRouter docs
            "temperature": 0.0,
        },
        timeout=60
    )
    response.raise_for_status()
    print(response.json)
    return response.json()


# ==============================================================================
#  AGENTIC LOOP
#  Keeps calling the model until it stops requesting tool calls and gives a
#  final text response. The user only ever sees that final response.
# ==============================================================================

def run_agentic_loop(messages: list, df: pd.DataFrame, max_iterations: int = 5) -> str:
    """
    Loop: call model → if it requests a tool, execute it and feed result back → repeat.
    Stops when the model returns a plain text response (no tool calls).
    """
    for iteration in range(max_iterations):
        logger.info(f"\n   [LOOP] Iteration {iteration + 1}/{max_iterations}")

        result = call_openrouter(messages)
        message = result["choices"][0]["message"]
        finish_reason = result["choices"][0].get("finish_reason", "")

        logger.info(f"   [LOOP] finish_reason: {finish_reason}")

        # --- Model wants to call a tool ---
        if finish_reason == "tool_calls" or message.get("tool_calls"):
            # IMPORTANT: append the model's tool-call message to history first
            messages.append(message)

            for tool_call in message.get("tool_calls", []):
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])

                if tool_name == "run_python":
                    tool_result = execute_python(tool_args["code"], df)
                else:
                    tool_result = f"Unknown tool: {tool_name}"

                logger.info(f"   [TOOL] Result preview: {str(tool_result)[:300]}")

                # Feed the result back into the conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result,
                })

            # Continue loop — model will now use the tool output to write its response

        # --- Model is done, return final text response ---
        else:
            ai_text = message.get("content", "")
            logger.info(f"   [LOOP] Final response ({len(ai_text)} chars)")
            return ai_text

    # Safety: if we hit max iterations, return whatever the last message content was
    logger.warning("   [LOOP] Max iterations reached")
    last = messages[-1]
    return last.get("content", "I wasn't able to complete the analysis. Please try again.")


# ==============================================================================
#  ROUTES
# ==============================================================================

@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, sid: str):
    logger.info(f"\n     Chat page requested | Session: {sid[:8]}...")

    redis_key = f"session:{sid}"
    if not redis_client.exists(redis_key):
        logger.warning(f"    Session not found, redirecting home")
        return RedirectResponse(url="/", status_code=303)

    session_data = redis_client.hgetall(redis_key)

    try:
        columns = json.loads(session_data.get("columns", "[]"))
        preview_rows = json.loads(session_data.get("preview_rows", "[]"))
    except json.JSONDecodeError:
        columns, preview_rows = [], []

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "session_id": sid,
        "file_name": session_data.get("file_name", "Unknown"),
        "file_extension": session_data.get("file_extension", ""),
        "file_size": session_data.get("file_size", "0 KB"),
        "upload_date": session_data.get("upload_date", ""),
        "upload_time": session_data.get("upload_time", ""),
        "num_rows": session_data.get("num_rows"),
        "num_columns": len(columns),
        "columns": columns,
        "preview_rows": preview_rows,
        "cache_buster": datetime.datetime.now().timestamp()
    })


@router.get("/chat/history")
async def chat_history(sid: str):
    """Return saved chat messages so the frontend can restore them on refresh."""
    redis_key = f"session:{sid}"
    if not redis_client.exists(redis_key):
        raise HTTPException(status_code=404, detail="Session not found")

    history_key = f"history:{sid}"
    raw_history = redis_client.lrange(history_key, 0, -1)  # get all messages

    messages = []
    for item in raw_history:
        msg_data = json.loads(item)
        # Map "model" role to "ai" for the frontend
        role = "ai" if msg_data["role"] == "model" else msg_data["role"]
        messages.append({"role": role, "content": msg_data["text"]})

    return {"messages": messages}


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
async def chat_endpoint(req: ChatRequest, sid: str):
    log_section("💬 CHAT MESSAGE RECEIVED", "─")

    redis_key = f"session:{sid}"
    if not redis_client.exists(redis_key):
        raise HTTPException(status_code=404,
            detail="Conversation Not Found. Session may be expired or deleted")

    # 1. Load session data
    session_data = redis_client.hgetall(redis_key)
    df_path = session_data.get("dataframe_path")

    if not df_path or not os.path.exists(df_path):
        raise HTTPException(status_code=500, detail="Data file missing")

    try:
        df = pd.read_parquet(df_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load data: {e}")

    # 2. Get AI context
    ai_context = session_data.get("ai_context") or session_data.get("data_context")
    if not ai_context:
        ai_context = make_ai_context(df, session_data.get("file_name", "Data"))

    # 3. Load chat history
    past_history = get_chat_history_openrouter(sid, limit=10)

    # 4. Build messages
    system_content = f"""{SYSTEM_INSTRUCTION}

### Context of the User's Data
{ai_context}

### Tool Usage Guidance
You have access to a `run_python` tool that executes real Python code against the user's dataframe.
Use it whenever a question requires actual computation — counts, averages, filtering, ranking, etc.
Never guess numbers. Run the code and use the real output in your response.
After getting tool results, respond naturally in HTML as if you simply knew the answer.
Never show raw code, print statements, or tool output to the user.
"""

    messages = [{"role": "system", "content": system_content}]
    messages.extend(past_history)
    messages.append({"role": "user", "content": req.message})

    # 5. Save user message before sending (so history is preserved even if AI fails)
    save_chat_message(sid, "user", req.message)

    # 6. Run the agentic loop
    try:
        ai_text = run_agentic_loop(messages, df)
    except requests.exceptions.Timeout:
        logger.error("   ✗ OpenRouter request timed out")
        raise HTTPException(status_code=504, detail="AI request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        logger.error(f"   ✗ OpenRouter request failed: {e}")
        raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")
    except Exception as e:
        logger.error(f"   ✗ Chat Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

    # 7. Save and return AI response
    save_chat_message(sid, "model", ai_text)
    logger.info(f"   ✓ Response saved ({len(ai_text)} chars)")

    print(ai_text)
    return {
        "response": {
            "text": ai_text,
        }
    }