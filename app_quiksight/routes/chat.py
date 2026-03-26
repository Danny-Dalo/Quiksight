import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app_quiksight.storage.auth_deps import get_optional_user
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
                "Execute Python code against the user's dataset(s) to answer questions that require "
                "real computation: counts, sums, averages, filters, groupbys, comparisons, rankings, "
                "correlations, or anything you cannot determine from the context summary alone. "
                "Use this whenever the user asks a question that needs actual numbers from the data. "
                "Do NOT guess or estimate — run code and use real output. "
                "The data is loaded as a dictionary of pandas DataFrames called `dfs`, where keys are dataset names. "
                "The result of the last expression is returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Valid Python code. `dfs` (dict of pandas DataFrames) and `pd` are pre-loaded. "
                            "Make the last line an expression whose value is the answer "
                            "(e.g. `dfs['sales.csv']['revenue'].sum()`). "
                            "For multi-line code, store your final answer in a variable called `result` "
                            "and make the last line just `result`."
                        )
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": (
                "Generate a Plotly chart to visually represent data findings. "
                "Use this when a chart would genuinely help the user understand the answer — "
                "rankings, trends over time, distributions, comparisons across categories. "
                "Do NOT use for single numbers or simple facts that don't benefit from visualization. "
                "Write Python code using plotly.express (imported as `px`) and `dfs` (dict of pandas DataFrames). "
                "Assign your final figure to a variable called `fig`. "
                "Returns Plotly JSON that will be rendered as an interactive chart."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code using `dfs` (dict of pandas DataFrames), `pd`, and `px` (plotly.express). "
                            "Must assign the final Plotly figure to `fig`. "
                            "Apply a clean layout: white background, subtle gridlines, clear axis labels, "
                            "a descriptive title. Use `color_discrete_sequence` or `color` where helpful. "
                            "Example:\n"
                            "  result = dfs['sales.csv'].groupby('region')['sales'].sum().reset_index()\n"
                            "  fig = px.bar(result, x='region', y='sales', title='Sales by Region')\n"
                            "  fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')"
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

def execute_python(code: str, dfs: dict) -> str:
    """
    Execute model-generated Python code against the user's dataframes.
    Returns the result as a string, or a formatted error message.
    """
    logger.info(f"\n   [TOOL] run_python called:\n{code}")

    # Give the model access to dfs and pandas only — nothing else
    local_env = {
        "dfs": {k: v.copy() for k, v in dfs.items()},  # copy so the model can't mutate the real dfs
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
        # Return the error to the model so it can retry — but instruct it not to expose this to the user
        return (
            f"[INTERNAL TOOL ERROR — DO NOT SHOW THIS TO THE USER] {error_msg}. "
            "Retry with corrected code, or respond to the user with a friendly message "
            "saying you encountered a hiccup processing their request."
        )


def execute_chart_code(code: str, dfs: dict) -> str:
    """
    Execute model-generated Plotly code. Returns Plotly figure as JSON string,
    or an error message string if execution fails.
    """
    logger.info(f"\n   [TOOL] generate_chart called:\n{code}")

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        return "ChartError: Charting library is unavailable."

    local_env = {
        "dfs": {k: v.copy() for k, v in dfs.items()},
        "pd": pd,
        "px": px,
        "go": go,
        "fig": None,
    }

    try:
        exec(code, local_env)
        fig = local_env.get("fig")

        if fig is None:
            return "ChartError: Chart generation did not produce a result."

        # Return the Plotly figure as JSON — frontend renders it with Plotly.js
        return fig.to_json()

    except Exception as e:
        error_msg = f"ChartError: {type(e).__name__}: {str(e)}"
        logger.warning(f"   [TOOL] Chart generation failed: {error_msg}")
        return (
            f"[INTERNAL TOOL ERROR — DO NOT SHOW THIS TO THE USER] {error_msg}. "
            "Skip the chart silently and provide a text or table answer instead."
        )

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

def run_agentic_loop(messages: list, dfs: dict, max_iterations: int = 5) -> dict:
    """
    Loop: call model → if it requests a tool, execute it and feed result back → repeat.
    Stops when the model returns a plain text response (no tool calls).
    Returns {"text": str, "charts": list[str]} where charts are Plotly JSON strings.
    """
    charts = []  # Collect chart JSONs as we go

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
                    tool_result = execute_python(tool_args["code"], dfs)
                elif tool_name == "generate_chart":
                    chart_json = execute_chart_code(tool_args["code"], dfs)
                    # If it's valid JSON (not an error string), collect it
                    if not chart_json.startswith(("ChartError:", "[INTERNAL TOOL ERROR")):
                        charts.append(chart_json)
                        tool_result = "Chart generated successfully. Place a <!-- chart --> comment in your HTML response where this chart should appear."
                    else:
                        tool_result = chart_json
                else:
                    tool_result = (
                        "[INTERNAL TOOL ERROR — DO NOT SHOW THIS TO THE USER] "
                        f"Unknown tool '{tool_name}'. Respond naturally without mentioning this error."
                    )

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
            logger.info(f"   [LOOP] Final response ({len(ai_text)} chars), {len(charts)} charts collected")
            return {"text": ai_text, "charts": charts}

    # Safety: if we hit max iterations, return whatever the last message content was
    logger.warning("   [LOOP] Max iterations reached")
    last = messages[-1]
    return {"text": last.get("content", "I wasn't able to complete the analysis. Please try again."), "charts": charts}


# ==============================================================================
#  ROUTES
# ==============================================================================

# endpoint to handle user sessions
@router.get("/sessions")
async def list_sessions(user = Depends(get_optional_user)):
    """Return all active sessions for the logged-in user."""
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_sessions_key = f"user_sessions:{user.id}"
    session_ids = redis_client.smembers(user_sessions_key)

    sessions = []
    expired_ids = []
    for sid in session_ids:
        redis_key = f"session:{sid}"
        if not redis_client.exists(redis_key):
            expired_ids.append(sid)
            continue
        data = redis_client.hgetall(redis_key)
        sessions.append({
            "session_id": sid,
            "file_name": data.get("file_name", "Unknown"),
            "upload_date": data.get("upload_date", ""),
            "upload_time": data.get("upload_time", ""),
            "num_rows": data.get("num_rows", "0"),
            "file_size": data.get("file_size", ""),
        })

    # Clean up expired sessions from the user's set
    if expired_ids:
        redis_client.srem(user_sessions_key, *expired_ids)

    # Sort newest first by date and time in session bar
    sessions.sort(key=lambda s: (s["upload_date"], s["upload_time"]), reverse=True)
    return {"sessions": sessions}


@router.delete("/sessions/{sid}")
async def delete_session(sid: str, user = Depends(get_optional_user)):
    """Delete a session and its chat history."""
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    redis_key = f"session:{sid}"
    session_data = redis_client.hgetall(redis_key)

    # Ensures the session belongs to the user
    if session_data.get("user_id") != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Delete session data, chat history, and remove from user's set
    redis_client.delete(redis_key)
    redis_client.delete(f"history:{sid}")
    redis_client.srem(f"user_sessions:{user.id}", sid)

    return {"status": "deleted"}


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, sid: str, user = Depends(get_optional_user)):
    if not user:
        return RedirectResponse(url="/login", status_code=303)
        
    logger.info(f"\n     Chat page requested | Session: {sid[:8]}...")

    redis_key = f"session:{sid}"
    if not redis_client.exists(redis_key):
        logger.warning(f"    Session not found, redirecting home")
        return RedirectResponse(url="/", status_code=303)

    session_data = redis_client.hgetall(redis_key)

    try:
        columns_dict = json.loads(session_data.get("columns", "{}"))
        preview_data = json.loads(session_data.get("preview_rows", "{}"))
        
        if isinstance(columns_dict, dict) and columns_dict:
            first_key = list(columns_dict.keys())[0]
            columns_view = columns_dict[first_key]
            preview_view = preview_data[first_key]
        else:
            columns_view = columns_dict if isinstance(columns_dict, list) else []
            preview_view = preview_data if isinstance(preview_data, list) else []
            
    except json.JSONDecodeError:
        columns_view, preview_view = [], []

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "user": user,
        "session_id": sid,
        "file_name": session_data.get("file_name", "Unknown"),
        "file_extension": session_data.get("file_extension", ""),
        "file_size": session_data.get("file_size", "0 KB"),
        "upload_date": session_data.get("upload_date", ""),
        "upload_time": session_data.get("upload_time", ""),
        "num_rows": session_data.get("num_rows"),
        "num_columns": len(columns_view),
        "columns": columns_view,
        "preview_rows": preview_view,
        "cache_buster": datetime.datetime.now().timestamp()
    })


@router.get("/chat/history")
async def chat_history(sid: str, user = Depends(get_optional_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

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
        entry = {"role": role, "content": msg_data["text"]}
        if "charts" in msg_data:
            entry["charts"] = msg_data["charts"]
        messages.append(entry)

    return {"messages": messages}


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
async def chat_endpoint(req: ChatRequest, sid: str, user = Depends(get_optional_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    log_section("💬 CHAT MESSAGE RECEIVED", "─")

    redis_key = f"session:{sid}"
    if not redis_client.exists(redis_key):
        raise HTTPException(status_code=404,
            detail="Conversation Not Found. Session may be expired or deleted")

    # 1. Load session data
    session_data = redis_client.hgetall(redis_key)
    
    dfs = {}
    df_paths_json = session_data.get("dataframe_paths")
    if df_paths_json:
        try:
            df_paths = json.loads(df_paths_json)
            for name, path in df_paths.items():
                if os.path.exists(path):
                    dfs[name] = pd.read_parquet(path)
        except Exception as e:
            logger.error(f"   ✗ Failed to load parquet from dict: {e}")
            raise HTTPException(status_code=500,
                detail="We had trouble loading your data. Please try uploading your file again.")
    else:
        # Fallback to single file syntax
        df_path = session_data.get("dataframe_path")
        if not df_path or not os.path.exists(df_path):
            raise HTTPException(status_code=500,
                detail="Your data file could not be found. It may have expired — please try uploading your file again.")
        try:
            dfs[session_data.get("file_name", "Data")] = pd.read_parquet(df_path)
        except Exception as e:
            logger.error(f"   ✗ Failed to load parquet: {e}")
            raise HTTPException(status_code=500,
                detail="We had trouble loading your data. Please try uploading your file again.")
        
    if not dfs:
        raise HTTPException(status_code=500, detail="No readable datasets found.")

    ai_context = session_data.get("data_context")
    if not ai_context:
        # Fallback to creating context for legacy sessions
        contexts = []
        for name, _df in dfs.items():
            contexts.append(f"📁 Dataset: {name}\n" + make_ai_context(_df, name))
        ai_context = "\n\n---\n\n".join(contexts)

    # 3. Load chat history
    past_history = get_chat_history_openrouter(sid, limit=10)

    # 4. Build messages
    system_content = f"""{SYSTEM_INSTRUCTION}

    ### Context of the User's Data
    {ai_context}

    ### Tool Usage Guidance
    You have access to two tools:

    1. `run_python` — use for any question requiring real computation: counts, averages, filters, rankings, etc.
    2. `generate_chart` — use when a chart genuinely helps understand the answer (comparisons, trends, distributions, rankings). Do not use for single values or simple facts.

    Rules:
    - Never guess numbers. Run code and use real output.
    - After getting tool results, respond naturally in HTML as if you simply knew the answer.
    - Never show raw code, print statements, tool output, or internal error messages to the user.
    - If a tool call fails, DO NOT mention the error. Simply respond with what you can, or say you had a hiccup and ask the user to try again.
    - When generate_chart succeeds, place the HTML comment <!-- chart --> in your response where the chart should appear. The system will inject the actual chart there automatically. Do NOT try to embed any JSON yourself.
    - If generate_chart returns an error, skip the chart silently and just provide the table/text answer.
    """

    messages = [{"role": "system", "content": system_content}]
    messages.extend(past_history)
    messages.append({"role": "user", "content": req.message})

    # 5. Save user message before sending (so history is preserved even if AI fails)
    save_chat_message(sid, "user", req.message)

    # 6. Run the agentic loop
    try:
        loop_result = run_agentic_loop(messages, dfs)
    except requests.exceptions.Timeout:
        logger.error("   ✗ OpenRouter request timed out")
        raise HTTPException(status_code=504,
            detail="The AI is taking too long to respond. Please try sending your message again.")
    except requests.exceptions.RequestException as e:
        logger.error(f"   ✗ OpenRouter request failed: {e}")
        raise HTTPException(status_code=502,
            detail="We're having trouble reaching the AI service right now. Please try again in a moment.")
    except Exception as e:
        logger.error(f"   ✗ Chat Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500,
            detail="Something unexpected happened while processing your request. Please try again.")

    ai_text = loop_result["text"]
    charts = loop_result["charts"]

    # 7. Save and return AI response
    save_chat_message(sid, "model", ai_text, charts=charts)
    logger.info(f"   ✓ Response saved ({len(ai_text)} chars, {len(charts)} charts)")

    return {
        "response": {
            "text": ai_text,
            "charts": charts,
        }
    }