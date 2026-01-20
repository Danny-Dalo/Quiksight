import redis
import json
from google.genai import types

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)



# --- HISTORY HELPERS ---
def get_chat_history(sid: str, limit: int = 20):
    """
    Fetches the last 'limit' messages from Redis and formats them 
    for the Gemini API.
    """
    history_key = f"history:{sid}"
    # Get last N messages (Redis lists are 0-indexed)
    # 0 to -1 means "everything", but we'll implement the limit logic manually if needed
    # or just grab the tail. Let's grab everything and slice in Python for simplicity.
    raw_history = redis_client.lrange(history_key, -limit, -1)
    
    formatted_history = []
    for item in raw_history:
        # Redis returns bytes/strings, we stored JSON
        msg_data = json.loads(item)
        
        # Convert to Gemini's expected format
        formatted_history.append(
            types.Content(
                role=msg_data["role"],
                parts=[types.Part(text=msg_data["text"])]
            )
        )
    return formatted_history

def save_chat_message(sid: str, role: str, text: str):
    """Saves a single message to Redis history."""
    history_key = f"history:{sid}"
    message_data = json.dumps({"role": role, "text": text})
    
    # Push to the end of the list (Right Push)
    redis_client.rpush(history_key, message_data)
    
    # Optional: Trim list to keep only last 50 items to save Redis space
    # (This automatically handles the storage bottle neck)
    redis_client.ltrim(history_key, -20, -1)