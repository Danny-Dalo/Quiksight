import os
import redis
import json
from google.genai import types
import logging

logging.basicConfig(
    level=logging.INFO,
    format='\n%(asctime)s | %(levelname)-8s | %(name)-10s | %(message)s',
    datefmt='%H:%M:%S'
)

# Use REDIS_URL if available (common in cloud deployments)
redis_url = os.getenv("REDIS_URL")

if redis_url:
    logging.info("Redis URL present")
    redis_client = redis.from_url(redis_url, decode_responses=True)
else:
    logging.info("Redis URL not present. Switching to local redis client")
    redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)


"""Gets chat history, only works if there has been history saved already"""
def get_chat_history(sid: str, limit: int = 20):
    """
    Fetches the last 'limit' messages from Redis and formats them 
    for the Gemini API.
    """
    history_key = f"history:{sid}"
    raw_history = redis_client.lrange(history_key, -limit, -1)  # history ranges from last 20th message(-limit) down to last message(-1)
    
    formatted_history = []
    for item in raw_history:
        # JSON.loads cuz redis sends bytes normally
        msg_data = json.loads(item)
        
        """Convert to a suitable format for Gemini"""
        formatted_history.append(
            types.Content(
                role=msg_data["role"],
                parts=[types.Part(text=msg_data["text"])]
            )
        )
    return formatted_history


def get_chat_history_openrouter(sid: str, limit: int = 20):
    """
    Fetches the last 'limit' messages from Redis and formats them 
    for the OpenRouter API (OpenAI-compatible format).
    """
    history_key = f"history:{sid}"
    raw_history = redis_client.lrange(history_key, -limit, -1)
    
    formatted_history = []
    for item in raw_history:
        msg_data = json.loads(item)
        
        # OpenRouter uses "assistant" instead of "model"
        role = "assistant" if msg_data["role"] == "model" else msg_data["role"]
        
        formatted_history.append({
            "role": role,
            "content": msg_data["text"]
        })
    return formatted_history


def save_chat_message(sid: str, role: str, text: str, charts: list = None):
    """Saves a single message to Redis history"""
    history_key = f"history:{sid}"
    msg = {"role": role, "text": text}
    if charts:
        msg["charts"] = charts
    message_data = json.dumps(msg)
    
    """Push to the end of the list (Right Push) to maintain chat order"""
    redis_client.rpush(history_key, message_data)
    
    """Removes oldest items beyond limit"""
    redis_client.ltrim(history_key, -60, -1)
    redis_client.expire(history_key, 3600)