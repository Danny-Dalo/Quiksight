import redis
import json
from google.genai import types

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



def save_chat_message(sid: str, role: str, text: str):
    """Saves a single message to Redis history"""
    history_key = f"history:{sid}"
    message_data = json.dumps({"role": role, "text": text})
    
    """Push to the end of the list (Right Push) to maintain chat order"""
    redis_client.rpush(history_key, message_data)
    
    """Removes 20th item"""
    redis_client.ltrim(history_key, -20, -1)

    redis_client.expire(history_key, 3600)