import json
import time
from .file_path import data
from .call_gemini import call_gemini_api, GEMINI_API_KEY



def generate_data_dictionary(df, api_key, num_samples=3, delay=3):  # Added delay parameter

    sample_data = df.sample(min(num_samples, len(df)), random_state=42).to_dict(orient="records")

    prompt = f"""
    You are analyzing a dataset. Below is a small sample:

    **Sample Data:**
    {json.dumps(sample_data, indent=4)}

    **Task:**
    - Identify what each column represents.
    - Provide a short, clear description of each column.
    - Identify if the column is **categorical, numerical, date, or identifier**.
    - Specify the data type (text, numeric, date, boolean).
    - Mention how the column may relate to others (if applicable). Make this a bit more detailed. Don't be rigid with description language here.
    - Return **only valid JSON output**, without extra text.

    **Expected JSON Output Format:**
    {{
        "column_name": {{
            "description": "Short explanation of what this column represents.",
            "type": "categorical/numerical/date/identifier",
            "data_type": "text/numeric/date/boolean",
            "format": "e.g., YYYY-MM-DD for dates, 2 decimal places for floats",
            "relationships": "Optional: Mention related columns if obvious"
        }}
    }}
    """

    try:
        # Introduce a delay to avoid hitting the rate limit
        time.sleep(delay)  # Adjust delay if needed

        response = call_gemini_api(prompt, api_key=api_key)

        # Clean potential markdown formatting
        cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()

        # Validate and parse JSON
        response_dict = json.loads(cleaned_response)
        return response_dict

    except json.JSONDecodeError:
        print("Invalid JSON received:", response)
        return None
    except Exception as e:
        print("API Request Failed:", str(e))
        return None


# Usage with delay
data_dictionary = generate_data_dictionary(data, GEMINI_API_KEY, delay=2)

if data_dictionary is not None:
  Json_data_dictionary = json.dumps(data_dictionary)
#   print(Json_data_dictionary)
else:
  print("❌ Failed to generate a valid data dictionary.")

Json_data_dictionary = json.dumps(data_dictionary, indent=4)

