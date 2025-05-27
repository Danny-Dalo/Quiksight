# import json
# import time
# from .call_gemini import call_gemini_api, GEMINI_API_KEY



# def generate_data_dictionary(df, api_key, num_samples=1, delay=3):  # Added delay parameter

#     sample_data = df.sample(min(num_samples, len(df)), random_state=42).to_dict(orient="records")

#     prompt = f"""
# You are a data expert assisting in understanding the structure and semantics of a dataset. Below is a small representative sample:

# **Sample Data:**
# {json.dumps(sample_data, indent=4)}

# **Your Task:**
# Analyze the sample data and generate a clear and insightful data dictionary. For each column:

# - Identify what the column represents using natural, descriptive language.
# - Provide a concise, human-readable description (aim for clarity, not rigidity).
# - Classify the column as one of: **categorical, numerical, date, or identifier or others**.
# - Specify the **data type**: text, numeric, date, boolean.
# - If applicable, mention the expected **format** (e.g., "YYYY-MM-DD", "2 decimal places").
# - Identify any **obvious relationships** or dependencies with other columns (e.g., "ID links to department", "derived from date of birth").

# **Important Notes:**
# - You are helping build context for an automated cleaning tool, so clarity and precision are essential.
# - If any column is unclear, infer meaning based on data patterns.
# - Do not include any explanations outside the JSON. Return **only** valid JSON output in the structure below.

# **Expected Output Format:**
# {{
#     "column_name": {{
#         "description": "Brief explanation of what this column represents.",
#         "type": "categorical/numerical/date/identifier",
#         "data_type": "text/numeric/date/boolean",
#         "format": "Optional: format like YYYY-MM-DD, float with 2 decimals, etc.",
#         "relationships": "Optional: brief note on any related columns."
#     }},
#     ...
# }}
# """


#     try:
#         # Introduce a delay to avoid hitting the rate limit
#         time.sleep(delay)  # Adjust delay if needed

#         response = call_gemini_api(prompt, api_key=api_key)

#         # Clean potential markdown formatting
#         cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()

#         # Validate and parse JSON
#         response_dict = json.loads(cleaned_response)
#         return response_dict

#     except json.JSONDecodeError:
#         print("Invalid JSON received:", response)
#         return None
#     except Exception as e:
#         print("API Request Failed:", str(e))
#         return None


















