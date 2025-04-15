import json
import time
import os
# from .data_context import json_dataset_context
# from .data_dictionary import Json_data_dictionary

from .data_context import data_information
from .data_dictionary import generate_data_dictionary
from .call_gemini import call_gemini_api, GEMINI_API_KEY
from .file_path import data





def analyze_larger_dataframe(df, API_KEY=GEMINI_API_KEY, delay=2):
    

    # Convert sample of dataset to JSON for context
    sample_size = int(min(max(0.01 * len(df), 5), 500))  # 1% of data, min 5, max 500
    dataset_sample = df.sample(sample_size, random_state=42).to_dict(orient="records")

    # Generate data dictionary and context
    Json_data_dictionary = generate_data_dictionary(df, API_KEY)
    if not Json_data_dictionary:
        print("Error: Failed to generate data dictionary.")
        return None

    json_dataset_context = data_information(df)
    if not json_dataset_context:
        print("Error: Failed to generate dataset context.")
        return None

    prompt = f"""
    You are a data quality analyst specializing in **business intelligence and data governance**. 
    Your task is to analyze a dataset and provide a **structured, business-friendly** summary 
    highlighting data issues, anomalies, and areas for improvement.

    ---
    ### **Dataset Overview**
    - **Context**: This dataset is provided with a data dictionary and metadata.
    - **Sample Data**: {json.dumps(dataset_sample, indent=4)}
    - **Data Dictionary**: {json.dumps(Json_data_dictionary, indent=4)}
    - **Metadata Summary**: {json.dumps(json_dataset_context, indent=4)}

    ---
    ### **Analysis Objectives**
    1. **Provide a High-Level Summary**  
       - What does this dataset seem to represent?
       - What key columns stand out?
       - Any **noteworthy observations** at first glance?

    2. **Identify Data Issues** (Categorized by Severity)  
       Group issues based on their potential impact:
       - **Critical Issues (Must Fix Immediately)**:  
         - Examples: Extremely high missing values, duplicate records, inconsistent formats, incorrect data types.
       - **Moderate Issues (Affects Data Reliability)**:  
         - Examples: Outliers that may skew analysis, inconsistent category values, mixed data types.
       - **Minor Issues (Low Impact, But Worth Fixing)**:  
         - Examples: Minor formatting inconsistencies, redundant columns, minor data drift.

    3. **Business Impact & Recommendations**  
       - Explain **how these issues might affect business insights**.
       - Suggest practical steps to **resolve or mitigate** the issues.

    4. **Data Quality Score (0-100)**  
       - Provide an overall data quality score based on how clean and reliable the dataset is.
       - Justify the score with clear reasoning.

    ---
    ### **Response Format**
    Ensure a **structured and user-friendly response** in this format:

    ```json
    {{
#         "summary": "Short, high-level description of the dataset.",
#         "issues": {{
#             "critical": [
#                 {{"column": "Column Name", "issue": "Description of the issue", "impact": "Business impact"}}
#             ],
#             "moderate": [
#                 {{"column": "Column Name", "issue": "Description of the issue", "impact": "Business impact"}}
#             ],
#             "minor": [
#                 {{"column": "Column Name", "issue": "Description of the issue", "impact": "Business impact"}}
#             ]
#         }},
#         "recommendations": [
#             "Practical, business-friendly steps to improve data quality."
#         ],
#         "data_quality_score": 85  // Justification for the score
#     }}
    ```

    **DO NOT include extra explanations or unnecessary text. Only return structured insights.**  
    """

    try:
        time.sleep(delay)
        response = call_gemini_api(prompt, api_key=API_KEY)
        
        if not response:
            print("Empty API response")
            return None


# =========================================================================
        # Clean the response and ensure it's valid JSON
        response = response.strip()
        
        # Remove any markdown code block indicators
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        # Remove any comments from the JSON
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove any line comments
            if '//' in line:
                line = line[:line.find('//')]
            cleaned_lines.append(line.strip())
        response = '\n'.join(cleaned_lines)
# ==============================================================================



# ==============================================================================

        try:
            response_dict = json.loads(response)
            return response_dict
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print("Raw response:", response)
            # Try to fix common JSON issues
            try:
                # Remove any trailing incomplete objects
                response = response[:response.rfind("}") + 1]
                response_dict = json.loads(response)
                return response_dict
            except:
                print("Failed to fix JSON parsing")
                return None
# ==============================================================================


    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    analysis_result = analyze_larger_dataframe(data)
    if analysis_result:
        analysis_result = json.dumps(analysis_result, indent=4)
        # print(analysis_result)















# ===========================================================================
# ===========================================================================