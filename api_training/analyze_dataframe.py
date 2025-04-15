import json
import time
import os

from data_context import json_dataset_context
from data_dictionary import Json_data_dictionary
from call_gemini import call_gemini_api, GEMINI_API_KEY
from file_path import data





def analyze_larger_dataframe(df, API_KEY, delay=2):
    """
    Analyzes the dataset for potential issues, assigns a data quality score, 
    and presents findings in a structured, business-friendly manner.
    
    Args:
    - df (pd.DataFrame): The dataset to analyze.
    - API_KEY (str): The API key for the AI model.
    
    Returns:
    - str: A structured report highlighting data issues and quality insights.
    """

    # Convert sample of dataset to JSON for context
    sample_size = int(min(max(0.01 * len(df), 5), 500))  # 1% of data, min 5, max 500
    dataset_sample = df.sample(sample_size, random_state=42).to_dict(orient="records")

    # Ensure Json_data_dictionary and json_dataset_context are valid
    if not Json_data_dictionary or not isinstance(Json_data_dictionary, str):
        print("Error: Json_data_dictionary is missing or invalid.")
        return None

    if not json_dataset_context or not isinstance(json_dataset_context, str):
        print("Error: json_dataset_context is missing or invalid.")
        return None

    prompt = f"""
    You are a data quality analyst specializing in **business intelligence and data governance**. 
    Your task is to analyze a dataset and provide a **structured, business-friendly** summary 
    highlighting data issues, anomalies, and areas for improvement.

    ---
    ### **Dataset Overview**
    - **Context**: This dataset is provided with a data dictionary and metadata.
    - **Sample Data**: {json.dumps(dataset_sample, indent=4)}
    - **Data Dictionary**: {Json_data_dictionary}
    - **Metadata Summary**: {json_dataset_context}

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

        # Extract JSON from markdown response
        response = response.strip().replace("```json", "").replace("```", "").strip()

        response_dict = json.loads(response)
        return response_dict

    except json.JSONDecodeError:
        print(f"Failed to parse JSON from:\n{response}")
        return None
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        return None
    

analysis_result = analyze_larger_dataframe(data, GEMINI_API_KEY)
if analysis_result:

    analysis_result = json.dumps(analysis_result, indent=4)













# ===========================================================================
# ===========================================================================