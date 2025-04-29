import json
import time


from .data_context import data_information
from .data_dictionary import generate_data_dictionary
from .call_gemini import call_gemini_api, GEMINI_API_KEY
import pandas as pd
from app_quiksight.services.data_overview import _clean_value_for_json



def calculate_detailed_data_quality_score(df, critical_issues, moderate_issues):
    total_rows = len(df)
    total_columns = len(df.columns)
    
    # Calculate basic missing data
    missing_values = df.isnull().sum().sum()
    total_cells = df.size
    missing_percentage = (missing_values / total_cells) * 100

    # Calculate duplicates
    duplicate_rows = df.duplicated().sum()
    duplicate_percentage = (duplicate_rows / total_rows) * 100

    # Columns with constant values (e.g., single value columns)
    constant_columns = df.nunique() == 1
    constant_columns_percentage = (constant_columns.sum() / total_columns) * 100

    # Calculate data type mismatches (non-numeric values in numeric columns)
    data_type_issues = {}
    for column in df.select_dtypes(include=['number']).columns:
        non_numeric_values = df[column].apply(lambda x: isinstance(x, (str, bool)))
        data_type_issues[column] = non_numeric_values.sum()
    data_type_issues_percentage = sum(data_type_issues.values()) / total_cells * 100

    # Data consistency: Check for out-of-range values (numeric columns with negative values if not allowed)
    out_of_range_issues = {}
    for column in df.select_dtypes(include=['number']).columns:
        if df[column].min() < 0:  # Assuming negative values are an issue
            out_of_range_issues[column] = df[column].min()
    out_of_range_percentage = (sum(1 for v in out_of_range_issues.values()) / total_columns) * 100

    # Uniqueness: Percentage of unique values in categorical columns
    unique_percentage = {}
    for column in df.select_dtypes(include=['object']).columns:
        unique_percentage[column] = (df[column].nunique() / total_rows) * 100

    # Empty categorical fields where missing values are unacceptable
    empty_categorical = df.select_dtypes(include=['object']).isnull().sum() / total_rows
    empty_categorical_percentage = empty_categorical.sum()

    # Cleanliness: Whitespace errors and formatting issues (e.g., dates)
    cleanliness_issues = {}
    for column in df.select_dtypes(include=['object']).columns:
        cleanliness_issues[column] = df[column].str.contains(r'\s').sum()
    cleanliness_percentage = sum(cleanliness_issues.values()) / total_cells * 100

    # Start from a base score of 100.
    score = 100

    # Penalize based on missing data percentage.
    if missing_percentage > 5:
        score -= missing_percentage * 0.5

    # Penalize for duplicate rows.
    if duplicate_percentage > 2:
        score -= duplicate_percentage * 2

    # Penalize for constant columns.
    if constant_columns_percentage > 10:
        score -= constant_columns_percentage * 1

    # Penalize for data type mismatches.
    if data_type_issues_percentage > 2:
        score -= data_type_issues_percentage * 1.5

    # Penalize for out-of-range values.
    if out_of_range_percentage > 5:
        score -= out_of_range_percentage * 2

    # Penalize for low uniqueness in categorical columns.
    for column, percentage in unique_percentage.items():
        if percentage < 20:
            score -= (20 - percentage) * 0.5

    # Penalize for empty categorical fields.
    if empty_categorical_percentage > 5:
        score -= empty_categorical_percentage * 1.5

    # Penalize for cleanliness issues (whitespace errors).
    if cleanliness_percentage > 5:
        score -= cleanliness_percentage * 0.5

    # Penalize for detected issues (critical/moderate)
    score -= (len(critical_issues) * 10)  # Each critical issue subtracts 10 points.
    score -= (len(moderate_issues) * 5)    # Each moderate issue subtracts 5 points.

    # Ensure the score is between 0 and 100.
    return max(int(score), 0)






































def analyze_larger_dataframe(df, API_KEY=GEMINI_API_KEY, delay=2):
    if len(df) <= 1000:
        # For small datasets (≤10k rows), use the entire dataset
        dataset_sample = df.to_dict(orient="records")
    else:
        # For large datasets (>10k rows), use 1% with min 5 and max 500 samples
        sample_size = max(min(int(0.001 * len(df)), 500), 5)
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
    
    # =============================================================================
    critical_issues = []
    moderate_issues = []

    # Calculate the detailed data quality score
    score = calculate_detailed_data_quality_score(df, critical_issues, moderate_issues)
    # =================================================================================




    prompt = f"""
        You are a data quality assistant specialized in **business intelligence and data governance**.  
Your role is to **analyze datasets and report findings** to a **data scientist** — speaking professionally, clearly, and to the point.

---
### **Dataset Overview**
- **Context**: This dataset comes with a data dictionary and metadata.
- **Sample Data**: {json.dumps(dataset_sample, indent=4)}
- **Data Dictionary**: {json.dumps(Json_data_dictionary)}
- **Metadata Summary**: {json.dumps(json_dataset_context)}

---
### **Analysis Objectives**
1. **High-Level Summary**  
   - Briefly state what the dataset represents.
   - Highlight key columns and any immediate structural observations.

2. **Identify and Categorize Data Issues**  
   Group findings into:
   - **Critical Issues** (Severe problems requiring urgent attention)  
   - **Moderate Issues** (Problems that could affect reliability)  
   - **Minor Issues** (Low impact but worth noting)

   Focus on **precise, factual observations**.  
   Keep descriptions **clear, technical, and business-aware but concise**.

3. **Business Impact & Recommendations**  
   - Comment briefly on the **potential effect** of these issues.
   - Provide **straightforward, actionable recommendations** — no overexplaining.

4. **Data Quality Score (0-100)**  
   - Assign a score based on overall quality.
   - Justify with **direct, technical reasoning**.

---
### **Response Format**
Return a **clean, structured JSON** like this:

```json
{{
    "summary": "Concise description of the dataset.",
    "issues": {{
        "critical": [
            {{"column": "Column Name", "issue": "Description of the problem.", "impact": "Brief business or analysis risk."}}
        ],
        "moderate": [
            {{"column": "Column Name", "issue": "Description of the problem.", "impact": "Brief business or analysis risk."}}
        ],
        "minor": [
            {{"column": "Column Name", "issue": "Description of the problem.", "impact": "Brief business or analysis risk."}}
        ]
    }},
    "recommendations": [
        "Actionable steps to improve data quality."
    ],
    "data_quality_score": {{
    "score": {score},
    "justification": "Justification as to why you gave it the assigned score"
}}

}}

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


# ===========================================================================
# ===========================================================================





#     prompt = f"""
#     You are a data quality analyst specializing in **business intelligence and data governance**. 
#     Your task is to analyze a dataset and provide a **structured, business-friendly** summary 
#     highlighting data issues, anomalies, and areas for improvement.

#     ---
#     ### **Dataset Overview**
#     - **Context**: This dataset is provided with a data dictionary and metadata.
#     - **Sample Data**: {json.dumps(dataset_sample, indent=4)}
#     - **Data Dictionary**: {json.dumps(Json_data_dictionary)}
#     - **Metadata Summary**: {json.dumps(json_dataset_context)}

#     ---
#     ### **Analysis Objectives**
#     1. **Provide a High-Level Summary**  
#        - What does this dataset seem to represent?
#        - What key columns stand out?
#        - Any **noteworthy observations** at first glance?

#     2. **Identify Data Issues** (Categorized by Severity)  
#        Group issues based on their potential impact:
#        - **Critical Issues (Must Fix Immediately)**:  
#          - Examples: Extremely high missing values, duplicate records, inconsistent formats, incorrect data types.
#        - **Moderate Issues (Affects Data Reliability)**:  
#          - Examples: Outliers that may skew analysis, inconsistent category values, mixed data types.
#        - **Minor Issues (Low Impact, But Worth Fixing)**:  
#          - Examples: Minor formatting inconsistencies, redundant columns, minor data drift.

#     3. **Business Impact & Recommendations**  
#        - Explain **how these issues might affect business insights**.
#        - Suggest practical steps to **resolve or mitigate** the issues.

#     4. **Data Quality Score (0-100)**  
#        - Provide an overall data quality score based on how clean and reliable the dataset is.
#        - Justify the score with clear reasoning.

#     ---
#     ### **Response Format**
#     Ensure a **structured and user-friendly response** in this format:

#     ```json
#     {{
# #         "summary": "Short, high-level description of the dataset.",
# #         "issues": {{
# #             "critical": [
# #                 {{"column": "Column Name", "issue": "Description of the issue", "impact": "Business impact"}}
# #             ],
# #             "moderate": [
# #                 {{"column": "Column Name", "issue": "Description of the issue", "impact": "Business impact"}}
# #             ],
# #             "minor": [
# #                 {{"column": "Column Name", "issue": "Description of the issue", "impact": "Business impact"}}
# #             ]
# #         }},
# #         "recommendations": [
# #             "Practical, business-friendly steps to improve data quality."
# #         ],
# #         "data_quality_score": ["An overall score of the quality of the data"  // Justification for the score]
# #     }}
#     ```

#     **DO NOT include extra explanations or unnecessary text. Only return structured insights.**  
#     """