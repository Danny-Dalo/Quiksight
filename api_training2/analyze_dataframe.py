import json
import time

from .data_context import data_information
# from .data_dictionary import generate_data_dictionary
from .call_gemini import call_gemini_api, GEMINI_API_KEY





def analyze_larger_dataframe(df, API_KEY=GEMINI_API_KEY, delay=2):
    if len(df) <= 1000:
        # For small datasets (≤10k rows), use the entire dataset
        dataset_sample = df.to_dict(orient="records")
    else:
        # For large datasets (>10k rows), use 1% with min 5 and max 500 samples
        sample_size = max(min(int(0.001 * len(df)), 500), 5)
        dataset_sample = df.sample(sample_size, random_state=42).to_dict(orient="records")


 


    

    # Generate data dictionary and context
    # Json_data_dictionary = generate_data_dictionary(df, API_KEY)
    # if not Json_data_dictionary:
    #     print("Error: Failed to generate data dictionary.")
    #     return None
    # - **Data Dictionary**: {json.dumps(Json_data_dictionary)}

    json_dataset_context = data_information(df)
    if not json_dataset_context:
        print("Error: Failed to generate dataset context.")
        return None



    prompt = f"""
You are a Data Quality Specialist with expertise in **business intelligence, data cleaning, and governance**.  
You will analyze the dataset below and report findings as if assisting a **professional data scientist**.  

Your job is to identify **real, context-aware data quality issues** — not just syntactic issues, but **semantic, factual, and structural ones**.  
Examples include: misspellings, invalid mappings (e.g., wrong continent for a country), unrealistic values, unexpected/improper formats, inconsistencies and other anomalies that may be hard to spot.
---
### 🔍 Dataset Inputs:
- **Sample Data**: {json.dumps(dataset_sample, indent=4)}

- **Metadata/Context**: {json.dumps(json_dataset_context)}

---
###  Analysis Objectives:

1. **High-Level Summary**
   - Summarize what the dataset is about in 2-3 clear sentences.
   - Highlight important columns and structural characteristics (e.g., ID columns, date ranges, groupings).

2. **Identify & Categorize Issues**
   Review all columns in context and list issues under:
   - **Critical Issues**: Severe problems (e.g., factual inaccuracies, broken keys, nulls in key columns).
   - **Moderate Issues**: Likely to affect analysis (e.g., inconsistent formats, suspicious outliers).
   - **Minor Issues**: Mild inconsistencies (e.g., minor spelling errors, naming variations).

   ✔ Think critically. Only report issues that **matter in real-world usage**.  
   ❌ Do **not** make up issues just to fill space. If data is clean, state it confidently.

3. **Business Impact & Fix Suggestions**
   - For each issue, include:
     - **Impact**: What effect could this have on business analysis or decision-making?

---
### **Response Format**
# Return a **clean, structured JSON** like this:
```json
{{
    "summary": "Short, clear overview of the dataset.",
    "issues": {{
        "critical": [
            {{
                "column": "ColumnName",
                "issue": "What is wrong and why it's a problem.",
                "impact": "What this could affect in real-world use."
            }}
        ],
        "moderate": [{{
                "column": "ColumnName",
                "issue": "What is wrong and why it's a problem.",
                "impact": "What this could affect in real-world use."
        }}],
        "minor": [{{
                "column": "ColumnName",
                
                "issue": "What is wrong and why it's a problem.",
                "impact": "What this could affect in real-world use."
        }}]
    }},
    "recommendations": [
        "List practical suggestions to improve the data going forward."
    ],
}}```
  """


    
    

    try:
        time.sleep(delay)
        response = call_gemini_api(prompt, api_key=API_KEY)
        
        if not response:
            print("Empty API response")
            return None
        
        response = response.strip()
        
        # =================================================== HANDLING INVALID JSON FORMAT
        if response.startswith("```"):
            # Remove the first line (e.g., ```json or ```)
            response = "\n".join(response.splitlines()[1:])
            # Remove the last line if it's ```
            if response.strip().endswith("```"):
                response = "\n".join(response.splitlines()[:-1])
            response = response.strip()
        # ======================================================== HANDLING INVALID JSON FORMAT

        try:
            result = json.loads(response)
        except Exception:
            # If response is already a dict or not JSON, just return as is
            result = response

        return result
        


    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        return None

