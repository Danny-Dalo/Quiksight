import json
import time

from .data_context import data_information
# from .data_dictionary import generate_data_dictionary
from .call_gemini import call_gemini_api, GEMINI_API_KEY






# DATA SAMPLING FUNCTION
def sample_dataframe_for_ai(df, max_records=500):
    # Initialize sampled rows as empty set
    sampled_rows = set()
    
    # 1. Include rows with NaNs to show missing data handling needs
    nan_rows = df[df.isna().any(axis=1)]
    sampled_rows.update(nan_rows.index[:5])  # Add up to 5

    # 2. For each column, sample:
    for col in df.columns:
        # a) Include most frequent value row
        top_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
        if top_val is not None:
            top_row_idx = df[df[col] == top_val].index[0]
            sampled_rows.add(top_row_idx)

        # b) Include rarest value row (if possible)
        value_counts = df[col].value_counts()
        if len(value_counts) > 0:
            rare_val = value_counts.index[-1]
            rare_row_idx = df[df[col] == rare_val].index[0]
            sampled_rows.add(rare_row_idx)

        # c) Add a few random rows per column to capture variety
        random_rows = df.sample(min(2, len(df)), random_state=42).index
        sampled_rows.update(random_rows)

    # 3. Add overall random rows if total is still small
    if len(sampled_rows) < max_records:
        remaining = max_records - len(sampled_rows)
        random_extra = df.drop(index=list(sampled_rows)).sample(
            min(remaining, len(df) - len(sampled_rows)), random_state=42
        ).index
        sampled_rows.update(random_extra)

    # Final sample
    final_df = df.loc[list(sampled_rows)].copy()
    return final_df.to_dict(orient="records")






def analyze_larger_dataframe(df, API_KEY=GEMINI_API_KEY, delay=2):

    dataset_sample = sample_dataframe_for_ai(df, max_records=500)


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

   - **Date-Time Quality Check**
  - Review all date-like fields for:
    - Inconsistent or invalid formats (e.g., mixed `YYYY-MM-DD`, `DD/MM/YYYY`)
    - Wrong data types (e.g., string instead of datetime)
    - Unrealistic values (e.g., 1800s, 2100s, etc.)
    - Misaligned date logic (e.g., future birth dates, negative durations)


3. **Business Impact & Fix Suggestions**
   - For each issue, include:
     - **Impact**: What effect could this have on business analysis or decision-making?

---
### **Response Format**
# Return a **clean, structured JSON** like this:


### **Output Format Instructions:**

- Return only **clean JSON**. Do NOT include markdown formatting like `**bold**`, `*italic*`, or code fences (no ```json or ```).
- For all `"column"` values, return them as a **comma-separated string** if multiple columns are affected — e.g., `"column": "reporter name, partner name_adj"` instead of a list like `["..."]`.
- Keep all values as simple strings — no lists, markdown, or extra syntax.
- Each recommendation should be a simple string, no bullet points or asterisks.
- Avoid any unnecessary formatting or noise — return raw, readable, clean JSON only.


# Return JSON like:
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
    "recommendations": [{
        "What are the best recommendations to fix the issues."
    }],

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

