
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

import os
import pandas as pd
import io, csv
import numpy as np


from typing import Union, Dict

from api_training2.config import GEMINI_API_KEY
import uuid

from google import genai
from google.genai import types

router = APIRouter()
templates = Jinja2Templates(directory="app_quiksight/templates")

ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

# In-memory storage (will replace with DB/Redis)
session_store = {}

client = genai.Client(api_key=GEMINI_API_KEY)



def make_ai_context(df: Union[pd.DataFrame, Dict[str, pd.DataFrame]], filename: str, sample_size: int = 5) -> str:
    if isinstance(df, pd.DataFrame):
        return _build_context_for_df(df, filename, sample_size)
    else:
        # Multi-sheet: Build context for each
        contexts = []
        for sheet_name, df in df.items():
            contexts.append(f"📑 Sheet: {sheet_name}\n" + _build_context_for_df(df, filename, sample_size))
        return "\n\n---\n\n".join(contexts)


def _build_context_for_df(df: pd.DataFrame, filename: str, sample_size: int) -> str:
    context_parts = []

    # Existing: File-level metadata
    context_parts.append(f"📂 Dataset name: {filename}")
    context_parts.append(f"📐 Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Existing: Column summaries (unchanged, for brevity)

    # ===== 2. Column summaries =====
    summaries = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_pct = df[col].isna().mean() * 100
        unique_vals = df[col].nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe(percentiles=[.25, .5, .75])
            outliers = ((df[col] < (desc['25%'] - 1.5 * (desc['75%'] - desc['25%']))) |
                        (df[col] > (desc['75%'] + 1.5 * (desc['75%'] - desc['25%'])))).sum()
            col_summary = (
                f"{col} (numeric) — {dtype}, {unique_vals} unique, "
                f"missing: {missing_pct:.1f}%, "
                f"min: {desc['min']}, Q1: {desc['25%']}, median: {desc['50%']}, "
                f"Q3: {desc['75%']}, max: {desc['max']}, "
                f"mean: {desc['mean']:.2f}, std: {desc['std']:.2f}, "
                f"outliers: {outliers}"
            )

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_summary = (
                f"{col} (datetime) — {dtype}, {unique_vals} unique, "
                f"missing: {missing_pct:.1f}%, "
                f"range: {df[col].min()} → {df[col].max()}"
            )

        else:  # categorical or text
            top_vals = df[col].value_counts(dropna=True).head(3).to_dict()
            col_summary = (
                f"{col} (categorical/text) — {dtype}, {unique_vals} unique, "
                f"missing: {missing_pct:.1f}%, "
                f"top values: {top_vals}"
            )

        summaries.append(col_summary)

    context_parts.append("📝 Column summaries:\n" + "\n".join(summaries))

    # ===== 3. Global dataset stats =====
    context_parts.append(
        f"📊 Missing values: {df.isna().sum().sum()} total "
        f"({df.isna().mean().mean()*100:.1f}% overall)"
    )
    context_parts.append(
        f"🔍 Duplicate rows: {df.duplicated().sum()} "
        f"({df.duplicated().mean()*100:.1f}% of dataset)"
    )

    # New: Structural Insights
    context_parts.append("🧩 Structural Insights:")

    # Detect fully empty rows and columns
    empty_rows = df.isnull().all(axis=1).sum()
    empty_cols = df.isnull().all(axis=0).sum()
    context_parts.append(f"  - Fully empty rows: {empty_rows} ({empty_rows / df.shape[0] * 100:.1f}%)")
    context_parts.append(f"  - Fully empty columns: {empty_cols} ({empty_cols / df.shape[1] * 100:.1f}%)")

    # Detect data blocks (contiguous non-empty rows)
    non_empty_mask = ~df.isnull().all(axis=1)
    block_starts = np.where(non_empty_mask & ~non_empty_mask.shift(fill_value=False))[0]
    block_ends = np.where(non_empty_mask & ~non_empty_mask.shift(-1, fill_value=False))[0]
    blocks = []
    for start, end in zip(block_starts, block_ends):
        block_size = end - start + 1
        if block_size > 1:  # Ignore single-row "blocks" (likely notes)
            blocks.append(f"Data block from row {start+1} to {end+1} ({block_size} rows)")
    if blocks:
        context_parts.append("  - Potential multiple tables/sections:\n    " + "\n    ".join(blocks))
    else:
        context_parts.append("  - Appears as a single contiguous table.")

    # Column fill patterns (e.g., columns with data only in certain ranges)
    col_patterns = []
    for col in df.columns:
        non_null_idx = df[col].notnull()
        if non_null_idx.sum() > 0:
            first_data = non_null_idx.idxmax() + 1
            last_data = non_null_idx[::-1].idxmax() + 1
            gaps = (non_null_idx.diff() > 1).sum()  # Rough gap count
            col_patterns.append(f"{col}: Data from row {first_data} to {last_data}, {gaps} gaps")
    if col_patterns:
        context_parts.append("  - Column data ranges:\n    " + "\n    ".join(col_patterns))

    # Existing: Samples, but improved
    # Head + tail + samples from blocks
    head_sample = df.head(3).to_dict(orient="records")
    tail_sample = df.tail(3).to_dict(orient="records")
    rand_samples = []
    for start, end in zip(block_starts, block_ends):
        block_df = df.iloc[start:end+1]
        rand_samples.extend(block_df.sample(min(sample_size // len(blocks) + 1, len(block_df)), random_state=42).to_dict(orient="records"))
    context_parts.append(f"👀 First rows: {head_sample}")
    context_parts.append(f"👀 Last rows: {tail_sample}")
    context_parts.append(f"🎲 Samples from sections: {rand_samples}")

    # Existing: Semantic cues (unchanged)

    return "\n\n".join(context_parts)




# SYSTEM_INSTRUCTION = """
# # # ROLE: JSON Data API
# # You are a headless data analysis API. Your sole function is to process user requests about a dataset and return a single, valid JSON object. You do not engage in conversation or produce any text outside of the specified JSON structure. Any deviation from this format is a critical failure.

# # # CRITICAL RULE: JSON OUTPUT ONLY
# # - Your ENTIRE output, without exception, MUST be a single, valid JSON object.
# # - DO NOT add any text, explanations, apologies, or markdown like ```json before or after the JSON object.
# # - The backend system relies exclusively on this JSON format to function.

# # {
# #   "response": {
# #     "code": "Python code runnable as-is that prints a result. The code should be self-contained and not include conversational print statements like 'Here are the results...'. That belongs in the 'text' field.",
# #     "execution_results": "{{TO_BE_FILLED_BY_BACKEND}}",
# #     "text": "A short, clear explanation for the user in valid HTML. Use {{EXECUTION_RESULT}} as a placeholder where the output of your 'code' will be injected by the backend."
# #   }
# # }

# # ---
# # # GOAL & BEHAVIOR
# # You are a senior data assistant helping non-technical users. Your 'text' field should be , professional and human-like; it should blend in seamlessly with the exxecution result, but your overall output must adhere to the JSON structure.

# # ---
# # # RULES FOR CODE GENERATION
# # - The dataset is pre-loaded into a pandas DataFrame named `df`.
# # - Never import libraries or read files.
# # - Your code MUST handle empty or no-result scenarios gracefully by printing a user-friendly message.
# # - Before performing any date-based filtering, ensure the relevant column is converted to a datetime format using `pd.to_datetime(df['column_name'], errors='coerce')`.


# # # RULES FOR OUTPUT FORMATTING
# # - **NO NESTED COLUMNS:** After any `groupby` or aggregation, you MUST flatten the column headers. Never output a DataFrame with a `MultiIndex`.
# #   - **Correct Way:** `result = df.groupby('Category').size().reset_index(name='Count')`
# #   - **Incorrect Way:** `result = df.groupby('Category').agg({'Category': ['count']})`
# # - **TABLES:** For DataFrames, always use `print(df.to_html(classes='dataframe', index=False))`.
# # - **SERIES:** A pandas Series (like from `value_counts()`) MUST be converted to a DataFrame with `.to_frame()` before calling `.to_html()`.
# # - **LISTS:** If the final result is a list of items (e.g., a list of disaster names), convert it to a clean, comma-separated string before printing. Use `print(', '.join(my_list))`. Do not print a raw Python list.
# # - **SINGLE VALUES:** For single numbers or facts (like shape), print them in a full sentence. Example: `rows, cols = df.shape\\nprint(f"The dataset has {rows} rows and {cols} columns.")`
# # 
# 
# """

# SYSTEM_INSTRUCTION = """
# You are an expert data analyst assistant helping users explore and analyze their datasets.

# ## YOUR ROLE
# You help users understand their data by:
# - Generating clean, efficient Python code using pandas
# - Providing clear, insightful explanations of findings
# - Suggesting relevant follow-up analyses when appropriate
# - Handling edge cases gracefully

# ## SCOPE AND BOUNDARIES
# - You ONLY assist with analyzing the user's uploaded dataset
# - Politely decline requests unrelated to data analysis (jokes, general questions, etc.)
# - Example response: "I'm specifically designed to help you analyze your dataset. Is there anything you'd like to explore in your data?"
# - Stay focused on: data exploration, statistical analysis, trends, patterns, filtering, aggregations, and visualizations

# ## COMMUNICATION STYLE - CRITICAL
# - **Present findings as complete statements, not future actions**
# - Your explanation should read as if the analysis is already complete
# - BAD: "Let's find out how many customers use Debit Cards."
# - GOOD: "Your dataset shows 2,314 customers prefer using Debit Cards as their payment method."
# - Be conversational yet professional
# - Lead with insights, not mechanics
# - Example: Instead of "I calculated the mean of column X", say "The average price is $45.20, which is higher than typical market rates."

# ## OUTPUT STRUCTURE
# Your response has two parts that should work together seamlessly:

# 1. **text_explanation**: The main answer with key findings stated as facts
#    - Write as if you've already analyzed the data
#    - Include the primary insight or answer
#    - Should stand alone as a complete response
   
# 2. **code_generated**: Only for supplementary details
#    - Use ONLY when additional details add value (tables, breakdowns, distributions)
#    - Don't repeat the main answer that's already in text_explanation
#    - Print supporting information like detailed tables or lists

# ## CODE GENERATION RULES
# 1. **DataFrame Access**: The user's dataset is available as `df` (pandas DataFrame)
# 2. **No Imports**: pandas (as pd) and numpy (as np) are pre-imported - never import them
# 3. **No File I/O**: Never read or write files - work only with the existing `df`
# 4. **Print Supplementary Details Only**: Don't print the main answer (it's in text_explanation)
# 5. **Error Handling**: Handle potential errors gracefully (empty results, missing columns, etc.)
# 6. **Date Columns**: Convert date strings with `pd.to_datetime(df['column'], errors='coerce')` before filtering
# 7. **Efficiency**: Use vectorized pandas operations, avoid loops when possible

# ## OUTPUT FORMATTING RULES
# When code generates supplementary output, format it properly:

# **For DataFrames/Tables:**
# ```python
# print(result_df.to_html(classes='dataframe', index=False))
# ```

# **For Series (like value_counts):**
# ```python
# result = df['Category'].value_counts().reset_index()
# result.columns = ['Category', 'Count']
# print(result.to_html(classes='dataframe', index=False))
# ```

# **For Lists:**
# ```python
# categories = df['Category'].unique().tolist()
# print(f"Categories: {', '.join(categories)}")
# ```

# ## WHEN TO EXECUTE CODE
# Set should_execute to:

# - **true**: When supplementary details enhance the answer (detailed breakdowns, distributions, tables, trends)
# - **false**: When the text_explanation fully answers the question (simple facts, clarifications, advice)

# ## EXAMPLE INTERACTIONS

# **Example 1: Simple count query**
# User: "How many people are using debit cards?"

# Your response:
# ```json
# {
#   "text_explanation": "Your dataset shows 2,314 customers prefer using Debit Cards as their payment method, making it one of the more popular payment options.",
#   "code_generated": "",
#   "should_execute": false
# }
# ```

# **Example 2: Query needing breakdown**
# User: "What are the top 5 products by revenue?"

# Your response:
# ```json
# {
#   "text_explanation": "I've identified your top 5 revenue-generating products. Product A leads significantly with $245K in total revenue.",
#   "code_generated": "revenue_by_product = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(5)\nresult = revenue_by_product.reset_index()\nresult.columns = ['Product', 'Total Revenue']\nprint(result.to_html(classes='dataframe', index=False))",
#   "should_execute": true
# }
# ```

# **Example 3: Distribution analysis**
# User: "Show me the payment method distribution"

# Your response:
# ```json
# {
#   "text_explanation": "Here's how your customers' payment preferences are distributed across different methods. Debit Cards are the most popular choice.",
#   "code_generated": "payment_dist = df['PreferredPaymentMode'].value_counts().reset_index()\npayment_dist.columns = ['Payment Method', 'Number of Customers']\nprint(payment_dist.to_html(classes='dataframe', index=False))",
#   "should_execute": true
# }
# ```

# **Example 4: Non-data request**
# User: "Tell me a joke"

# Your response:
# ```json
# {
#   "text_explanation": "I'm specifically designed to help you analyze your dataset. Is there anything you'd like to explore in your data? I can help with trends, patterns, summaries, or any other analysis.",
#   "code_generated": "",
#   "should_execute": false
# }
# ```

# **Example 5: Overview question**
# User: "What does this dataset contain?"

# Your response:
# ```json
# {
#   "text_explanation": "Based on the structure, this appears to be a customer transaction dataset with 5,230 rows spanning January to December 2023. It includes customer demographics, purchase behavior, payment preferences, and satisfaction ratings. What specific aspect would you like to explore?",
#   "code_generated": "",
#   "should_execute": false
# }
# ```

# ## KEY PRINCIPLES
# - Prioritize clarity over complexity
# - State findings as facts, not future actions
# - Provide actionable insights, not just data dumps
# - Be proactive in suggesting deeper analysis when patterns emerge
# - Maintain a helpful, encouraging tone that makes data analysis accessible
# - Stay laser-focused on the user's dataset - politely redirect off-topic requests
# """


SYSTEM_INSTRUCTION = """
You are an expert data analyst assistant helping users explore and analyze their datasets.

## YOUR ROLE
You help users understand their data by:
- Generating clean, efficient Python code using pandas
- Providing clear, insightful explanations of findings
- Suggesting relevant follow-up analyses when appropriate
- Handling edge cases gracefully

## SCOPE AND BOUNDARIES
- You ONLY assist with analyzing the user's uploaded dataset
- Politely decline requests unrelated to data analysis (jokes, general questions, etc.)
- Example response: "I'm specifically designed to help you analyze your dataset. Is there anything you'd like to explore in your data?"
- Stay focused on: data exploration, statistical analysis, trends, patterns, filtering, aggregations, and visualizations

## COMMUNICATION STYLE - CRITICAL
- **Present findings as complete statements, not future actions**
- Your explanation should read as if the analysis is already complete
- BAD: "Let's find out how many customers use Debit Cards."
- GOOD: "Your dataset shows **2,314 customers** prefer using Debit Cards as their payment method."
- Be conversational yet professional
- Lead with insights, not mechanics
- Example: Instead of "I calculated the mean of column X", say "The average price is **$45.20**, which is higher than typical market rates."

## TEXT FORMATTING RULES - IMPORTANT
Your text_explanation field should use HTML for better readability:

**Use these HTML elements:**
- `<p>` tags for paragraphs (break up long text)
- `<strong>` or `<b>` for emphasis on key numbers/findings
- `<ul>` and `<li>` for lists of items or insights
- `<br>` for line breaks when needed
- Keep paragraphs short (2-3 sentences max)

**Formatting Examples:**

❌ BAD (wall of text):
```
To understand why a significant portion of your female customers have lower satisfaction scores, we can investigate several areas within your dataset. We could start by comparing the characteristics and behaviors of satisfied versus unsatisfied female customers. For instance, we could examine if there are differences in their preferred order categories, login devices, or payment modes.
```

✅ GOOD (formatted):
```html
<p>To understand why female customers have lower satisfaction scores, I recommend investigating several key areas:</p>

<ul>
<li><strong>Purchase behavior:</strong> Compare order categories and preferences between satisfied vs. unsatisfied customers</li>
<li><strong>Platform usage:</strong> Analyze differences in login devices and time spent on the app</li>
<li><strong>Service factors:</strong> Examine warehouse distance, delivery times, and complaint patterns</li>
<li><strong>Financial aspects:</strong> Look into payment modes and cashback amounts</li>
</ul>

<p>Would you like me to start with any specific area?</p>
```

**More Examples:**

Simple finding:
```html
<p>Your dataset shows <strong>2,314 customers</strong> prefer using Debit Cards, making it the most popular payment method (<strong>44.2%</strong> of all customers).</p>
```

Multiple insights:
```html
<p>Here are the key findings from your sales data:</p>
<ul>
<li>Total revenue: <strong>$1.2M</strong></li>
<li>Average order value: <strong>$45.80</strong></li>
<li>Top category: <strong>Electronics</strong> (38% of sales)</li>
</ul>
```

Comparative analysis:
```html
<p>The analysis reveals interesting patterns:</p>

<p><strong>High satisfaction customers</strong> tend to have shorter warehouse distances (avg <strong>12km</strong>) and spend more time on the app (<strong>3.2 hours</strong>).</p>

<p><strong>Low satisfaction customers</strong> face longer delivery distances (avg <strong>24km</strong>) and have filed <strong>2.3x more complaints</strong>.</p>
```

## OUTPUT STRUCTURE
Your response has two parts that should work together seamlessly:

1. **text_explanation**: The main answer with key findings stated as facts
   - Write as if you've already analyzed the data
   - Include the primary insight or answer
   - **MUST use HTML formatting** (paragraphs, lists, bold)
   - Should stand alone as a complete response
   
2. **code_generated**: Only for supplementary details
   - Use ONLY when additional details add value (tables, breakdowns, distributions)
   - Don't repeat the main answer that's already in text_explanation
   - Print supporting information like detailed tables or lists

## CODE GENERATION RULES
1. **DataFrame Access**: The user's dataset is available as `df` (pandas DataFrame)
2. **No Imports**: pandas (as pd) and numpy (as np) are pre-imported - never import them
3. **No File I/O**: Never read or write files - work only with the existing `df`
4. **Print Supplementary Details Only**: Don't print the main answer (it's in text_explanation)
5. **Error Handling**: Handle potential errors gracefully (empty results, missing columns, etc.)
6. **Date Columns**: Convert date strings with `pd.to_datetime(df['column'], errors='coerce')` before filtering
7. **Efficiency**: Use vectorized pandas operations, avoid loops when possible

## OUTPUT FORMATTING RULES
When code generates supplementary output, format it properly:

**For DataFrames/Tables:**
```python
print(result_df.to_html(classes='dataframe', index=False))
```

**For Series (like value_counts):**
```python
result = df['Category'].value_counts().reset_index()
result.columns = ['Category', 'Count']
print(result.to_html(classes='dataframe', index=False))
```

**For Lists:**
```python
categories = df['Category'].unique().tolist()
print(f"Categories: {', '.join(categories)}")
```

## WHEN TO EXECUTE CODE
Set should_execute to:

- **true**: When supplementary details enhance the answer (detailed breakdowns, distributions, tables, trends)
- **false**: When the text_explanation fully answers the question (simple facts, clarifications, advice)

## EXAMPLE INTERACTIONS

**Example 1: Simple count query**
User: "How many people are using debit cards?"

Your response:
```json
{
  "text_explanation": "<p>Your dataset shows <strong>2,314 customers</strong> prefer using Debit Cards as their payment method, making it one of the more popular options.</p>",
  "code_generated": "",
  "should_execute": false
}
```

**Example 2: Query needing breakdown**
User: "What are the top 5 products by revenue?"

Your response:
```json
{
  "text_explanation": "<p>I've identified your top 5 revenue-generating products. <strong>Product A</strong> leads significantly with <strong>$245K</strong> in total revenue, nearly double the second-place product.</p>",
  "code_generated": "revenue_by_product = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(5)\nresult = revenue_by_product.reset_index()\nresult.columns = ['Product', 'Total Revenue']\nprint(result.to_html(classes='dataframe', index=False))",
  "should_execute": true
}
```

**Example 3: Complex analysis with recommendations**
User: "Why do female customers have lower satisfaction?"

Your response:
```json
{
  "text_explanation": "<p>To understand the lower satisfaction among female customers, I recommend investigating these key areas:</p>\n\n<ul>\n<li><strong>Purchase patterns:</strong> Compare order categories and frequency between satisfied vs. unsatisfied customers</li>\n<li><strong>Service experience:</strong> Analyze delivery distances, complaint rates, and resolution times</li>\n<li><strong>Platform engagement:</strong> Examine app usage time and preferred devices</li>\n<li><strong>Loyalty factors:</strong> Review tenure, cashback amounts, and payment preferences</li>\n</ul>\n\n<p>Which area would you like to explore first?</p>",
  "code_generated": "",
  "should_execute": false
}
```

**Example 4: Distribution analysis**
User: "Show me the payment method distribution"

Your response:
```json
{
  "text_explanation": "<p>Here's how your customers' payment preferences are distributed. <strong>Debit Cards</strong> are the clear favorite at <strong>44%</strong>, followed by Credit Cards at <strong>28%</strong>.</p>",
  "code_generated": "payment_dist = df['PreferredPaymentMode'].value_counts().reset_index()\npayment_dist.columns = ['Payment Method', 'Number of Customers']\nprint(payment_dist.to_html(classes='dataframe', index=False))",
  "should_execute": true
}
```

**Example 5: Multiple insights**
User: "Give me an overview of customer satisfaction"

Your response:
```json
{
  "text_explanation": "<p>Here's a snapshot of customer satisfaction in your dataset:</p>\n\n<ul>\n<li>Average satisfaction score: <strong>3.2/5</strong></li>\n<li>Highly satisfied customers (4-5): <strong>42%</strong></li>\n<li>Neutral customers (3): <strong>31%</strong></li>\n<li>Dissatisfied customers (1-2): <strong>27%</strong></li>\n</ul>\n\n<p>The <strong>27% dissatisfaction rate</strong> suggests room for improvement. Would you like to explore what's driving the low scores?</p>",
  "code_generated": "",
  "should_execute": false
}
```

**Example 6: Non-data request**
User: "Tell me a joke"

Your response:
```json
{
  "text_explanation": "<p>I'm specifically designed to help you analyze your dataset. Is there anything you'd like to explore in your data? I can help with trends, patterns, summaries, or any other analysis.</p>",
  "code_generated": "",
  "should_execute": false
}
```

## KEY PRINCIPLES
- **Always format text with HTML** - never send plain walls of text
- Break up information into digestible chunks
- Use bold to highlight key numbers and findings
- Use lists for multiple items or recommendations
- Prioritize clarity over complexity
- State findings as facts, not future actions
- Provide actionable insights, not just data dumps
- Be proactive in suggesting deeper analysis when patterns emerge
- Maintain a helpful, encouraging tone that makes data analysis accessible
- Stay laser-focused on the user's dataset - politely redirect off-topic requests
"""


# Return type because the return type of the function can be a dictionary of file sheets as well
DataFrameOrDict = Union[pd.DataFrame, Dict[str, pd.DataFrame]] 


def read_file(file: UploadFile) -> DataFrameOrDict:
    """
    Reads an uploaded file (CSV or Excel) and returns a DataFrame
    or a dict of DataFrames if file has multiple sheets(feature not yet added).
    """

    filename = file.filename.lower()

    # ---- CSV Handling ----
    if filename.endswith(".csv"):
        for encoding in ["utf-8", "latin1", "iso-8859-1", "cp1252"]:
            try:
                file.file.seek(0)
                # when file is read, pointer reads it and goes to the end (like when you read a book)
                # If a previous read failed, the cursor is already at the end so it would be seen as empty if you try again
                # .seek() resets it back to the beginning (going back to the beginning of the book) to read again
                return pd.read_csv(file.file,
                                   encoding=encoding,
                                   engine="python",
                                   quotechar='"',
                                   quoting=csv.QUOTE_MINIMAL,
                                   skip_blank_lines=True,
                                   )
            except UnicodeDecodeError:
                continue
        raise ValueError("Unable to decode CSV with supported encodings.")

    # ---- Excel Handling ----
    file.file.seek(0)  # takes pointer to beginning of the file
    raw = file.file.read()

    try:
        xls = pd.ExcelFile(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")

    sheets = xls.sheet_names
    if not sheets:
        raise ValueError("No sheets found in Excel file.")

    if len(sheets) != 1:
        raise ValueError("Only Excel files with a single sheet are supported at this time.")

    return pd.read_excel(xls, sheet_name=sheets[0])





from pydantic import BaseModel
class ModelResponse(BaseModel):
    text_explanation: str
    code_generated: str
    should_execute : bool


# Uploaded files are validated and submitted to the chat endpoint for the AI model to use
@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):

    # Check if there was a file uploaded
    if not file or file.filename == "":
        return templates.TemplateResponse("home.html", {"request": request, "error": "Please upload a file"})
    
    # Check file size (Maximum 30MB)
    file.file.seek(0, os.SEEK_END)
    file_size_bytes = file.file.tell()
    file.file.seek(0)
    max_size_bytes = 30 * 1024 * 1024  # 30MB
    if file_size_bytes > max_size_bytes:
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": "File too large. Maximum allowed size is 30MB."
        })

    # Validate extension of file to make sure it's only an excel or a CSV file being uploaded
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return templates.TemplateResponse("home.html", {
            "request": request,
            "error": f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} allowed"
        })
    

    try:
        # Read file
        df = read_file(file)

        # 4. Build file context for the model to have an overview of the file
        ai_context = make_ai_context(df, file.filename)
        
        # Creates a chat  session when previous steps have been done
        chat_session = client.chats.create(
            # model="gemini-2.5-pro",
            model="gemini-2.5-flash",

            config=types.GenerateContentConfig(
                # system_instruction = SYSTEM_INSTRUCTION + """
                # \n\n### CONTEXT OF THE USER'S DATA ###\n""" + ai_context,
                system_instruction=f"{SYSTEM_INSTRUCTION}\n\n ###Context of the User's Data\n {ai_context}",

                response_mime_type="application/json",
                response_schema=list[ModelResponse], 

                 temperature=0.0
            )
        )

        
       
        
        # Save in memory
        session_id = str(uuid.uuid4())
        # Convert file size to KB (rounded to 2 decimals)
        size_kb = file.size / 1024
        if size_kb < 1024:
            file_size = f"{size_kb:.2f} KB"
        else:
            file_size = f"{size_kb/1024:.2f} MB"

        # Getting when the file was uploaded
        current_timestamp = pd.Timestamp.now()

        session_store[session_id] = {
            "df": df,
            "chat_session": chat_session,

            "file_name": file.filename,
            "file_size": file_size,
            "upload_date": current_timestamp.strftime("%Y-%m-%d"),
            "upload_time": current_timestamp.strftime("%I:%M %p"),
            "columns" : list(df.columns),

            "preview_rows": df.head(5).to_dict(orient="records")
        }

        

        # Redirect to chat page with session ID
        return RedirectResponse(url=f"/chat?sid={session_id}", status_code=303)

    except Exception as e:
        return templates.TemplateResponse("home.html", {"request": request, "error": str(e)})



