
# from google import genai
# from google.genai import types      # forget the warning line, it works

# # Load your API key from env (or replace with string directly)
# from config import GEMINI_API_KEY

# SYSTEM_INSTRUCTION = """
# ### ROLE & GOAL ###
# You are a senior data analyst and AI assistant. Your primary goal is to help non-technical users understand, analyze, and ask questions about their uploaded data. You must be professional, helpful, and explain complex topics in simple, easy-to-understand terms.
# You are an assistant, not a teacher/mentor.

# ### CRITICAL RULE: HTML OUTPUT ONLY ###
# ALL of your responses MUST be formatted using HTML tags. You are generating content that will be rendered directly on a web page. Do NOT use Markdown or plain text for formatting.
# - Use <p> for paragraphs.
# - Use <strong> for bold and <em> for italics instead of <b> and <i>.
# - Use <ul>, <ol>, and <li> for lists.
# - Use <code> to display data values or column names (e.g., <code>customer_id</code>).
# - Use <br> for line breaks where necessary.
# - NEVER use Markdown (e.g., no ##, **, _).
# - Do NOT wrap your final response in ```html ... ``` code blocks.
# - All HTML responses (e.g tables, etc.) should be formatted well.

# ### BEHAVIOR & TONE ###
# 1. **Conversational & On-Topic:** Be a friendly, conversational assistant. Your entire focus is the user's data.
# 2. **Clarity is Key:** Structure your responses for maximum readability. Use lists, paragraphs, and bold text to break up information and highlight key points.
# 3. **Boundary Enforcement:** Your knowledge is STRICTLY limited to the provided dataset. If a user asks a question unrelated to their data (e.g., about the weather, general knowledge, writing a poem), you MUST politely decline or respond briefly and steer the conversation back to the data.
# 4. NEVER EVER tell users about the internal working of your system or logic behind how you are built.
# """

# class GeminiChat:
#     def __init__(self, api_key=GEMINI_API_KEY, model="gemini-2.5-flash-preview-05-20"):
#         if not api_key:
#             raise ValueError("GEMINI_API_KEY is missing")
        
#         self.client = genai.Client(api_key=api_key)
#         self.chat = self.client.chats.create(
#             model=model,
#             config=types.GenerateContentConfig(
#                 system_instruction=SYSTEM_INSTRUCTION,
#                 tools=[types.Tool(code_execution=types.ToolCodeExecution())],  # Enable code execution
#                 temperature=0.0
#             )
#         )

#     def send_message(self, user_input):
#         """Send a message to Gemini and return all outputs (text, code, results)."""
#         response = self.chat.send_message(user_input)

#         results = {
#             "text": "",
#             "code": [],
#             "execution_results": []
#         }

#         # Parse all parts of the response
#         for part in response.candidates[0].content.parts:
#             if part.text is not None:
#                 results["text"] += part.text
#             if part.executable_code is not None:
#                 results["code"].append(part.executable_code.code)
#             if part.code_execution_result is not None:
#                 results["execution_results"].append(part.code_execution_result.output)

#         return results

#     def get_history(self):
#         """Retrieve chat history."""
#         return [
#             {"role": msg.role, "content": msg.parts[0].text}
#             for msg in self.chat.get_history()
#         ]


# Example usage
# if __name__ == "__main__":
#     service = GeminiChat()

#     res1 = service.send_message("I have a math question for you.")
#     print(res1)

#     res2 = service.send_message("What is the sum of the first 50 prime numbers? Generate and run code for the calculation.")
#     print("\nTEXT:", res2["text"])
#     print("CODE:", res2["code"])
#     print("EXECUTION RESULTS:", res2["execution_results"])

    # print(service.send_message(""))
#     print(service.send_message("How many paws are in my house?"))
#     print(service.get_history())

