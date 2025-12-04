import google.generativeai as genai
from config.config import *


class LLM:
    def __init__(self, model_name=MODEL_NAME_LLM, api_key: str = None):
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel(model_name=model_name)
        self.template = None

    def get_query_prompt(self, question: str):
        self.template = {question}
        return self.template

    def post_request(self, prompt: str):
        response = self.llm.generate_content(prompt)
        return response.text