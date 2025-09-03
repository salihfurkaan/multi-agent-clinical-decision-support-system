from crewai import BaseLLM
from llama_cpp import Llama
from typing import Union, List, Dict, Any, Optional
from dotenv import load_dotenv
import os

load_dotenv()

class QuantizedLLM(BaseLLM):
    def __init__(
            self,
            repo_id: str,
            filename: str,
    ):
        super().__init__(model=f"{repo_id}/{filename}")

        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=4096,
        )



    def call(
            self,
            messages: Union[str, List[Dict[str, str]]],
            callbacks: Optional[List[Any]] = None,
            available_functions: Optional[Dict[str, Any]] = None,
            from_task: Optional[bool] = None,
            from_agent: Optional[bool] = None,
            tools: Optional[List[dict]] = None,
            **kwargs  
    ) -> str:
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = ""
            for message in messages:
                content = message.get("content", "")
                prompt += content + "\n"

        response = self.llm(
            prompt        )

        return response["choices"][0]["text"].strip()

    def supports_function_calling(self) -> bool:
        """Return True if your LLM supports function calling."""
        return False  # Set to False since your LLM doesn't support tools
