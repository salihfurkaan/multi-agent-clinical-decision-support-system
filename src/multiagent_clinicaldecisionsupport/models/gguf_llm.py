from crewai import BaseLLM
from llama_cpp import Llama
from typing import Union, List, Dict, Any, Optional
from dotenv import load_dotenv

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
            filename=filename
        )
        
    def call(
        self, 
        messages: Union[str, List[Dict[str, str]]], 
        tools=None, 
        callbacks=None, 
        available_functions=None
    ) -> str:
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = ""
            for message in messages:
                content = message.get("content", "")
                prompt += content + "\n"
        
        response = self.llm(
            prompt,
            max_tokens=512,
            echo=False
        )
        
        return response["choices"][0]["text"].strip()
