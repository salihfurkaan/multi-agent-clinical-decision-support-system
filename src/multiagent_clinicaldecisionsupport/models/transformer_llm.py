from crewai import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
import os

load_dotenv()

class TransformersLLM(BaseLLM):
    def __init__(self, model_name: str, temperature: float = 0.7):
        super().__init__(model=model_name, temperature=temperature)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kwargs):
        if isinstance(messages, str):
            text = messages
        else:
            text = messages[-1]["content"]
            
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=self.temperature,
                do_sample=True,
                max_new_tokens=4096,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(text):].strip()

    def supports_function_calling(self) -> bool:
        # Since you're using MCP tools, return True
        return True
