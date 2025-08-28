from crewai import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv

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
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(text):].strip()
