import base64
import io
from openai import OpenAI
from PIL import Image

class VLLMMultimodalClient:
    def __init__(self, base_url="http://localhost:8005/v1", model_name="NCSOFT/VARCO-VISION-2.0-14B"):
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url
        )
        self.model_name = model_name
    
    def _image_to_base64(self, image):
        """Convert PIL Image to base64"""
        if isinstance(image, str):
            image = Image.open(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate(self, image, text_prompt, system_prompt=None, **kwargs):
        """Generate response with multimodal input"""
        base64_image = self._image_to_base64(image)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                },
                {"type": "text", "text": text_prompt}
            ]
        })
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 2048),
            temperature=kwargs.get("temperature", 0.0),
            # Enable thinking mode for compatible models
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature"]}
        )
        
        return response.choices[0].message.content

# Usage example
def main():
    from datasets import load_dataset
    
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )
    
    client = VLLMMultimodalClient()
    dataset = load_dataset("lmms-lab/multimodal-open-r1-8k-verified", split="train[:1%]")
    
    # Single example
    response = client.generate(
        image=dataset[0]["image"],
        text_prompt=dataset[0]["problem"],
        system_prompt=SYSTEM_PROMPT
    )
    print(f"Response: {response}")

if __name__ == "__main__":
    main()