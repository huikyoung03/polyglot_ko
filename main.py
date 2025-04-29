from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

print("🔄 모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained("junelee/ko-llama-1.3b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "junelee/ko-llama-1.3b", trust_remote_code=True, torch_dtype=torch.float16
)
model.eval()
print("✅ 모델 로딩 완료")

class TransformRequest(BaseModel):
    prompt_template: str
    text: str

@app.post("/transform/")
async def transform(req: TransformRequest):
    prompt = req.prompt_template.format(text=req.text)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {
        "original": req.text,
        "prompt_used": prompt,
        "transformed": result.split("### Assistant:")[-1].strip()
    }
