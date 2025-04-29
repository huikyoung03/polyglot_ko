from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

print("🔄 모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-5.8b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/polyglot-ko-5.8b", trust_remote_code=True)
model.eval()
print("✅ 모델 로딩 완료")

class TextRequest(BaseModel):
    text: str

@app.post("/transform/")
async def transform(request: TextRequest):
    prompt = f"다음 문장을 더 자연스럽게 바꿔줘:\n{request.text}\n\n변환된 문장:"

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=200,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"original": request.text, "transformed": result.split("변환된 문장:")[-1].strip()}
