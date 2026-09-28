# 데이터 로드
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 모델 로드
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 입력 텍스트
text = "The weather is"
input_ids = tokenizer.encode(text, return_tensors="pt")

# 다음 토큰 예측
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]  # 마지막 위치의 로짓
    probabilities = torch.softmax(logits, dim=-1)

# 상위 5개 토큰
top5_probs, top5_ids = torch.topk(probabilities, 5)

print("다음 토큰 예측 (Top 5):")
for prob, token_id in zip(top5_probs[0], top5_ids[0]):
    token = tokenizer.decode([token_id])
    print(f"{token:15s} : {prob:.4f}")
s