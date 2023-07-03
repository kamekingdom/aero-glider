from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# トークナイザーの準備
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# モデルの準備
model = GPT2LMHeadModel.from_pretrained('gpt2')

max_length = 30

while True:
    # 入力文
    input_text = input("\nType input_text or max_length (or 'q' to quit) >> ")

    if input_text.lower() == 'q':
        break
    # 数字かどうか判別
    if input_text.isdigit():
        max_length = int(input_text)
        print(f"updated: max_length is {max_length} now")
        continue

    # 入力文をトークン化して数値化
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # モデルによる予測
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1, repetition_penalty=1.0, do_sample=True, pad_token_id=model.config.pad_token_id)

    # 予測結果をデコードして表示
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n\nResult: {generated_text}")
