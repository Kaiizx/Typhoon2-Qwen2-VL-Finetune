from openai import OpenAI
import pandas as pd
import os
import re
from tqdm import tqdm
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def infer(image_path):
    instruction = 'You are image captioning model that answer in Thai.'
    image_url = f'file://{image_path}'
    chat_response = client.chat.completions.create(
        model="~/Qwen2-VL-Finetune/merged_weight/typhoon2-ft-3e",
        messages=[
            
            {"role": "system", "content": instruction},
            {
            "role": "user",
            "content": [
                # NOTE: The prompt formatting with the image token `<image>` is not needed
                # since the prompt will be processed automatically by the API server.
                # {"type": "text", "text": "จงบรรยายรูปภาพนี้เท่านั้นโดยที่ไม่ต้องมีคำนำหน้าเช่น `รูปภาพนี้แสดงถึง` หรือ `ภาพนี้แสดงให้เห็น`"},
                {"type": "text", "text": "จงอธิบายรูปภาพนี้"},  
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
        max_tokens=32
    )
    return str(chat_response.choices[0].message.content)
# print("Chat completion output:", chat_response.choices[0].message.content)

# Single-image input inference
dir_path = '~/Qwen2-VL-Finetune/inference/test'
test_df = pd.read_csv('sample_submission.csv')
test_df['image_id'] = test_df['image_id'].astype(str).str.zfill(5)

for idx,path in enumerate(tqdm(test_df['image_id'])):
    img_path = os.path.join(dir_path,str(path))
    img_path += '.jpg'
    # print(img_path)
    res = infer(img_path)
    res = str(res.replace('\n',''))
    # print(res)
    test_df.loc[idx,'caption'] = res

test_df.to_csv('result/typhoon2-ft-10tk-sub.csv',index=False)
print('Success')

