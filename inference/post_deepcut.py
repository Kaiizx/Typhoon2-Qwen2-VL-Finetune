from pythainlp.tokenize import word_tokenize
import pandas as pd
from tqdm import tqdm

def truncate_thai_text(text, word_limit=4):
    words = word_tokenize(text, engine="deepcut")  # Use DeepCut for Thai tokenization
    return "".join(words[:word_limit])

df = pd.read_csv('../typhoon2-ft-sub.csv')

for idx,text in enumerate(tqdm(df['caption'])):
    t = truncate_thai_text(text)
    df.loc[idx,'caption'] = t

df.to_csv('../typhoon2-ft-cut4-sub.csv',index=False)
print('Success')
