from openai import OpenAI
import os

## Set the API key and model name
MODEL="gpt-4o-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
completion = client.chat.completions.create(
  model=MODEL,
  messages=[
    {"role": "system", "content": "You are an expert bilingual translator specializing in English to Korean translation."},
    {"role": "user", "content": """
Your task is to accurately translate English sentences into Korean. Here are some examples:

Example 1:
- English: "Hello, how are you?"
- Korean: "안녕하세요, 어떻게 지내세요?"

Example 2:
- English: "The weather is nice today."
- Korean: "오늘 날씨가 좋네요."

Example 3:
- English: "I would like to order a coffee."
- Korean: "커피 한 잔 주문하고 싶어요."

Example 4:
- English: "Can you help me with this?"
- Korean: "이거 좀 도와주실 수 있나요?"

Example 5:
- English: "Where is the nearest hospital?"
- Korean: "가장 가까운 병원이 어디에 있나요?"

Now, translate the following sentence:

- English: "dog"
- Korean: 
    """}
  ]
)

translation = completion.choices[0].message.content
print("Translation: " + translation)