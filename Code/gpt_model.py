
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from textblob import TextBlob  # مكتبة تحليل المشاعر
import streamlit as st

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

tokenizer.pad_token = tokenizer.eos_token

motivational_quotes_positive = [
    "Fear is not your enemy—it's the compass pointing you toward growth.",
    "The wall of fear seems high until you climb it.",
    "Stars can’t shine without darkness. Your struggle is just the universe making space for your light.",
    "Growth is born in the soil of struggle, watered by hope. Keep planting.",
    "Even the tallest trees were once seeds that refused to stay underground.",
    "Success is the sum of small efforts, repeated day in and day out.",
    "The best way to predict the future is to create it.",
    "Success is not final, failure is not fatal: It is the courage to continue that counts."
]

motivational_quotes_negative = [
    "Failure is fertilizer for success. What feels like an ending now will feed your growth tomorrow.",
    "You are not a drop in the ocean—you are the ocean in a drop.",
    "This test is just one chapter in your story, not the whole book.",
    "Giving up is a chapter you don’t want to end with. The next page might say: And then everything changed.",
    "Sometimes, you have to go through the worst to get to the best.",
    "The darkest hour is just before the dawn.",
    "Great things are not accomplished in a day. You need to keep going, even when it's tough.",
    "Don't let your mistakes define you. Let them teach you."
]


def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  
    return sentiment


def generate_motivational_quote(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return full_text[len(prompt):].strip()


def get_keyword_based_quote(user_input):
    positive_keywords = ['sucess', 'optimism', 'strength', 'ambition', 'positive', 'opportunity','happy']
    negative_keywords = ['failure', 'weakness', 'sad', 'frustration', 'loss', 'difficulty']

    for keyword in positive_keywords:
        if keyword in user_input:
            return random.choice(motivational_quotes_positive)

    for keyword in negative_keywords:
        if keyword in user_input:
            return random.choice(motivational_quotes_negative)

    return None  

st.set_page_config(page_title="ChatGPT", layout="centered")
st.title("Motivation Chat GPT-2🥳")

user_input = st.text_input("Write Your Feeling: 💬")
use_gpt2 = st.checkbox("استخدام GPT-2 للتوليد الآلي", value=False)

if st.button("Submit 🎯"):
    if user_input.strip() == "":
        st.warning("Please Write Anythings ")
    else:
        sentiment = analyze_sentiment(user_input)

        keyword_quote = get_keyword_based_quote(user_input)

        if keyword_quote:
            response = keyword_quote
        elif sentiment > 0:
            response = random.choice(motivational_quotes_positive)
        elif sentiment < 0:
            response = random.choice(motivational_quotes_negative)
        else:
            response = "It's okay, we all go through similar moments. Keep trying!"

        if use_gpt2:
            prompt = f"User: {user_input}\nGPT-2:"
            response = generate_motivational_quote(prompt)

        st.success("💡 Motivation Quote ")
        st.markdown(f"> {response}")
