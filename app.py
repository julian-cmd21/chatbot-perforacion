import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

st.set_page_config(
    page_title="Chatbot de Perforación Petrolera",
    page_icon="🛢️",
    layout="centered"
)

st.title("🛢️ Chatbot de Perforación Petrolera")
st.markdown("Asistente especializado en ingeniería de perforación. Consulta parámetros de pozos, NPT y datos de correlación.")

MODEL_ID = "Julian992992/llama-3-8b-chat-Perforation-Dataset"
HF_TOKEN = st.secrets["HF_TOKEN"]

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN
    )
    return model, tokenizer

model, tokenizer = load_model()

SYSTEM_PROMPT = """Eres un asistente experto en ingeniería de perforación petrolera. 
Responde preguntas técnicas sobre operaciones de perforación, parámetros de pozos, 
NPT, y datos de pozos de correlación de forma precisa y concisa."""

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta sobre perforación..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in st.session_state.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("assistant")[-1].strip()
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
