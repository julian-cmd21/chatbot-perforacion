import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(
    page_title="Chatbot de Perforación Petrolera",
    page_icon="🛢️",
    layout="centered"
)

st.title("🛢️ Chatbot de Perforación Petrolera")
st.markdown("Asistente especializado en ingeniería de perforación. Consulta parámetros de pozos, NPT y datos de correlación.")

HF_TOKEN = st.secrets["HF_TOKEN"]
MODEL_ID = "Julian992992/llama-3-8b-chat-Perforation-Dataset"

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

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

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = client.chat_completion(
                messages=messages,
                max_tokens=300,
                temperature=0.7,
            )
            reply = response.choices[0].message.content
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
