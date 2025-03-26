import streamlit as st
import ollama

# Cập nhật tên mô hình nếu cần (đảm bảo bạn đã chạy: ollama pull <model_name>)
MODEL = "llama3.2"

def summarize_text(text):
    system_prompt = (
        "You are an assistant that provides a concise summary of the given text "
        "in markdown format."
    )
    user_prompt = f"Please summarize the following text:\n\n{text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = ollama.chat(model=MODEL, messages=messages)
    return response['message']['content']

# Giao diện Streamlit
st.title("Text Summary App 📝")
st.markdown("Nhập vào đoạn văn bản để nhận bản tóm tắt từ Ollama API.")

# Input văn bản từ người dùng
input_text = st.text_area("Nhập đoạn văn cần tóm tắt:", "")

if st.button("Tóm tắt"):
    if input_text.strip():
        with st.spinner("Đang tóm tắt... ⏳"):
            summary = summarize_text(input_text)
        st.markdown("### 📌 Bản tóm tắt:")
        st.markdown(summary)
    else:
        st.warning("⚠️ Vui lòng nhập đoạn văn!")
