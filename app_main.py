import streamlit as st
import ollama
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

# Cấu hình model Ollama
MODEL = "llama3.2"

# ==================== HÀM TÓM TẮT VĂN BẢN ==================== #
def summarize_text(text):
    """Tóm tắt văn bản nhập vào."""
    system_prompt = (
        "You are an assistant that provides a concise summary of the given text in markdown format."
    )
    user_prompt = f"Please summarize the following text:\n\n{text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = ollama.chat(model=MODEL, messages=messages)
    return response['message']['content']

# ==================== HÀM TÓM TẮT WEBSITE ==================== #
class Website:
    """Lấy nội dung trang web và trích xuất văn bản bằng Selenium & BeautifulSoup."""
    def __init__(self, url):
        self.url = url
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        try:
            driver.get(url)
            html = driver.page_source
        except Exception as e:
            driver.quit()
            raise RuntimeError(f"Could not load page: {e}")
        driver.quit()

        soup = BeautifulSoup(html, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

def summarize_website(url):
    """Tóm tắt nội dung trang web từ URL."""
    try:
        website = Website(url)
        messages = [
            {"role": "system", "content": "You are an AI that summarizes webpage content in markdown."},
            {"role": "user", "content": f"Summarize this webpage titled '{website.title}':\n\n{website.text}"}
        ]
        response = ollama.chat(model=MODEL, messages=messages)
        return response['message']['content']
    except Exception as e:
        return f"❌ Error: {e}"

# ==================== GIAO DIỆN STREAMLIT ==================== #
st.sidebar.title("🚀 Chọn Tính Năng")
option = st.sidebar.radio("Chọn ứng dụng:", ["📑 Text Summary App", "🌐 Website Summary App"])

if option == "📑 Text Summary App":
    st.title("📖 Text Summary App")
    st.markdown("Nhập đoạn văn bản và nhận bản tóm tắt từ Ollama API.")
    input_text = st.text_area("✍️ Nhập đoạn văn cần tóm tắt:", "")

    if st.button("📌 Tóm tắt ngay"):
        if input_text.strip():
            with st.spinner("⏳ Đang tóm tắt..."):
                summary = summarize_text(input_text)
            st.markdown("### ✨ Bản tóm tắt:")
            st.success(summary)
        else:
            st.warning("⚠️ Vui lòng nhập đoạn văn!")

elif option == "🌐 Website Summary App":
    st.title("🌐 Website Summary App")
    st.markdown("Nhập URL để nhận bản tóm tắt nội dung trang web.")

    url = st.text_input("🔗 Nhập URL:", "")

    if st.button("🌟 Tóm tắt trang web"):
        if url.strip():
            with st.spinner("🔍 Đang tóm tắt..."):
                summary = summarize_website(url)
            st.markdown("### 📌 Tóm tắt nội dung:")
            st.success(summary)
        else:
            st.warning("⚠️ Vui lòng nhập URL hợp lệ!")
