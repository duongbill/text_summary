import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# -------------------- PHẦN 1: LOAD MÔ HÌNH T5 -------------------- #
# Đảm bảo thư mục fine_tuned_t5 đã chứa mô hình đã fine-tune của bạn
model_dir = "./fine_tuned_t5"
model = T5ForConditionalGeneration.from_pretrained(model_dir)
tokenizer = T5Tokenizer.from_pretrained(model_dir)

def summarize_text_t5(text):
    """
    Sử dụng mô hình T5 để tóm tắt văn bản.
    """
    # Tạo input với prefix "summarize: "
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary

# -------------------- PHẦN 2: HÀM TÓM TẮT WEBSITE -------------------- #
class Website:
    """
    Lấy nội dung trang web và trích xuất văn bản bằng Selenium & BeautifulSoup.
    """
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
        # Xóa các thẻ không cần thiết
        for tag in soup.body(["script", "style", "img", "input"]):
            tag.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

def summarize_website_t5(url):
    """
    Sử dụng mô hình T5 để tóm tắt nội dung của trang web.
    """
    try:
        website = Website(url)
        # Sử dụng nội dung văn bản của website để tóm tắt
        summary = summarize_text_t5(website.text)
        return summary
    except Exception as e:
        return f"❌ Error: {e}"

# -------------------- PHẦN 3: GIAO DIỆN STREAMLIT -------------------- #
st.sidebar.title("🚀 Chọn Tính Năng")
option = st.sidebar.radio("Chọn ứng dụng:", ["📑 Text Summary App", "🌐 Website Summary App"])

if option == "📑 Text Summary App":
    st.title("📖 Text Summary App")
    st.markdown("Nhập vào đoạn văn bản để nhận bản tóm tắt từ mô hình T5 đã fine-tune.")
    input_text = st.text_area("✍️ Nhập đoạn văn cần tóm tắt:", "")

    if st.button("📌 Tóm tắt ngay"):
        if input_text.strip():
            with st.spinner("⏳ Đang tóm tắt..."):
                summary = summarize_text_t5(input_text)
            st.markdown("### ✨ Bản tóm tắt:")
            st.success(summary)
        else:
            st.warning("⚠️ Vui lòng nhập đoạn văn!")

elif option == "🌐 Website Summary App":
    st.title("🌐 Website Summary App")
    st.markdown("Nhập URL để nhận bản tóm tắt nội dung trang web từ mô hình T5 đã fine-tune.")
    url = st.text_input("🔗 Nhập URL:", "")

    if st.button("🌟 Tóm tắt trang web"):
        if url.strip():
            with st.spinner("🔍 Đang tóm tắt..."):
                summary = summarize_website_t5(url)
            st.markdown("### 📌 Tóm tắt nội dung:")
            st.success(summary)
        else:
            st.warning("⚠️ Vui lòng nhập URL hợp lệ!")
