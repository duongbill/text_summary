import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import ollama

# Constants
MODEL = "llama3.2"

# A class to represent a Webpage
class Website:
    """
    A utility class to represent a Website that we have scraped
    """
    url: str
    title: str
    text: str

    def __init__(self, url):
        """
        Create this Website object from the given URL using Selenium and BeautifulSoup
        """
        self.url = url

        # Khởi tạo trình duyệt Chrome chạy nền
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

        # Parse nội dung trang web
        soup = BeautifulSoup(html, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

# System prompt for the model
system_prompt = (
    "You are an assistant that analyzes the contents of a website "
    "and provides a short summary, ignoring text that might be navigation related. "
    "Respond in markdown."
)

# Function to create the user prompt
def user_prompt_for(website):
    return f"You are looking at a website titled '{website.title}'.\n\n" \
           f"Here is the content:\n\n{website.text}\n\n" \
           "Please provide a concise summary of this webpage in markdown format."

# Function to create messages for the model
def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)},
    ]

# Function to summarize a URL
def summarize(url):
    try:
        website = Website(url)
        messages = messages_for(website)
        response = ollama.chat(model=MODEL, messages=messages)
        return response['message']['content']
    except Exception as e:
        return f"❌ Error: {e}"

# Streamlit app
st.title("Website Summary App 📝")
st.markdown("Enter a URL to summarize the content of the website using the Ollama API.")

# Input URL
url = st.text_input("Enter a URL:", "")

if st.button("Summarize"):
    if url:
        with st.spinner("Summarizing... ⏳"):
            summary = summarize(url)
        st.markdown(summary)
    else:
        st.warning("⚠️ Please enter a valid URL.")
