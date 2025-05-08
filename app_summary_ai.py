import streamlit as st  # type: ignore
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time

# Thiết lập trang
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stTextArea textarea {
        font-size: 14px;
    }
    .css-1d391kg {
        padding: 1rem 0.5rem;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .stMarkdown {
        font-size: 14px;
    }
    .stButton button {
        padding: 0.5rem 2rem;
        width: auto;
        min-width: 120px;
        margin: 1rem auto;
        display: block;
        border-radius: 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    div[data-testid="stSidebar"] {
        padding: 1rem 0.5rem;
    }
    div[data-testid="stButton"] {
        text-align: center;
        width: 100%;
        display: flex;
        justify-content: center;
    }
    div[data-testid="stButton"] > button {
        margin: 1rem auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Tải punkt
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Danh sách các từ nối cần loại bỏ
stopwords = [
    "trước khi", "sau khi", "trong khi", "khi mà", "lúc đó", "bây giờ", "hiện tại", "lúc này", "ngay lập tức",
    "bởi vì", "vì", "do", "bởi", "kết quả là", "dẫn đến", "vì lẽ đó", "chính vì thế", "từ đó",
    "nhưng", "nhưng mà", "tuy", "mặc dù vậy", "trái lại", "ngược lại", "tuy vậy", "dù vậy", "tuy thế",
    "ngoài ra", "thêm vào đó", "hơn nữa", "không những thế", "bên cạnh đó", "cũng như", "đồng thời",
    "nếu", "nếu như", "giả sử", "với điều kiện", "trong trường hợp", "miễn là", "cho dù", "dù cho",
    "để", "nhằm", "với mục đích", "nhằm mục đích", "hầu", "để mà", "nhằm mục tiêu",
    "giống như", "tương tự", "cũng giống", "khác với", "so với", "hơn", "kém", "như là",
    "hay là", "sau đó", "vì vậy", "cho nên", "tuy nhiên", "mặc dù", "do đó", "vì thế", "thế nhưng",
    "bên cạnh đó", "ngoài ra", "đồng thời", "trong khi đó", "hơn nữa", "có thể nói",
    "mặt khác", "từ đó", "trên thực tế", "kết luận lại", "cụ thể là", "nói chung",
    "nhìn chung", "nói cách khác", "tóm lại", "từ trước đến nay", "về cơ bản", "thực ra",
    "điều đó cho thấy", "như vậy", "kết quả là", "suy ra", "nói tóm lại"
]

# --- Tiền xử lý ---
def tokenize_sentences(text):
    return nltk.sent_tokenize(text)

def remove_stopwords(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        cleaned_sentence = sentence
        for stopword in stopwords:
            cleaned_sentence = cleaned_sentence.replace(stopword, "")
        cleaned_sentences.append(cleaned_sentence)
    return cleaned_sentences

def encode_sentences(sentences):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(sentences)

# --- Phân cụm ---
def find_optimal_clusters(sentence_vectors, max_k=8):
    distortions, silhouette_scores, K_valid = [], [], []
    for k in range(2, max_k + 1):
        if k >= sentence_vectors.shape[0]:
            break
        kmeans = KMeans(n_clusters=k, random_state=0).fit(sentence_vectors)
        labels = kmeans.labels_
        if len(set(labels)) < 2: continue
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(sentence_vectors, labels))
        K_valid.append(k)
    return K_valid, distortions, silhouette_scores

def cluster_sentences(sentence_vectors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(sentence_vectors)
    return kmeans

def summarize_by_clustering(sentences, kmeans):
    labels = kmeans.labels_
    summary = []
    for cluster in set(labels):
        indices = [i for i, label in enumerate(labels) if label == cluster]
        summary.append(sentences[indices[0]])  # Câu đại diện mỗi cụm
    return ' '.join(summary)

# --- Heuristic Search ---
def compute_tfidf(sentences):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(sentences)

def compute_cosine_similarity(tfidf_matrix):
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def heuristic_search(sentences, cosine_sim, num_summary_sentences=3):
    sentence_scores = cosine_sim.sum(axis=1)
    ranked_sentences = sentence_scores.argsort()[::-1]
    summary = [sentences[i] for i in ranked_sentences[:num_summary_sentences]]
    return ' '.join(summary)

# --- Mô hình học sâu ---
@st.cache_resource
def load_finetuned_model(model_path="duonggbill/dbill-model-summary"):  # <-- dùng Hugging Face model
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return model, tokenizer

def summarize_with_t5(text, model, tokenizer):
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, min_length=80, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- Tóm tắt kết hợp ---
def hybrid_summarization(text, model, tokenizer, num_summary_sentences=3):
    sentences = tokenize_sentences(text)
    cleaned_sentences = remove_stopwords(sentences)

    if len(cleaned_sentences) < num_summary_sentences:
        return "Văn bản quá ngắn để tóm tắt."

    sentence_vectors = encode_sentences(cleaned_sentences)
    _, _, silhouette_scores = find_optimal_clusters(sentence_vectors)
    num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2 if silhouette_scores else 1
    kmeans = cluster_sentences(sentence_vectors, num_clusters)
    clustered_summary = summarize_by_clustering(sentences, kmeans)

    # Tóm tắt heuristic từ kết quả phân cụm
    heuristic_sentences = tokenize_sentences(clustered_summary)
    cleaned = remove_stopwords(heuristic_sentences)
    tfidf_matrix = compute_tfidf(cleaned)
    cosine_sim = compute_cosine_similarity(tfidf_matrix)
    selected = heuristic_search(heuristic_sentences, cosine_sim, num_summary_sentences)

    # Đưa vào mô hình học sâu để sinh bản tóm tắt cuối cùng
    return summarize_with_t5(selected, model, tokenizer)

# --- Giao diện Streamlit ---
def main():
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Cài đặt")
        st.markdown("---")
        
        # Cấu hình tóm tắt
        st.subheader("Cấu hình tóm tắt")
        num_sentences = st.slider(
            "Số câu trong bản tóm tắt",
            min_value=1,
            max_value=10,
            value=3,
            help="Số lượng câu bạn muốn trong bản tóm tắt"
        )
        
        # Hiển thị thông tin
        st.markdown("---")
        st.subheader("ℹ️ Thông tin")
        st.markdown("""
        Ứng dụng sử dụng kết hợp 3 phương pháp:
        - 🤖 Mô hình học sâu (T5)
        - 📊 Phân cụm (Clustering)
        - 🔍 Tìm kiếm heuristic
        """)

    # Main content
    st.title("📄 Tóm tắt văn bản bằng AI")
    st.markdown("Ứng dụng này giúp bạn tóm tắt văn bản một cách thông minh.")

    # Input area
    col1, col2 = st.columns([3, 2])
    with col1:
        text_input = st.text_area(
            "✍️ Nhập đoạn văn bản cần tóm tắt:",
            height=200,
            help="Nhập hoặc dán văn bản cần tóm tắt vào đây"
        )
        
        # Process button - centered within the input column
        if st.button("🚀 Tóm tắt"):
            if not text_input.strip():
                st.warning("⚠️ Vui lòng nhập đoạn văn bản.")
                return

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load model
            status_text.text("🔄 Đang tải mô hình...")
            model, tokenizer = load_finetuned_model()
            progress_bar.progress(20)
            
            # Process text
            status_text.text("⚙️ Đang xử lý văn bản...")
            time.sleep(0.5)
            progress_bar.progress(50)
            
            # Generate summary
            status_text.text("📝 Đang tạo bản tóm tắt...")
            summary = hybrid_summarization(text_input, model, tokenizer, num_sentences)
            progress_bar.progress(100)
            
            # Display results
            status_text.text("✅ Hoàn thành!")
            time.sleep(0.5)
            
            # Results section
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📝 Bản tóm tắt:")
                st.markdown(f"<div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px; font-size: 14px;'>{summary}</div>", unsafe_allow_html=True)
            
            with col2:
                st.subheader("📊 Thống kê:")
                original_length = len(text_input.split())
                summary_length = len(summary.split())
                compression_ratio = (1 - summary_length/original_length) * 100
                
                st.metric("Độ dài văn bản gốc", f"{original_length} từ")
                st.metric("Độ dài bản tóm tắt", f"{summary_length} từ")
                st.metric("Tỷ lệ nén", f"{compression_ratio:.1f}%")
    
    with col2:
        st.markdown("### 📋 Hướng dẫn")
        st.markdown("""
        1. Nhập hoặc dán văn bản
        2. Điều chỉnh số câu tóm tắt
        3. Nhấn nút 'Tóm tắt'
        4. Xem kết quả
        """)

if __name__ == "__main__":
    main()
