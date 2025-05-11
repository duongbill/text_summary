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
def hybrid_summarization(text, model, tokenizer, num_summary_sentences=None):
    sentences = tokenize_sentences(text)
    cleaned_sentences = remove_stopwords(sentences)

    if len(cleaned_sentences) < 2:
        return "Văn bản quá ngắn để tóm tắt."

    # --- Xác định số câu tóm tắt dựa trên độ dài ---
    if num_summary_sentences is None:
        total_sentences = len(cleaned_sentences)
        if total_sentences <= 5:
            num_summary_sentences = 1
        elif total_sentences <= 10:
            num_summary_sentences = 2
        elif total_sentences <= 20:
            num_summary_sentences = 3
        else:
            num_summary_sentences = min(5, total_sentences // 4)

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
    st.title("📄 Tóm tắt văn bản bằng AI")
    st.write("Tích hợp Heuristic + Phân cụm + Mô hình học sâu để tóm tắt tối ưu")

    text_input = st.text_area("✍️ Nhập đoạn văn bản cần tóm tắt:", height=200)

    if st.button("🚀 Tóm tắt"):
        if not text_input.strip():
            st.warning("Vui lòng nhập đoạn văn bản.")
            return

        with st.spinner("⏳ Đang tóm tắt văn bản..."):
            # Hiển thị thanh tiến trình
            progress_bar = st.progress(0)
            
            # Load model
            progress_bar.progress(25)
            model, tokenizer = load_finetuned_model()
            
            # Xử lý văn bản
            progress_bar.progress(50)
            sentences = tokenize_sentences(text_input)
            cleaned_sentences = remove_stopwords(sentences)
            
            # Phân cụm và tìm câu đại diện
            progress_bar.progress(75)
            
            # Tóm tắt cuối cùng
            summary = hybrid_summarization(text_input, model, tokenizer)
            progress_bar.progress(100)

        st.subheader("📝 Bản tóm tắt:")
        st.write(summary)

if __name__ == "__main__":
    main()
