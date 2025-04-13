import streamlit as st
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Tải punkt nếu chưa có
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Tóm tắt phân cụm ---
def tokenize_sentences(text):
    return nltk.sent_tokenize(text)

def encode_sentences(sentences):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(sentences)

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
        summary.append(sentences[indices[0]])  # Câu đại diện
    return ' '.join(summary)

# --- Tóm tắt học sâu ---
@st.cache_resource
def load_finetuned_model(model_path="duong-ai/bart-finetuned-summary"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return pipeline("summarization", model=model, tokenizer=tokenizer)

# @st.cache_resource
# def load_finetuned_model(model_path="sshleifer/distilbart-cnn-12-6"):  # hoặc bất kỳ model nào phù hợp
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
#     return pipeline("summarization", model=model, tokenizer=tokenizer)


def summarize_with_bart(text, model):
    result = model(text, max_length=45, min_length=10, do_sample=False)
    return result[0]['summary_text']

# --- Vẽ cụm ---
def plot_clusters(vectors, labels, method="pca"):
    reducer = PCA(n_components=2) if method == "pca" else TSNE(n_components=2, random_state=42)
    reduced = reducer.fit_transform(vectors.toarray())
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=labels, palette="tab10")
    plt.title(f"Biểu đồ cụm theo {method.upper()}")
    st.pyplot(plt.gcf())

# --- Streamlit App ---
def main():
    st.title("📄 Tóm tắt văn bản bằng AI (TF-IDF + Học sâu)")
    st.write("Tóm tắt văn bản bằng KMeans và mô hình học sâu fine-tuned (BART/T5)")

    text_input = st.text_area("✍️ Nhập đoạn văn bản cần tóm tắt:")

    if st.button("🚀 Tóm tắt"):
        if not text_input.strip():
            st.warning("Vui lòng nhập đoạn văn bản.")
            return

        sentences = tokenize_sentences(text_input)
        if len(sentences) < 3:
            st.warning("Văn bản cần ít nhất 3 câu để thực hiện phân cụm.")
            return

        vectors = encode_sentences(sentences)
        max_k = min(8, len(sentences))
        K, distortions, silhouette_scores = find_optimal_clusters(vectors, max_k)

        if not K:
            st.error("Không tìm được cụm hợp lệ.")
            return

        # Biểu đồ Elbow + Silhouette
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(K, distortions, marker='o')
        ax[0].set_title("Elbow Method")
        ax[0].set_xlabel("Số cụm")
        ax[0].set_ylabel("Distortion")

        ax[1].plot(K, silhouette_scores, marker='x', color='green')
        ax[1].set_title("Silhouette Score")
        ax[1].set_xlabel("Số cụm")
        ax[1].set_ylabel("Score")

        st.pyplot(fig)

        optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
        st.info(f"✅ Số cụm tối ưu: {optimal_k}")

        kmeans = cluster_sentences(vectors, num_clusters=optimal_k)
        summary = summarize_by_clustering(sentences, kmeans)
        st.subheader("📚 Bản tóm tắt theo phân cụm:")
        st.write(summary)

        # PCA/TSNE
        if st.checkbox("📊 Hiển thị cụm nội dung"):
            method = st.selectbox("Phương pháp giảm chiều:", ["pca", "tsne"])
            plot_clusters(vectors, kmeans.labels_, method=method)

        # Tóm tắt siêu ngắn
        if st.checkbox("✨ Tạo bản tóm tắt siêu ngắn (học sâu)"):
            with st.spinner("Đang chạy mô hình học sâu..."):
                model = load_finetuned_model()
                short_summary = summarize_with_bart(text_input, model)
                st.success("📝 Tóm tắt siêu ngắn:")
                st.write(short_summary)

if __name__ == "__main__":
    main()
