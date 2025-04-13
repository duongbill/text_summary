import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
import nltk
import matplotlib.pyplot as plt

# Tải punkt tokenizer nếu chưa có
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Tách câu
def tokenize_sentences(text):
    return nltk.sent_tokenize(text)

# Mã hóa bằng TF-IDF
def encode_sentences(sentences):
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    return sentence_vectors

# Tìm số cụm tối ưu bằng Elbow + Silhouette
def find_optimal_clusters(sentence_vectors, max_k=8):
    distortions = []
    silhouette_scores = []
    K_valid = []

    for k in range(2, max_k + 1):
        if k >= sentence_vectors.shape[0]:
            break

        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(sentence_vectors)
        labels = kmeans.labels_

        if len(set(labels)) < 2:
            continue

        distortion = kmeans.inertia_
        score = silhouette_score(sentence_vectors, labels)

        distortions.append(distortion)
        silhouette_scores.append(score)
        K_valid.append(k)

    return K_valid, distortions, silhouette_scores

# Phân cụm
def cluster_sentences(sentence_vectors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(sentence_vectors)
    return kmeans

# Tóm tắt bằng cách chọn câu trung tâm mỗi cụm
def summarize_text(sentences, kmeans, sentence_vectors):
    clusters = kmeans.labels_
    summary = []

    for cluster in set(clusters):
        indices = [i for i, label in enumerate(clusters) if label == cluster]
        center = kmeans.cluster_centers_[cluster].reshape(1, -1)
        cluster_vectors = sentence_vectors[indices]
        distances = cosine_distances(cluster_vectors, center)
        representative_idx = indices[distances.argmin()]
        summary.append(sentences[representative_idx])

    return ' '.join(summary)

from transformers import pipeline

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def generate_short_summary(text):
    summarizer = load_summarizer()
    summary = summarizer(text, max_length=45, min_length=10, do_sample=False)
    return summary[0]['summary_text']


# Giao diện Streamlit
def main():
    st.title("🧠 Tóm tắt văn bản bằng TF-IDF & KMeans")
    st.write("Ứng dụng sử dụng TF-IDF và phân cụm KMeans để tạo bản tóm tắt văn bản.")

    text_input = st.text_area("📄 Nhập đoạn văn bản:")

    if text_input.strip():
        sentences = tokenize_sentences(text_input)

        if len(sentences) < 3:
            st.warning("⚠️ Cần ít nhất 3 câu để thực hiện phân cụm.")
            return

        sentence_vectors = encode_sentences(sentences)

        max_k = min(6, len(sentences))  # Giới hạn số cụm để giữ tóm tắt ngắn
        K, distortions, silhouette_scores = find_optimal_clusters(sentence_vectors, max_k=max_k)

        if not K:
            st.warning("Không đủ câu hoặc cụm hợp lệ để phân cụm.")
            return

        # Vẽ biểu đồ Elbow & Silhouette
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

        # Chọn số cụm thủ công
        selected_k = st.slider("🔧 Chọn số câu tóm tắt mong muốn:", min_value=2, max_value=max_k, value=K[silhouette_scores.index(max(silhouette_scores))])
        st.info(f"Số cụm được sử dụng để tóm tắt: {selected_k}")

        kmeans = cluster_sentences(sentence_vectors, num_clusters=selected_k)
        summary = summarize_text(sentences, kmeans, sentence_vectors)

        st.subheader("📌 Bản tóm tắt:")
        st.write(summary)

if __name__ == "__main__":
    main()
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

def plot_clusters(sentence_vectors, labels, method="pca"):
    if method == "pca":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42)
        
    reduced = reducer.fit_transform(sentence_vectors.toarray())
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=labels, palette='tab10')
    plt.title(f"Biểu đồ cụm bằng {method.upper()}")
    st.pyplot(plt.gcf())
if st.checkbox("Hiển thị cụm theo nội dung"):
    method = st.selectbox("Chọn phương pháp giảm chiều:", ["pca", "tsne"])
    plot_clusters(sentence_vectors, kmeans.labels_, method=method)
