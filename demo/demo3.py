import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np

# Tải punkt tokenizer nếu chưa có
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Tách câu bằng nltk
def tokenize_sentences(text):
    return nltk.sent_tokenize(text)

# Mã hóa câu thành vector bằng TF-IDF
def encode_sentences(sentences):
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    return sentence_vectors

# Elbow method tìm số cụm tối ưu
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

        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(sentence_vectors, labels))
        K_valid.append(k)

    return K_valid, distortions, silhouette_scores

# Phân cụm
def cluster_sentences(sentence_vectors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(sentence_vectors)
    return kmeans

# Tóm tắt văn bản
def summarize_text(sentences, kmeans, sentence_vectors, top_n=2):
    clusters = kmeans.labels_
    summary = []

    # Sắp xếp cụm theo kích thước (ưu tiên cụm lớn)
    cluster_sizes = sorted(set(clusters), key=lambda c: list(clusters).count(c), reverse=True)

    for cluster in cluster_sizes[:top_n]:
        indices = [i for i, label in enumerate(clusters) if label == cluster]

        # Tìm câu có tổng điểm TF-IDF cao nhất
        sub_vectors = sentence_vectors[indices]
        scores = sub_vectors.sum(axis=1)
        best_idx = indices[int(np.argmax(scores))]

        summary.append(sentences[best_idx])
    return ' '.join(summary)

# Giao diện Streamlit
def main():
    st.title("Tóm tắt văn bản ngắn gọn với TF-IDF & KMeans")
    st.write("Nhập văn bản và chọn số câu tóm tắt mong muốn.")

    text_input = st.text_area("Nhập đoạn văn bản:")
    top_n = st.slider("Số câu trong bản tóm tắt", min_value=1, max_value=5, value=2)

    if st.button("Tóm tắt"):
        if text_input.strip():
            sentences = tokenize_sentences(text_input)
            if len(sentences) < 3:
                st.warning("Cần ít nhất 3 câu để thực hiện phân cụm.")
                return

            sentence_vectors = encode_sentences(sentences)
            max_k = min(8, len(sentences))
            K, distortions, silhouette_scores = find_optimal_clusters(sentence_vectors, max_k=max_k)

            if not K:
                st.warning("Không tìm được số cụm hợp lệ.")
                return

            # Vẽ biểu đồ
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(K, distortions, marker='o')
            ax[0].set_title("Elbow Method (Distortion)")
            ax[0].set_xlabel("Số cụm")
            ax[0].set_ylabel("Distortion")

            ax[1].plot(K, silhouette_scores, marker='x', color='green')
            ax[1].set_title("Silhouette Scores")
            ax[1].set_xlabel("Số cụm")
            ax[1].set_ylabel("Score")

            st.pyplot(fig)

            # Số cụm tối ưu
            optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
            st.info(f"Số cụm tối ưu: {optimal_k}")

            # Phân cụm và tóm tắt
            kmeans = cluster_sentences(sentence_vectors, num_clusters=optimal_k)
            summary = summarize_text(sentences, kmeans, sentence_vectors, top_n=top_n)

            st.write("**Bản tóm tắt ngắn gọn:**")
            st.success(summary)
        else:
            st.warning("Vui lòng nhập đoạn văn bản để tóm tắt.")

if __name__ == "__main__":
    main()
