

# thay vì số cụm cố định là 3, ta sẽ tìm số cụm tối ưu bằng Elbow Method và Silhouette Score từ đó chọn ra số câu nhưng vẫn bị nhược điểm là nếu có 1 cụm thì không thể tóm tắt được và tóm tắt
# nó chưa được tối ưu


import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Tải punkt tokenizer nếu chưa có
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Bước 1: Tách câu bằng nltk
def tokenize_sentences(text):
    return nltk.sent_tokenize(text)

# Bước 2: Mã hóa câu thành vector bằng TF-IDF
def encode_sentences(sentences):
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    return sentence_vectors

# Bước 3: Sử dụng Elbow Method để tìm số cụm tốt nhất
from sklearn.exceptions import NotFittedError

def find_optimal_clusters(sentence_vectors, max_k=8):
    distortions = []
    silhouette_scores = []
    K_valid = []

    for k in range(2, max_k+1):
        if k >= sentence_vectors.shape[0]:
            break  # Không thể có nhiều cụm hơn số câu

        try:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(sentence_vectors)
            labels = kmeans.labels_
            
            # Kiểm tra số cụm thực sự (tránh TH tất cả dính vào 1 cụm)
            if len(set(labels)) < 2:
                continue

            distortion = kmeans.inertia_
            score = silhouette_score(sentence_vectors, labels)

            distortions.append(distortion)
            silhouette_scores.append(score)
            K_valid.append(k)

        except ValueError:
            continue

    return K_valid, distortions, silhouette_scores


# Bước 4: Phân cụm với số cụm tối ưu
def cluster_sentences(sentence_vectors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(sentence_vectors)
    return kmeans

# Bước 5: Xây dựng bản tóm tắt
def summarize_text(sentences, kmeans):
    clusters = kmeans.labels_
    summary = []
    for cluster in set(clusters):
        indices = [i for i, label in enumerate(clusters) if label == cluster]
        representative_idx = indices[0]
        summary.append(sentences[representative_idx])
    return ' '.join(summary)

# Giao diện bằng Streamlit
def main():
    st.title("Tóm tắt văn bản với TF-IDF và phân cụm (Elbow Method)")
    st.write("Nhập đoạn văn bản để tạo bản tóm tắt ngắn gọn.")

    text_input = st.text_area("Nhập đoạn văn bản:")
    
    if st.button("Tóm tắt"):
        if text_input.strip():
            sentences = tokenize_sentences(text_input)
            if len(sentences) < 3:
                st.warning("Cần ít nhất 3 câu để thực hiện phân cụm.")
                return

            sentence_vectors = encode_sentences(sentences)

            # Tìm số cụm tối ưu bằng Elbow Method
            max_k = min(8, len(sentences))  # Không vượt quá số câu
            K, distortions, silhouette_scores = find_optimal_clusters(sentence_vectors, max_k=max_k)

            # Hiển thị biểu đồ Elbow và Silhouette
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

            # Chọn số cụm tối ưu (score cao nhất)
            optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
            st.info(f"Số cụm tối ưu được chọn: {optimal_k}")

            # Thực hiện phân cụm và tóm tắt
            kmeans = cluster_sentences(sentence_vectors, num_clusters=optimal_k)
            summary = summarize_text(sentences, kmeans)

            st.write("**Bản tóm tắt:**")
            st.write(summary)
        else:
            st.warning("Vui lòng nhập đoạn văn bản để tóm tắt.")

if __name__ == "__main__":
    main()
