
# ở đây t đang phát triển theo 3 cụm cố định. thường nó sẽ lấy 3 câu đầu tiên của mỗi đoạn để xử lý

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk

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

# Bước 3: Phân cụm
def cluster_sentences(sentence_vectors, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(sentence_vectors)
    return kmeans

# Bước 4: Xây dựng bản tóm tắt
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
    st.title("Tóm tắt văn bản với TF-IDF và phân cụm")
    st.write("Nhập đoạn văn bản để tạo bản tóm tắt ngắn gọn.")

    # Nhập đoạn văn bản
    text_input = st.text_area("Nhập đoạn văn bản:")
    
    if st.button("Tóm tắt"):
        if text_input.strip():
            # Tách câu từ đoạn văn bản
            sentences = tokenize_sentences(text_input)

            # Mã hóa câu và thực hiện phân cụm
            sentence_vectors = encode_sentences(sentences)
            kmeans = cluster_sentences(sentence_vectors, num_clusters=3)  # Số lượng cụm cố định là 3

            # Tạo bản tóm tắt
            summary = summarize_text(sentences, kmeans)

            st.write("**Bản tóm tắt:**")
            st.write(summary)
        else:
            st.warning("Vui lòng nhập đoạn văn bản để tóm tắt.")

if __name__ == "__main__":
    main()
