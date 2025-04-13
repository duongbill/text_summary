import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from transformers import pipeline
import heapq

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

# Bước 4: Tóm tắt văn bản sử dụng mô hình học sâu (T5/BART)
def deep_learning_summary(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Bước 5: Áp dụng A* để chọn câu tốt nhất (từ các câu trong cụm)
def a_star_select_sentences(sentences, sentence_vectors, kmeans):
    # Giả sử mục tiêu của chúng ta là chọn câu có độ quan trọng cao nhất trong mỗi cụm
    clusters = kmeans.labels_
    selected_sentences = []

    for cluster in set(clusters):
        # Lọc các câu trong mỗi cụm
        cluster_sentences = [sentences[i] for i in range(len(sentences)) if clusters[i] == cluster]
        cluster_vectors = sentence_vectors[clusters == cluster]
        
        # Tính toán độ tương đồng của các câu trong cụm với tóm tắt (heuristic)
        heuristic_scores = [sum(cluster_vectors[i].toarray()) for i in range(len(cluster_sentences))]
        
        # Dùng thuật toán A* để chọn câu có độ quan trọng cao nhất
        top_sentences_idx = heapq.nlargest(3, range(len(heuristic_scores)), heuristic_scores.__getitem__)
        
        for idx in top_sentences_idx:
            selected_sentences.append(cluster_sentences[idx])
    
    return ' '.join(selected_sentences)

# Giao diện bằng Streamlit
def main():
    st.title("Tóm tắt văn bản với TF-IDF, Học sâu và A*")
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

            # Tạo bản tóm tắt bằng mô hình học sâu (BART)
            deep_summary = deep_learning_summary(text_input)

            # Tạo bản tóm tắt sử dụng A* để chọn câu
            summary = a_star_select_sentences(sentences, sentence_vectors, kmeans)

            st.write("**Bản tóm tắt (Học sâu):**")
            st.write(deep_summary)

            st.write("**Bản tóm tắt (A* và Phân cụm):**")
            st.write(summary)
        else:
            st.warning("Vui lòng nhập đoạn văn bản để tóm tắt.")

if __name__ == "__main__":
    main()
