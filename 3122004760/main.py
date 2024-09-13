import re
import sys
import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cProfile


def read_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件路径错误或文件不存在: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise IOError(f"读取文件时出错: {e}")


def preprocess_text(text):
    """
    文本预处理：去除标点符号，并进行分词
    """
    # 避免输入文本为空出错
    if not text:
        return ""

    # 去除标点符号和换行符
    text = re.sub(r'[^\w\s]', '', text)

    # 使用 jieba 进行分词
    words = jieba.lcut(text)

    # 返回以空格分隔的词语序列
    return ' '.join(words)


def vectorize_texts(text1, text2):
    """
    使用 TF-IDF 将两个文本向量化
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return tfidf_matrix


def calculate_cosine_similarity(tfidf_matrix):
    """
    计算余弦相似度
    """
    if tfidf_matrix.shape[1] == 0:  # 如果向量的维度为0，返回0相似度
        return 0.0

    # 计算两个向量之间的余弦相似度
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity


def save_similarity_to_file(output_file, similarity):
    """
    将相似度结果保存到指定文件，保留两位小数
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{similarity:.2f}")
    except Exception as e:
        raise IOError(f"写入文件时出错: {e}")


def plagiarism_check(orig_file, plagiarism_file, output_file):
    """
    主函数：计算两个文本的相似度并保存结果
    """
    try:
        # 读取文件
        orig_text = read_file(orig_file)
        plagiarism_text = read_file(plagiarism_file)

        # 文本预处理
        processed_orig_text = preprocess_text(orig_text)
        processed_plagiarism_text = preprocess_text(plagiarism_text)

        # 向量化处理
        tfidf_matrix = vectorize_texts(processed_orig_text, processed_plagiarism_text)

        # 计算相似度
        similarity = calculate_cosine_similarity(tfidf_matrix)

        # 保存相似度结果
        save_similarity_to_file(output_file, similarity)

        print(f"相似度已计算并保存至文件: {output_file}")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except IOError as io_error:
        print(io_error)
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    # 从命令行获取文件路径
    if len(sys.argv) != 4:
        print("使用方法: python plagiarism_check.py <原文文件路径> <抄袭版文件路径> <输出文件路径>")
        sys.exit(1)
    else:
        orig_file = sys.argv[1]
        plagiarism_file = sys.argv[2]
        output_file = sys.argv[3]

        # 执行查重
        plagiarism_check(orig_file, plagiarism_file, output_file)
    cProfile.run("plagiarism_check(orig_file, plagiarism_file, output_file)", filename="performance_analysis_result")
