import pytest
import os
import main


class TestFunction:
    def test_read_file_existing(self):
        """
        1.1: 测试能否正确读取已存在的文件
        预期：text非空
        """
        test_file = './test_text/orig.txt'
        text = main.read_file(test_file)
        assert text is not None

    def test_read_file_not_found(self):
        """
        测试点 1.2: 测试读取不存在的文件
        预期：报错，并提示错误文件名
        """
        # 404.txt 不存在
        test_file = './test_text/404.txt'
        with pytest.raises(FileNotFoundError):
            main.read_file(test_file)

    def test_save_similarity_to_file(self):
        """
        测试点 1.3: 测试能否正确保存相似度结果(即输出内容写入文件）
        预期：读取文件内容与写入时一致
        """
        test_output_file = './test_text/test_output.txt'
        main.save_similarity_to_file(test_output_file, 0.75)

        with open(test_output_file, 'r', encoding='utf-8') as f:
            result = f.read()
        assert result == '0.75'
        os.remove(test_output_file)

    def test_preprocess_text_normal(self):
        """
        测试点 2.1: 测试正常文本预处理
        预期：返回分词后字符串
        """
        text = "你好，世界！这是一个测试。"
        result = main.preprocess_text(text)
        assert result == "你好 世界 这是 一个 测试"

    def test_preprocess_empty_text(self):
        """
        测试点 2.2: 测试空文本处理
        预期：返回空字符串
        """
        text = ""
        result = main.preprocess_text(text)
        assert result == ""

    def test_vectorize_texts_normal(self):
        """
        测试点 3.1: 测试正常文本向量化
        预期：与计算结果一致
        """
        text1 = "这是一个测试"
        text2 = "这是另一个测试"
        tfidf_matrix = main.vectorize_texts(text1, text2)
        assert tfidf_matrix.shape == (2, 2)  # 2 行（文本），2列（词汇数）

    def test_calculate_similarity_normal(self):
        """
        测试点 4.1: 测试不同文本的相似度计算
        预期：返回一个不大于1的浮点数
        """
        text1 = "这是一个测试"
        text2 = "这是另一个测试"
        text1 = main.preprocess_text(text1)
        text2 = main.preprocess_text(text2)
        tfidf_matrix = main.vectorize_texts(text1, text2)
        similarity = main.calculate_cosine_similarity(tfidf_matrix)
        assert 0 <= similarity <= 1

    def test_calculate_similarity_identical(self):
        """
        测试点 4.2: 测试相同文本的相似度计算
        预期：返回1.0
        """
        text1 = "这是一个测试"
        text2 = "这是一个测试"
        text1 = main.preprocess_text(text1)
        text2 = main.preprocess_text(text2)
        tfidf_matrix = main.vectorize_texts(text1, text2)
        similarity = main.calculate_cosine_similarity(tfidf_matrix)
        similarity = round(similarity, 2)
        assert similarity == 1.0

    def test_calculate_similarity_completely_different(self):
        """
        测试点 4.3: 测试完全不同文本（或空文本）的相似度计算
        预期：返回的余弦相似度极小，小于0.1
        """
        text1 = "这是一个测试"
        text2 = "完全不同的文本"
        text3 = ""
        tfidf_matrix1 = main.vectorize_texts(text1, text2)
        tfidf_matrix2 = main.vectorize_texts(text1, text3)
        similarity1 = main.calculate_cosine_similarity(tfidf_matrix1)
        similarity2 = main.calculate_cosine_similarity(tfidf_matrix2)
        assert similarity1 < 0.1
        assert similarity2 == 0.0

    def test_main_flow(self):
        """
        测试点 5.1: 测试主函数
        预期：各个断言通过
        """
        orig_file = './test_text/orig.txt'
        plagiarism_file = './test_text/orig_0.8_add.txt'
        output_file = './test_text/orig_output.txt'

        orig_file = main.read_file(orig_file)
        plagiarism_file = main.read_file(plagiarism_file)
        assert orig_file is not None
        assert plagiarism_file is not None

        tfidf_matrix = main.vectorize_texts(orig_file, plagiarism_file)

        similarity = main.calculate_cosine_similarity(tfidf_matrix)
        assert 0 <= float(similarity) <= 1

        main.save_similarity_to_file(output_file, similarity)
        with open(output_file, 'r', encoding='utf-8') as f:
            result = f.read()
        assert result == str(round(similarity, 2))

    def test_missing_arguments(self):
        """
        测试点 5.2:模拟缺少命令行参数的情况
        预期：退出码非正常
        """
        orig_file = './test_text/orig.txt'
        plagiarism_file = './test_text/orig_0.8_add.txt'
        output_file = './test_text/orig_output.txt'

        # 使用 os.system 运行命令，少传递一个参数来模拟缺少参数的情况
        exit_code = os.system(f'python main.py {orig_file} {plagiarism_file}')
        # 预期程序返回非零状态码，因为命令行参数不足
        assert exit_code != 0  # os.system() 返回的非零代码表示错误

    def test_extra_arguments(self):
        """
        测试点 5.3:模拟多于3个命令行参数的情况
        预期：退出码非正常
        """
        orig_file = './test_text/orig.txt'
        plagiarism_file = './test_text/orig_0.8_add.txt'
        extra_file = "./test_text/orig_0.8/del.txt"
        output_file = './test_text/orig_output.txt'

        # 使用 os.system 运行命令，多传递一个参数来模拟过多参数的情况
        exit_code = os.system(f'python main.py {orig_file} {plagiarism_file} {extra_file} {output_file}')
        # 预期程序返回非零状态码，因为命令行参数过多
        assert exit_code != 0  # os.system() 返回的非零代码表示错误


if __name__ == '__main__':
    pytest.main(['-vs', 'testDemo.py'])
