import nltk


from nltk.corpus import words

english_words = words.words()

nltk.download('words')  # 辞書のダウンロード
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')