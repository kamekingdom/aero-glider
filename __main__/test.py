import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def normalize_array(arr, length):
    arr = np.array(arr)
    arr = arr.T  # 配列を転置して処理します
    normalized = np.interp(np.linspace(0, arr.shape[1] - 1, length), np.arange(arr.shape[1]), arr)
    return normalized.reshape(1, -1)

def calculate_similarity(arr1, arr2):
    # normalize arrays
    min_length = min(len(arr1), len(arr2))
    normalized_arr1 = normalize_array(np.array(arr1), min_length)  # リストをNumPy配列に変換
    normalized_arr2 = normalize_array(np.array(arr2), min_length)  # リストをNumPy配列に変換

    # calculate cosine similarity
    similarity = cosine_similarity(normalized_arr1, normalized_arr2)

    return similarity[0][0]

# Test the function
arr1 = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
arr2 = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 4]]
similarity = calculate_similarity(arr1, arr2)
print("Similarity: ", similarity)
