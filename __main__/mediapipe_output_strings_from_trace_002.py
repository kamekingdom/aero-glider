import cv2
import mediapipe as mp
import numpy as np
import difflib
import nltk
from nltk.corpus import wordnet
import english_words

"""
from nltk.corpus import words
english_words = words.words()
english_words = set(english_words)
"""

english_words = english_words.english_words
print(f"登録単語数:{len(english_words)}")

# キーボードの配置
keyboard_mapping = {
    (1, 2): "Q", (2, 2): "W", (3, 2): "E", (4, 2): "R", (5, 2): "T", (6, 2): "Y", (7, 2): "U", (8, 2): "I", (9, 2): "O", (10, 2): "P",
    (1, 3): "A", (2, 3): "S", (3, 3): "D", (4, 3): "F", (5, 3): "G", (6, 3): "H", (7, 3): "J", (8, 3): "K", (9, 3): "L", (10, 3): ";",
    (1, 4): "Z", (2, 4): "X", (3, 4): "C", (4, 4): "V", (5, 4): "B", (6, 4): "N", (7, 4): "M", (8, 4): ",", (9, 4): ".", (10, 4): "/",
}

# ウィンドウの作成
cv2.namedWindow("Main", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Main", 800, 600)

# MediaPipeの初期化
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# カメラキャプチャの初期化
cap = cv2.VideoCapture(0)
# フレームレートを設定する
frame_rate = 30
cap.set(cv2.CAP_PROP_FPS, frame_rate)

# 軌跡座標のリスト
trajectory = []
is_count_start_number = 5 # 何個の座標の平均を取るか

from gensim.models import Word2Vec

import nltk
from nltk.corpus import wordnet

def calculate_next_word(sentence, new_words):
    # 文をトークン化
    tokens = nltk.word_tokenize(sentence)
    # 品詞タグ付け
    tagged_tokens = nltk.pos_tag(tokens)

    # 名詞のみを抽出
    nouns = [token for token, tag in tagged_tokens if tag.startswith('NN')]

    # 新しい単語との類似度を計算
    max_similarity = 0
    next_word = new_words[0]

    for new_word in new_words:
        new_word_synsets = wordnet.synsets(new_word)
        if new_word_synsets:
            for noun in nouns:
                noun_synsets = wordnet.synsets(noun)
                if noun_synsets:
                    similarity = noun_synsets[0].wup_similarity(new_word_synsets[0])
                    if similarity is not None and similarity > max_similarity:
                        max_similarity = similarity
                        next_word = new_word

    return next_word

number_of_next_word = 5

def find_closest_words(input_string):
    closest_words = []
    similarities = []

    print(f"入力結果: {input_string}")

    for word in english_words:
        similarity = difflib.SequenceMatcher(None, input_string, word).ratio()

        if len(closest_words) < number_of_next_word:
            closest_words.append(word)
            similarities.append(similarity)
        else:
            min_similarity = min(similarities)
            min_index = similarities.index(min_similarity)
            if similarity > min_similarity:
                closest_words[min_index] = word
                similarities[min_index] = similarity
    
    sorted_words = [x for _, x in sorted(zip(similarities, closest_words), reverse=True)]

    return sorted_words

def smooth_trajectory(trajectory, window_size=5):
    smoothed_trajectory = []
    for i in range(len(trajectory)):
        start_idx = max(0, i - window_size)
        end_idx = min(len(trajectory), i + window_size + 1)
        window = trajectory[start_idx:end_idx]
        smoothed_point = np.mean(window, axis=0)
        smoothed_trajectory.append(smoothed_point)
    return smoothed_trajectory

def remove_repeated_characters(input_string):
    if len(input_string) <= 1:
        return input_string
    
    result = input_string[0] + input_string[1]
    for i in range(2, len(input_string)):
        if input_string[i] != input_string[i-1]:
            result += input_string[i]
    
    return result

sub_trajectory_length = 10

def generate_word_suggestion(trajectory, image):
    # 軌跡から出力文字列を生成する処理
    word_suggestion = ""
    trajectory = smooth_trajectory(trajectory)
    for i in range(0, len(trajectory), sub_trajectory_length):
        sub_trajectory = trajectory[i:i+sub_trajectory_length]
        if len(sub_trajectory) == sub_trajectory_length:
            avg_x = sum(point[0] for point in sub_trajectory) // sub_trajectory_length
            avg_y = sum(point[1] for point in sub_trajectory) // sub_trajectory_length
            key = (avg_x // (image.shape[1] // 10) + 1, avg_y // (image.shape[0] // 5) + 1)
            if key in keyboard_mapping:
                word_suggestion += keyboard_mapping[key]

    word_suggestion = remove_repeated_characters(word_suggestion)
    word_suggestion = word_suggestion.lower()

    word_suggestion = find_closest_words(word_suggestion)
    return word_suggestion

# 直線判定のパラメータ
threshold_angle = 15
straight_counter = 0
straight_frame_threshold = 10
word_suggestions = ""
sentence = ""

# mp_hands.Handsのインスタンスを作成
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("カメラからのキャプチャができませんでした。")
            break

        # 鏡像反転
        frame = cv2.flip(frame, 1)

        # 骨格検出を実行
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if not results:
            straight_counter = 0

        # 検出結果の描画
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # キーボードの表示
            for key, value in keyboard_mapping.items():
                x = (key[0] - 1) * (image.shape[1] // 10)
                y = (key[1] - 1) * (image.shape[0] // 5)
                # キーボードの背景を透明にする
                overlay = image.copy()
                cv2.rectangle(overlay, (x, y), (x + (image.shape[1] // 10), y + (image.shape[0] // 5)), (255, 255, 255), -1)
                alpha = 0.2  # 透明度の設定（0.0から1.0の範囲）
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                # 座標が一致する場合に文字色を変更する
                if key in trajectory:
                    cv2.putText(image, value, (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(image, value, (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # 手のジェスチャーを識別
            for id, landmark in enumerate(hand_landmarks.landmark):
                if id == 8 and landmark.y < hand_landmarks.landmark[5].y:
                    # 人差し指が立っている状態
                    straight_counter += 1
                    if straight_counter >= straight_frame_threshold:
                        # 人差し指が立っている状態
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        trajectory.append((x, y))
                        # 軌跡を線で描画する
                        if len(trajectory) > is_count_start_number:
                            for i in range(1, len(trajectory)):
                                cv2.line(image, trajectory[i-1], trajectory[i], (0, 0, 255), 3)

                elif id == 8 and landmark.y > hand_landmarks.landmark[5].y:
                    # 軌跡を表示
                    if len(trajectory) > is_count_start_number:
                        # 軌跡から文字列を生成
                        word_suggestion = generate_word_suggestion(trajectory, image)
                        print("出力結果:", word_suggestion)
                        word_suggestions = ""
                        for word in word_suggestion:
                            word_suggestions += f"{word} "
                        # sentence += f"{calculate_next_word(sentence, word_suggestion)} "
                        trajectory = []
                    else:
                        trajectory = []
        cv2.putText(image, word_suggestions, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250,104,0), 2)
        cv2.putText(image, sentence, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (109,135,100), 4)
        # カメラの入力を表示
        cv2.imshow('Main', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('r'):
            sentence = ""
            word_suggestions = ""
finally:
    hands.close()
    cap.release()
    cv2.destroyAllWindows()