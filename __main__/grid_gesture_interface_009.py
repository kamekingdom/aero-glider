# ver.009
# 1. 少し最適化
# 2. 辞書に単語を追加できるように調整

import cv2
import mediapipe as mp
from nltk.corpus import wordnet
import pandas as pd
import difflib
import math
import numpy as np
from nltk.corpus import words
import string

### mediapipe setting ###
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
frame_rate = 30
cap.set(cv2.CAP_PROP_FPS, frame_rate)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
cv2.namedWindow("Main", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Main", 800, 600)

### definition of valuable ###
is_count_start_number = 5  # when the detect process start
trajectory = []  # trajectory database
keyboard_density = []  # save the keyboard density values
prediction_words = ["", "", ""]  # prediction words from the trajectory calculation
prediction_words_str = "" # prediction wordsを文字列に直したもの
previous_key = None  # store the previous detected key
density_filter_threshold = 5 # これ以下のdensityの物は消去
suggestion_word_num = 3 # 単語候補の出力数

### definition of functions ###
def generate_combinations():
    # 2文字の全通りの配列bを生成
    alphabet = string.ascii_lowercase
    combinations = [c1 + c2 for c1 in alphabet for c2 in alphabet]
    return combinations

def remove_substrings(word_list):
    # Generate all 2-character combinations
    combinations = set(c1 + c2 for c1 in string.ascii_lowercase for c2 in string.ascii_lowercase)
    for word in word_list:
        for substring in combinations.copy():  # Create a copy because we're modifying the set during iteration
            if substring in word:
                combinations.remove(substring)
    return combinations

def find_closest_words(input_string):
    closest_words = []
    similarities = []
    print(f"input_strings: {input_string}")
    input_string_length = len(input_string) # Calculate the length only once
    # Filter words based on input_string length
    filtered_words = [word for word in english_words if len(word) in range(input_string_length - 2, input_string_length + 3)]
    for word in filtered_words:
        similarity = difflib.SequenceMatcher(None, input_string, word).ratio()

        if len(closest_words) < suggestion_word_num:
            closest_words.append(word)
            similarities.append(similarity)
        else:
            min_similarity = min(similarities)
            min_index = similarities.index(min_similarity)
            if similarity > min_similarity:
                closest_words[min_index] = word
                similarities[min_index] = similarity

    sorted_words = [x for _, x in sorted(zip(similarities, closest_words), reverse=True)]
    print("output_strings:", sorted_words)
    return sorted_words, similarities

def remove_invalid_combinations(array):
    strings = [array[i][0] + array[i+1][0] for i in range(len(array)-1)]
    remove_list = []
    # 配列bに存在する文字列を処理
    for string in strings:
        if string in filtered_english_words:
            remove_list.append(string)
            string_index = strings.index(string)
            # a, bに着目し、値の大きいaのみを残す
            if string_index < len(array)-1:
                if array[string_index][1] > array[string_index+1][1]:
                    array.pop(string_index+1)
                else:
                    array.pop(string_index)
    print("filtered value:", remove_list)
    return array

# Use dict instead of set for english_words
additional_words = [".", "/", ";", ","]
english_words = words.words() + additional_words
english_words = {word: None for word in english_words}
filtered_english_words = remove_substrings(english_words.keys())

### Initial settings ###
filtered_english_words = remove_substrings(english_words)
print(f"registered words:{len(english_words)}")
print(f"not exists sequence strings:{filtered_english_words}")
keyboard_mapping = {
    (1, 1): "Q", (2, 1): "W", (3, 1): "E", (4, 1): "R", (5, 1): "T", (6, 1): "Y", (7, 1): "U", (8, 1): "I", (9, 1): "O", (10, 1): "P",
    (1, 2): "A", (2, 2): "S", (3, 2): "D", (4, 2): "F", (5, 2): "G", (6, 2): "H", (7, 2): "J", (8, 2): "K", (9, 2): "L", (10, 2): ";",
    (1, 3): "Z", (2, 3): "X", (3, 3): "C", (4, 3): "V", (5, 3): "B", (6, 3): "N", (7, 3): "M", (8, 3): ",", (9, 3): ".", (10, 3): "/"
}


### main process ###
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("カメラからのキャプチャができませんでした。")
            break

        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if not results:
            straight_counter = 0

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            ## keyboard GUI ##
            for pos, key in keyboard_mapping.items():
                x = (pos[0] - 1) * (image.shape[1] // 10)
                y = (pos[1] - 1) * (image.shape[0] // 5)
                overlay = image.copy()
                cv2.rectangle(overlay, (x, y), (x + (image.shape[1] // 10), y + (image.shape[0] // 5)), (255, 255, 255), -1)
                alpha = 0.2
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                cv2.putText(image, key, (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            for id, landmark in enumerate(hand_landmarks.landmark):
                if id == 8 and landmark.y < hand_landmarks.landmark[5].y:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    trajectory.append((x, y))
                    if len(trajectory) > is_count_start_number:
                        for i in range(1, len(trajectory)):
                            cv2.line(image, trajectory[i-1], trajectory[i], (0, 0, 255), 3)
                    # 検出された座標からキーを検出し、キーごとのカウントを更新
                    detected_key = None
                    for pos, key in keyboard_mapping.items():
                        key_x = (pos[0] - 1) * (image.shape[1] // 10)
                        key_y = (pos[1] - 1) * (image.shape[0] // 5)
                        key_width = image.shape[1] // 10
                        key_height = image.shape[0] // 5
                        if key_x <= x < key_x + key_width and key_y <= y < key_y + key_height:
                            detected_key = key
                            break
                    if detected_key:
                        detected_key = detected_key.lower()  # 小文字に変換
                        if detected_key == previous_key and keyboard_density:
                            previous_count = keyboard_density[-1]
                            previous_count[1] += 1
                        else:
                            keyboard_density.append([detected_key, 1])
                        previous_key = detected_key
                elif id == 8 and landmark.y > hand_landmarks.landmark[5].y:
                    # 四分位範囲を用いてdensityの外れ値を消去した値を作成
                    x = 0.5
                    keyboard_density_word = ""
                    if keyboard_density:
                        numbers = [item[1] for item in keyboard_density]
                        mean = np.mean(numbers)  # 平均値
                        std = np.std(numbers)    # 標準偏差
                        lower_bound = max(math.ceil(np.percentile(numbers, 40)), 7) # 40%の精度が良い
                        upper_bound = math.ceil(mean + 3 * std) # 範囲ではなくスリーシグマの方が精度が良い
                        print(f"lower: {lower_bound} / upper: {upper_bound}")
                        # lower_bound以下またはupper_bound以上なら外れ値と判定
                        keyboard_filtered_list = []
                        for item in keyboard_density:
                            if item[1] < upper_bound:
                                if lower_bound < item[1]:
                                    keyboard_filtered_list.append(item)
                            elif 10 < item[1]:
                                keyboard_filtered_list.append(item)
                                keyboard_filtered_list.append(item)
                        keyboard_filtered_list = remove_invalid_combinations(keyboard_filtered_list)


                        print(f"before: {keyboard_density}\nafter: {keyboard_filtered_list}")

                        keyboard_density_word = "".join(item[0] for item in keyboard_filtered_list)
                        if keyboard_density_word:
                            prediction_words, _ = find_closest_words(keyboard_density_word)
                            prediction_words_str = ', '.join(prediction_words)
                    # initialization #
                    keyboard_density = []
                    trajectory = []

        word_mapping = {(1, 1): prediction_words[1], (2, 2): prediction_words[0], (3, 3): prediction_words[2]}
        for pos, word in word_mapping.items():
            # Create the rectangles at the bottom of the screen
            rectangle_height = image.shape[0] // 5
            rectangle_width = image.shape[1] // 3
            rectangle_x = (pos[1] - 1) * rectangle_width
            rectangle_y = image.shape[0] - rectangle_height

            # Set the transparency (alpha) of the rectangles to 0.2
            overlay = image.copy()
            cv2.rectangle(overlay, (rectangle_x, rectangle_y), (rectangle_x + rectangle_width, rectangle_y + rectangle_height), (255, 255, 255), -1)
            image = cv2.addWeighted(overlay, 0.2, image, 0.8, 0)

            # Display the prediction words inside the rectangles
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            text_size, _ = cv2.getTextSize(word, font, font_scale, font_thickness)

            text_x = rectangle_x + (rectangle_width - text_size[0]) // 2
            text_y = rectangle_y + (rectangle_height + text_size[1]) // 2

            cv2.putText(image, word, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

        cv2.imshow('Main', image)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
