import cv2
import mediapipe as mp
import nltk
import math

"""
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

from nltk.corpus import words

english_words = words.words()
"""

english_words = ["apple", "banana", "melon"]
print(english_words)
import difflib

# MediaPipeの初期化
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# カメラキャプチャの初期化
cap = cv2.VideoCapture(0)

# 軌跡座標のリスト
trajectory = []
is_count_start_number = 5

def find_closest_word(input_string):
    closest_word = english_words[0]
    highest_similarity = 0

    for word in english_words:
        similarity = difflib.SequenceMatcher(None, input_string, word).ratio()

        if similarity > highest_similarity and len(word) >= 3:
            highest_similarity = similarity
            closest_word = word
    
    print(f"input:\t{input_string}")
    print(f"close:\t{closest_word}")
    print(f"simiv:\t{highest_similarity}")

    return closest_word


import numpy as np

def smooth_trajectory(trajectory, window_size=5):
    smoothed_trajectory = []
    for i in range(len(trajectory)):
        start_idx = max(0, i - window_size)
        end_idx = min(len(trajectory), i + window_size + 1)
        window = trajectory[start_idx:end_idx]
        smoothed_point = np.mean(window, axis=0)
        smoothed_trajectory.append(smoothed_point)
    return smoothed_trajectory


# Function to generate output string based on the trajectory
def generate_output_string(trajectory):
    # Define the mapping of keys on the keyboard
    keyboard_mapping = {
        (1, 1): "1", (2, 1): "2", (3, 1): "3", (4, 1): "4", (5, 1): "5", (6, 1): "6", (7, 1): "7", (8, 1): "8", (9, 1): "9", (10, 1): "0",
        (1, 2): "Q", (2, 2): "W", (3, 2): "E", (4, 2): "R", (5, 2): "T", (6, 2): "Y", (7, 2): "U", (8, 2): "I", (9, 2): "O", (10, 2): "P",
        (1, 3): "A", (2, 3): "S", (3, 3): "D", (4, 3): "F", (5, 3): "G", (6, 3): "H", (7, 3): "J", (8, 3): "K", (9, 3): "L", (10, 3): ";",
        (1, 4): "Z", (2, 4): "X", (3, 4): "C", (4, 4): "V", (5, 4): "B", (6, 4): "N", (7, 4): "M", (8, 4): ",", (9, 4): ".", (10, 4): "/",
        (1, 5): " ", (2, 5): " ", (3, 5): " ", (4, 5): " ", (5, 5): " ", (6, 5): " ", (7, 5): " ", (8, 5): " ", (9, 5): " ", (10, 5): " ",
    }

    output_string = ""
    trajectory = smooth_trajectory(trajectory)
    for i in range(0, len(trajectory), 5):
        sub_trajectory = trajectory[i:i+5]
        if len(sub_trajectory) == 5:
            # Calculate the average x and y coordinates of the sub-trajectory points
            avg_x = sum(point[0] for point in sub_trajectory) // 5
            avg_y = sum(point[1] for point in sub_trajectory) // 5

            # Determine the key based on the average coordinates
            key = (avg_x // (image.shape[1] // 10) + 1, avg_y // (image.shape[0] // 5) + 1)

            # Check if the key is valid in the mapping
            if key in keyboard_mapping:
                output_string += keyboard_mapping[key]

    output_string = find_closest_word(output_string)
    return output_string

# 直線判定のパラメータ
threshold_angle = 15  # 直線とみなす角度の閾値（単位：度）
straight_counter = 0  # 直線と判定されたフレームの数
straight_frame_threshold = 10  # 直線とみなすフレームの連続数の閾値
    
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:
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

        # 検出結果の描画
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 手のジェスチャーを識別
                for id, landmark in enumerate(hand_landmarks.landmark):
                    if id == 8 and landmark.y < hand_landmarks.landmark[5].y:
                        # 人差し指が立っている状態
                        straight_counter += 1
                        cv2.putText(image, "Straight Finger", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if straight_counter >= straight_frame_threshold:
                            cv2.putText(image, "Measuring in progress", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            # 人差し指が立っている状態
                            x = int(landmark.x * image.shape[1])
                            y = int(landmark.y * image.shape[0])
                            trajectory.append((x, y))
                            if len(trajectory) > is_count_start_number:
                                for point in trajectory:
                                    cv2.circle(image, point, 5, (0, 0, 255), -1)
                    elif id == 8 and landmark.y > hand_landmarks.landmark[5].y:
                        # 軌跡を表示
                        if len(trajectory) > is_count_start_number:
                            # 軌跡から文字列を生成
                            output_string = generate_output_string(trajectory)
                            print("出力結果:", output_string)
                            trajectory = []
                        else:
                            trajectory = []
                        

        # カメラの入力を表示
        cv2.imshow('Main', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()