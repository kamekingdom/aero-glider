import cv2
import mediapipe as mp
from nltk.corpus import wordnet
from english_words import english_words
import pandas as pd

### Initial settings ###
english_words = english_words
print(f"登録単語数:{len(english_words)}")
keyboard_mapping = {
    (1, 1): "Q", (2, 1): "W", (3, 1): "E", (4, 1): "R", (5, 1): "T", (6, 1): "Y", (7, 1): "U", (8, 1): "I", (9, 1): "O", (10, 1): "P",
    (1, 2): "A", (2, 2): "S", (3, 2): "D", (4, 2): "F", (5, 2): "G", (6, 2): "H", (7, 2): "J", (8, 2): "K", (9, 2): "L", (10, 2): ";",
    (1, 3): "Z", (2, 3): "X", (3, 3): "C", (4, 3): "V", (5, 3): "B", (6, 3): "N", (7, 3): "M", (8, 3): ",", (9, 3): ".", (10, 3): "/"
}

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
prediction_words = []  # prediction words from the trajectory calculation
previous_key = None  # store the previous detected key
density_filter_threshold = 5 # これ以下のdensityの物は消去

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
                    print(f"x:{x} y:{y}")
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
                        if detected_key == previous_key:
                            previous_count = keyboard_density[-1]
                            previous_count[1] += 1
                        else:
                            keyboard_density.append([detected_key, 1])
                        previous_key = detected_key
                elif id == 8 and landmark.y > hand_landmarks.landmark[5].y:
                    # english_wordsに含まれる単語を検出するまで、カウントが低い順にキーを削除
                    if keyboard_density:
                        keyboard_filtered_list = [item for item in keyboard_density if item[1] > density_filter_threshold]
                        keyboard_density_word = "".join(item[0] for item in keyboard_filtered_list)
                        keyboard_density_word = keyboard_density_word.lower()
                        print(keyboard_density_word)

                    print(prediction_words)


        cv2.imshow('Main', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
