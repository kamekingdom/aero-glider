import cv2
import mediapipe as mp
import numpy as np
import difflib
import nltk
from nltk.corpus import wordnet
from english_words import english_words

english_words = english_words

print(f"登録単語数:{len(english_words)}")

keyboard_mapping = {
    (1, 1): "Q", (2, 1): "W", (3, 1): "E", (4, 1): "R", (5, 1): "T", (6, 1): "Y", (7, 1): "U", (8, 1): "I", (9, 1): "O", (10, 1): "P",
    (1, 2): "A", (2, 2): "S", (3, 2): "D", (4, 2): "F", (5, 2): "G", (6, 2): "H", (7, 2): "J", (8, 2): "K", (9, 2): "L", (10, 2): ";",
    (1, 3): "Z", (2, 3): "X", (3, 3): "C", (4, 3): "V", (5, 3): "B", (6, 3): "N", (7, 3): "M", (8, 3): ",", (9, 3): ".", (10, 3): "/",
}

cv2.namedWindow("Main", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Main", 800, 600)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
frame_rate = 30
cap.set(cv2.CAP_PROP_FPS, frame_rate)

trajectory = []
is_count_start_number = 5

from gensim.models import Word2Vec

import nltk
from nltk.corpus import wordnet

def calculate_next_word(sentence, new_words):
    tokens = nltk.word_tokenize(sentence)
    tagged_tokens = nltk.pos_tag(tokens)
    nouns = [token for token, tag in tagged_tokens if tag.startswith('NN')]

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

suggestion_word_num = 5

def find_closest_words(input_string):
    closest_words = []
    similarities = []

    print(f"入力結果: {input_string}")

    for word in english_words:
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

threshold_angle = 15
straight_counter = 0
straight_frame_threshold = 10
word_suggestions = ""
sentence = ""

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

            for key, value in keyboard_mapping.items():
                x = (key[0] - 1) * (image.shape[1] // 10)
                y = (key[1] - 1) * (image.shape[0] // 5)
                overlay = image.copy()
                cv2.rectangle(overlay, (x, y), (x + (image.shape[1] // 10), y + (image.shape[0] // 5)), (255, 255, 255), -1)
                alpha = 0.2
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                if key in trajectory:
                    cv2.putText(image, value, (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(image, value, (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            for id, landmark in enumerate(hand_landmarks.landmark):
                if id == 8 and landmark.y < hand_landmarks.landmark[5].y:
                    straight_counter += 1
                    if straight_counter >= straight_frame_threshold:
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        trajectory.append((x, y))
                        if len(trajectory) > is_count_start_number:
                            for i in range(1, len(trajectory)):
                                cv2.line(image, trajectory[i-1], trajectory[i], (0, 0, 255), 3)

                elif id == 8 and landmark.y > hand_landmarks.landmark[5].y:
                    if len(trajectory) > is_count_start_number:
                        word_suggestion = generate_word_suggestion(trajectory, image)
                        print("出力結果:", word_suggestion)
                        word_suggestions = ""
                        for id, word in enumerate(word_suggestion):
                            word_suggestions += f"{id+1}:{word} "
                        trajectory = []
                    else:
                        trajectory = []
        cv2.putText(image, word_suggestions, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250,104,0), 2)
        cv2.putText(image, sentence, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (109,135,100), 4)
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
