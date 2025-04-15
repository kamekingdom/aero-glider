import cv2
import mediapipe as mp

def detect_extended_finger():
    cap = cv2.VideoCapture(0)  # カメラをキャプチャするためのオブジェクトを作成します

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    while True:
        ret, frame = cap.read()  # カメラからフレームを読み込みます
        if not ret:
            break

        # Mediapipeを使用して手のポーズを検出します
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 人差し指の指先の座標を取得します
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * frame.shape[1])
                y = int(index_finger_tip.y * frame.shape[0])

                # 指の先端がピーンと伸びているかどうかを判定します
                is_extended = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y

                # 指の先端の座標とピーンと伸びているかどうかを画像上に表示します
                cv2.circle(frame, (x, y), 5, (0, 255, 0) if is_extended else (0, 0, 255), -1)

        # 画像を表示します
        cv2.imshow("Extended Finger Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_extended_finger()
