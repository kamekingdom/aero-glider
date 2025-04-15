import cv2
import mediapipe as mp

# MediaPipeの初期化
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# カメラキャプチャの初期化
cap = cv2.VideoCapture(0)

# 軌跡座標のリスト
trajectory = []
is_count_start_number = 5

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
                        cv2.putText(image, "Measuring in progress", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # 指の先端の座標を追加
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        trajectory.append((x, y))
                        if len(trajectory) > is_count_start_number:
                            for point in trajectory:
                                cv2.circle(image, point, 5, (0, 0, 255), -1)
                    elif id == 8 and landmark.y > hand_landmarks.landmark[5].y:
                        # 軌跡を表示
                        if len(trajectory) > is_count_start_number:
                            trajectory_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            print("軌跡情報")
                            for point in trajectory:
                                print(point)
                            trajectory = []
                        else:
                            trajectory = []



        # カメラの入力を表示
        cv2.imshow('Main', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        

    cap.release()
    cv2.destroyAllWindows()


