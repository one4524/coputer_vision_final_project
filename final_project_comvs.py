import cv2
import mediapipe as mp
import random
import numpy as np

cap = cv2.VideoCapture(1)  # 카메라 연결
if not cap.isOpened():
    print('Video open failed!')

mpHands = mp.solutions.hands    # 손 인식
hands = mpHands.Hands()        # 손 검출기

gun = cv2.imread('Glock_18C.jpg')  # 총 사진
fire = cv2.imread('bullet.jpg')  # 탄알 사진

bullet = 8   # 탄알 개수
kill = 0  # 킬수
count = 0

# 사람 전신 좌표
human_x = []
human_y = []
human_h = []
human_w = []
# 사람 상체 좌표
human_half_x = []
human_half_y = []
human_half_h = []
human_half_w = []

classifier_upBody = cv2.CascadeClassifier('haarcascade_upperbody.xml')  # 사람 상체 데이터 Load
classifier_fullBody = cv2.CascadeClassifier('haarcascade_fullbody.xml')  # 사람 전신 데이터 Load
if classifier_upBody.empty() or classifier_fullBody.empty():
    print('XML load failed!')

while True:
    ret, frame = cap.read()  # 카메라 영상 읽기
    if not ret:
        break

    h_f, w_f, c = frame.shape  # 영상 크기 가져오기

    # 노이즈 제거 - GaussianBlur 사용
    filter_frame = cv2.GaussianBlur(frame, (0, 0), 2.0)

    #
    fullBody = classifier_fullBody.detectMultiScale(filter_frame)
    upBody = classifier_upBody.detectMultiScale(filter_frame)

    # 전신 좌표 저장
    for (x, y, w, h) in fullBody:
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # 사각형 색 정하기
        human_x.append(x)
        human_y.append(y)
        human_h.append(h)
        human_w.append(w)
        # cv2.rectangle(frame, (x, y, w, h), color, 2)

    # 상체 좌표 저장
    for (x, y, w, h) in upBody:
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # 사각형 색 정하기
        human_half_x.append(x)
        human_half_y.append(y)
        human_half_h.append(h)
        human_half_w.append(w)
        # cv2.rectangle(frame, (x, y, w, h), color, 2)

    frameRGB = cv2.cvtColor(filter_frame, cv2.COLOR_BGR2RGB)  # mediapipe를 사용하기 위해 BGR영상을 RGB영상으로 변환
    results = hands.process(frameRGB)  # 손 인식

    # 인식한 손 가져오기
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 핸드의 각 관절 포인트의 ID와 좌표를 알아 내서 원하는 그림을 그려 넣을 수 있다.
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w_f), int(lm.y * h_f)  # cx : 관절 x 좌표, cy : r관절 y 좌표

                # 손목관절
                if id == 0:
                    wrist_x = cx
                    wrist_y = cy

                # 엄지 뿌리
                if id == 2:
                    thumb_mcp_x = cx
                    thumb_mcp_y = cy
                # 엄지 끝
                if id == 4:
                    thumb_x = cx
                    thumb_y = cy

                # 검지 첫 관절
                if id == 5:
                    index_finger_mcp_x = cx
                    index_finger_mcp_y = cy

                # 검지 끝
                if id == 8:
                    index_finger_x = cx
                    index_finger_y = cy

                # 중지 끝
                if id == 12:
                    middle_finger_x = cx
                    middle_finger_y = cy

                # 새끼손가락 뿌리
                if id == 17:
                    pinky_x = cx
                    pinky_y = cy

        ################################# 영상에 총 넣기 - 총 모양 손 제스처 인식(제스처 1) #########################################

        myGun_h = int(abs(wrist_y - thumb_mcp_y) * 1.7) + 1  # 총의 높이 설정
        index_finger_x -= int(abs(wrist_x - index_finger_x) / 10)  # 검지 끝 x좌표 조정
        index_finger_y -= int(abs(wrist_y - index_finger_y) / 4)  # 검지 끝 y좌표 조정
        hand_len_x = wrist_x - index_finger_x
        imt = index_finger_y - thumb_mcp_y
        imt_inv = thumb_mcp_y - index_finger_y


        # 총 삽입 조건  - 1. 중지 끝이 검지 끝보다 손바닥 안쪽에 있고 엄지뿌리보다 왼쪽에 있어야 한다. 2. 손목보다 엄지 뿌리가 위에 있어야 한다.
        if thumb_mcp_x > middle_finger_x > index_finger_x and wrist_y > thumb_mcp_y:

            # 추가 조건 - 오른손만 가능하도록 설정
            # 검지 끝이 손목보다 왼쪽에 있을 때에만 총 삽입하기 위해
            if hand_len_x > 0 and middle_finger_y > thumb_y:

                # 어파인 변환을 위한 좌표 설정
                if imt > 0:  # 검지가 아래쪽으로 향했을 떄
                    thumb_mcp_y -= int(abs(wrist_y - thumb_mcp_y) / 5)  # 엄지 뿌리 y좌표 조정
                    gun_set = cv2.resize(gun, (hand_len_x, myGun_h + int(imt*1.2)))
                    gun_h, gun_w, gun_c = gun_set.shape

                    src = np.array([[gun_w, 0], [0, gun_h], [gun_w, gun_h]], dtype=np.float32)
                    dst = np.array([[gun_w, 0], [0, gun_h], [gun_w, gun_h - imt]],
                                   dtype=np.float32)

                    masked_frame = frame[thumb_mcp_y: thumb_mcp_y + gun_h,
                                   index_finger_x: index_finger_x + gun_w]  # 총을 삽입할 프레임

                else:  # 검지가 위쪽으로 향했을 때
                    gun_set = cv2.resize(gun, (hand_len_x + int(imt_inv/5), myGun_h + int(imt_inv*1.2)))
                    gun_h, gun_w, gun_c = gun_set.shape
                    src = np.array([[0, 0], [gun_w, 0], [gun_w, gun_h]], dtype=np.float32)
                    dst = np.array([[0, 0], [gun_w, thumb_mcp_y - index_finger_y], [gun_w, gun_h]],
                                   dtype=np.float32)
                    masked_frame = frame[index_finger_y: index_finger_y + gun_h,
                                   index_finger_x: index_finger_x + gun_w]  # 총을 삽입할 프레임

            else:
                continue

            gray_gun = cv2.cvtColor(gun_set, cv2.COLOR_BGR2GRAY)  # 총 사진을 그레이스케일로 변환 - 이진화를 하기위해
            ret, gun_bin = cv2.threshold(gray_gun, 120, 255, cv2.THRESH_BINARY_INV)  # 총 사진 이진화 - 총만 추출하기 위해

            # 총을 손의 각도에 맞게 설정하기 위해 AffineTransform
            affine_matrix = cv2.getAffineTransform(src, dst)
            myGun = cv2.warpAffine(gun_set, affine_matrix, (0, 0))
            gun_bin = cv2.warpAffine(gun_bin, affine_matrix, (0, 0))

            gun_inv = cv2.bitwise_not(gun_bin)  # 이진화 반대로

            mf_h, mf_w, mf_c = masked_frame.shape  # 총을 삽입할 프레임 크기 가져오기
            print(gun_h, mf_h, gun_w, mf_w)
            # 총 프레임과 총을 삽입할 원래 영상의 프레임이 불일치하는 오류 방지
            if gun_h == mf_h and gun_w == mf_w:

                myGun = cv2.bitwise_and(myGun, myGun, mask=gun_bin)  # 총만 추출

                masked_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=gun_inv)    # 총이 들어갈 프레임 추출

                # 원래 영상에 총 삽입
                if imt > 0:  # 검지가 아래쪽으로 향했을 떄

                    frame[thumb_mcp_y: thumb_mcp_y + gun_h,
                    index_finger_x: index_finger_x + gun_w] = cv2.add(myGun, masked_frame)
                else:  # 검지가 위쪽으로 향했을 때

                    frame[index_finger_y: index_finger_y + gun_h,
                    index_finger_x: index_finger_x + gun_w] = cv2.add(myGun, masked_frame)
            else:
                continue


            ############## 총의 영점 지정 ################

            zero_point_x = index_finger_x - int(hand_len_x / 3) - 5

            # x 영점이 프레임 아웃일 때
            if zero_point_x < 0:
                zero_point_x = 1

            if imt > 0:  # 검지가 아래쪽일 때
                zero_point_y = index_finger_y + int(imt/6)
            else:  # 검지가 위쪽일 때
                zero_point_y = index_finger_y - int(imt_inv/6)

            # y 영점이 프레임 아웃일 때
            if zero_point_y < 0:
                zero_point_y = 1
            elif zero_point_y > h_f:
                zero_point_y = h_f - 1

            # 화면에 영점 표시
            cv2.circle(frame, (zero_point_x, zero_point_y), 5, (0, 0, 0), cv2.FILLED)

            ############################### 총쏘기 이벤트 처리 - 손 제스처 인식(제스처 2) ###################################
            hand_len_y = pinky_y - index_finger_mcp_y

            if index_finger_mcp_y - abs(hand_len_y) / 2 < thumb_y < index_finger_mcp_y + abs(
                    hand_len_y) / 2 and bullet > 0:

                fire_set = cv2.resize(fire, (int(hand_len_x / 6) + 1, int(myGun_h / 5) + 1))  # 탄알 사진 resize

                fire_h, fire_w, fire_c = fire_set.shape  # 탄알 크기 가져오기

                gray_fire = cv2.cvtColor(fire_set, cv2.COLOR_BGR2GRAY)  # 탄알 사진을 그레이스케일로 변환 - 이진화를 하기위해

                ret, fire_bin = cv2.threshold(gray_fire, 80, 255, cv2.THRESH_BINARY)  # 탄알 사진 이진화

                fire_inv = cv2.bitwise_not(fire_bin)  # 이진화 반대로

                masked_frame_fire = frame[index_finger_y: index_finger_y + fire_h,
                                    index_finger_x - fire_w: index_finger_x]  # 탄알을 삽입할 프레임

                frame_fire_h, frame_fire_w, frame_fire_c = masked_frame_fire.shape  # 탄알을 삽입할 프레임 가져오기

                # 탄알 프레임과 탄알을 삽입할 원래 영상의 프레임이 불일치하는 오류 방지
                if fire_h == frame_fire_h and fire_w == frame_fire_w:
                    fire_set = cv2.bitwise_and(fire_set, fire_set, mask=fire_bin)  # 탄알만 추출

                    masked_frame = cv2.bitwise_and(masked_frame_fire, masked_frame_fire,
                                                   mask=fire_inv)  # 탄알 넣을 부분만 추출

                    # 탄알 삽입
                    frame[index_finger_y: index_finger_y + fire_h,
                    index_finger_x - fire_w: index_finger_x] = cv2.add(fire_set, masked_frame_fire)

                else:
                    continue

                bullet -= 1  # 총알 개수 -1

                ######################## 사람을 맞췄을 때 이벤트 처리 #######################

                # 영점이 사람의 상체에 존재하면 킬
                # 사람의 전신을 대상으로
                if len(human_x) != 0:
                    for i in range(0, len(human_x)):
                        if human_x[i] < zero_point_x < human_x[i] + human_w[i]:
                            if human_y[i] < zero_point_y < human_y[i] + human_h[i] / 2:
                                kill += 1
                                break

                # 사람의 상체를 대상으로
                elif len(human_half_x) != 0:
                    for i in range(0, len(human_half_x)):
                        if human_half_x[i] < zero_point_x < human_half_x[i] + human_half_w[i]:
                            if human_half_y[i] < zero_point_y < human_half_y[i] + human_half_h[i]:
                                kill += 1
                                break

        else:
            ################################# 탄창 채우기 기능 - ok 손 제스처 인식(제스처 3) ###########################
            ok_x = abs(hand_len_x)  # ok손 제스처 x좌표
            ok_y = abs(index_finger_y - wrist_y)  #ok손 제스처 y좌표

            # 엄지와 검지 끝이 가까이 있고 중지가 엄지보다 높이 있으면 ok 손 모양으로 인식
            if index_finger_x - ok_x / 5 <= thumb_x <= index_finger_x + ok_x / 5 and index_finger_y - ok_y / 5 <= thumb_y <= index_finger_y + ok_y / 5:
                if thumb_y > middle_finger_y:
                    bullet = 8      # 탄창 채우기

    # 화면에 총알 개수와 킬 수 표시
    cv2.putText(frame, str(bullet) + "/8", (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
    cv2.putText(frame, "kill : " + str(kill), (w_f - 120, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
    cv2.imshow('frame', frame)

    # esc 누르면 종료
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
