import os
import cv2
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_list_path", help="여러 비디오가 있는 폴더의 경로, 경로안에 비디오 파일만 있어야함.", default="/video")
    parser.add_argument("--save_path", help="저장될 경로.", default="/video_image")
    parser.add_argument("--non_zero_threshold", help="이미지 픽셀 변화량 기준.", default=10)
    parser.add_argument("--non_zero_limit", help="이미지 변화 갯수 (0~100%) 퍼센트.", default=100)
    args = parser.parse_args()

    #--------------------------------------변수 설정 필요 -----------------------------------------------------------#

    # 비디오 경로와 이미지 저장 경로를 설정합니다.
    #video_list_path = "/media/sj/data/datasets/20230303" #여러 비디오가 있는 폴더의 경로, 경로안에 비디오 파일만 있어야함.
    #save_path = "/media/sj/data/datasets/20230303_colon_image/" #저장될 경로
    save_name = "images/Sequence" 

    #-------------------------------------option--------------------------------------------------------------------#

    # 이미지 픽셀 변화량 기준
    #non_zero_threshold = 10 #

    #이미지 변화 갯수 (0~100%) 퍼센트
    # 0 이면 조금이라도 차이있으면 데이터로 넣음
    # 파라미터가 클수록 작은 차이가 발생하는 이미지는 데이터에서 삭제
    #non_zero_limit = 100

    #--------------------------------------------------------------------------------------------------------------#

    # 비디오 파일 목록을 가져옵니다.
    video_list = os.listdir(args.video_list_path)

    # 이미지 시퀀스 번호를 초기화합니다.
    video_num = 0

    # 폴더가 없으면 폴더를 생성합니다.
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)    

    split_path = os.path.join(args.save_path,"splits")
    if not os.path.exists(split_path):
        os.makedirs(split_path)    

    # 이미지 경로를 저장할 텍스트 파일을 생성합니다.
    train_save_txt_path = os.path.join(split_path,"train_files.txt")
    val_save_txt_path = os.path.join(split_path,"val_files.txt")
    test_save_txt_path = os.path.join(split_path,"test_files.txt")

    train_f = open(train_save_txt_path, "w")

    # 모든 비디오 파일에 대해 반복합니다.
    for vid in video_list:
        # 현재 비디오의 프레임 번호를 초기화합니다.
        frame_num = 0

        # 현재 처리 중인 비디오 번호를 출력합니다.
        print("Processed video", video_num)

       

        
        # 비디오 파일 경로를 가져옵니다.
        video_path = os.path.join(args.video_list_path, vid)
        
        # 비디오 캡처 객체를 생성합니다.
        cap = cv2.VideoCapture(video_path)
    
        # 비교할 이전 프레임을 저장할 리스트를 초기화합니다.
        prev = []

        # 모든 프레임에 대해 반복합니다.
        while True:
            # 비디오 프레임을 읽습니다.
            retval, frame = cap.read()

            # 더 이상 프레임이 없으면 반복문을 종료합니다.
            if not retval:
                break

            # 이전 프레임 리스트가 비어있으면 현재 프레임을 추가합니다.
            if len(prev) == 0:
                prev.append(frame)
                continue

            # 이전 프레임 리스트에서 프레임을 꺼내옵니다.
            prev_image = prev.pop()

            # 현재 프레임과 이전 프레임의 차이를 계산합니다.
            diff = cv2.absdiff(frame, prev_image)

            # 차이 이미지를 그레이스케일로 변환합니다.
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # 이진화를 수행합니다.
            ret, dst = cv2.threshold(gray_diff, args.non_zero_threshold, 255, cv2.THRESH_BINARY)

            # 이진화된 이미지에서 0이 아닌 픽셀 수를 계산합니다.
            nonzero = cv2.countNonZero(dst)

            # 차이가 없으면 이전 프레임 리스트에 현재 프레임을 추가하고 다음 프레임으로 넘어갑니다.
            if nonzero <= args.non_zero_limit:
                prev.append(frame)
                continue

            # 이미지 저장 경로를 생성합니다.
            file_dir_path = save_name + str(video_num).zfill(2)
            mkpath = os.path.join(args.save_path, file_dir_path)
            os.makedirs(mkpath, exist_ok=True)

            # 이미지를 저장합니다.
            image_save_path = os.path.join(args.save_path, file_dir_path, "frame" + str(frame_num).zfill(4) + ".jpg")
            cv2.imwrite(image_save_path, frame)

            # 이미지 경로를 저장할 텍스트 파일에 이미지 경로를 저장합니다.
            txt_save_path = file_dir_path + "/frame" + str(frame_num).zfill(4) + ".jpg"
            train_f.write(txt_save_path + "\n")

            # 현재 프레임 번호를 증가시킵니다.
            frame_num += 1
            
        # 다음 비디오의 이미지 시퀀스 번호를 업데이트합니다.
        video_num += 1    
    train_f.close()

    # 랜덤하게 10%를 추출하여 validation set과 test set으로 나눕니다.
    with open(train_save_txt_path, 'r') as train_file:
        lines = train_file.readlines()
        num_train = len(lines)
        indices = list(range(num_train))
        val_indices = set(random.sample(indices, int(num_train * 0.1)))
        test_indices = set(random.sample(indices, int(num_train * 0.1)))

    # train, validation, test set에 해당하는 파일을 생성합니다.
    with open(val_save_txt_path, 'w') as val_file, open(test_save_txt_path, 'w') as test_file:
        for i, line in enumerate(lines):
            if i in val_indices:
                # validation set에 이미지 경로를 저장합니다.
                val_file.write(line)
            
            if i in test_indices:
                # test set에 이미지 경로를 저장합니다.
                test_file.write(line)

       

if __name__ == '__main__':
    main()