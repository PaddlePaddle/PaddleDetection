import cv2


def cut_video(video_path, frameToStart, frametoStop, saved_video_path):
    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)

    TOTAL_FRAME = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

    size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    videoWriter = cv2.VideoWriter(
        saved_video_path,
        apiPreference=0,
        fourcc=cv2.VideoWriter_fourcc(* 'mp4v'),
        fps=FPS,
        frameSize=(int(size[0]), int(size[1])))

    COUNT = 0
    while True:
        success, frame = cap.read()
        if success:
            COUNT += 1
            if COUNT <= frametoStop and COUNT > frameToStart:  # 选取起始帧
                videoWriter.write(frame)
        else:
            print("cap.read failed!")
            break
        if COUNT > frametoStop:
            break

    cap.release()
    videoWriter.release()

    print(saved_video_path)
