import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

sigma = 15
TemplateSize = 256
TemplateWidth = 0
TemplateHeight = 0
initial_scale = 1
lr = 0.125
Affine_Transformation_Num = 8
Affine_Transformation_Flag = True
Scale_Filter = [0.95, 1.0, 1.05]
roi = None
target_box_position = None
gauss_response = None
Ai = None
Bi = None

def Affine_Transformation(initial_fi):
    a = -3
    b = 3
    angle = a + (b - a) * np.random.uniform()

    rot_matrix = cv2.getRotationMatrix2D((initial_fi.shape[1]/2, initial_fi.shape[0]/2), angle, 1)
    affine_fi = cv2.warpAffine(np.uint8(initial_fi * 255), rot_matrix, (initial_fi.shape[1], initial_fi.shape[0]))
    affine_fi = affine_fi.astype(np.float32) / 255
    return affine_fi

def pre_processing(fi):
    height , width = fi.shape

    cosine_window_col = np.hanning(width)
    cosine_window_row = np.hanning(height)
    col, row = np.meshgrid(cosine_window_col, cosine_window_row)
    cosine_window = col * row

    fi = np.log(fi + 1)
    fi = (fi - np.mean(fi)) / (np.std(fi) + 1e-5)
    fi = fi * cosine_window

    return fi


def train_tracker(fi, Gi):
    initial_fi = fi
    fi = pre_processing(fi)
    Fi = np.fft.fft2(fi)

    Ai = Gi * np.conjugate(Fi)
    Bi = Fi * np.conjugate(Fi)

    for i in range(Affine_Transformation_Num):
        if Affine_Transformation_Flag:
            affine_fi = pre_processing(Affine_Transformation(initial_fi))
            Ai = Ai + Gi * np.conjugate(np.fft.fft2(affine_fi))
            Bi = Bi + np.fft.fft2(affine_fi) * np.conjugate(np.fft.fft2(affine_fi))
    return Ai , Bi


def mosse_tracker(video_path, video_name):
    print("运行--MosseTracker--")

    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    if len(video_path) == 0:
        print("打开本地摄像头")
        cap = cv2.VideoCapture(0)
    else:
        print("读取本地视频")
        cap = cv2.VideoCapture(video_path)
    
    ret, frame = cap.read()
    first_frame_lable = True

    if not os.path.exists('./GenerateVideo'):
        os.mkdir('./GenerateVideo')
    generate_video_name = './GenerateVideo/' + video_name + '_generate.avi'

    fps = 25
    generate_video = cv2.VideoWriter(generate_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame.shape[1], frame.shape[0]))

    while cap.isOpened():
        if first_frame_lable:
            first_frame_lable = False
            initial_frame = frame
            roi = list(cv2.selectROI(video_name, initial_frame, False, False))

            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = frame.astype(np.float32)

            height , width = frame.shape

            target_box_position = np.array([roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]]).astype(np.int64)

            fi = frame[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
            # fi = fi.astype(np.uint8)
            # cv2.imshow('', fi)

            x_coordinate, y_coordinate = np.meshgrid(np.arange(width), np.arange(height))
            # print(x_coordinate)
            # print(y_coordinate)
            # time.sleep(3)

            center_x = roi[0] +  roi[2] // 2
            center_y = roi[1] +  roi[3] // 2

            gauss_response = np.exp(-((np.square(x_coordinate - center_x) + np.square(y_coordinate - center_y)) / (2 * sigma)))
            # print(gauss_response.shape)

            gauss_response = (gauss_response - gauss_response.min()) / (gauss_response.max() - gauss_response.min())

            gi = gauss_response[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]

            initial_scale = TemplateSize / max(roi[2], roi[3]) 
            TemplateWidth = int(roi[2] * initial_scale)
            TemplateHeight = int(roi[3] * initial_scale)
            # TemplateWidth = int(roi[2])
            # TemplateHeight = int(roi[3])

            fi = cv2.resize(fi, (TemplateWidth, TemplateHeight))
            gi = cv2.resize(gi, (TemplateWidth, TemplateHeight))
            # print(fi.shape)

            Gi = np.fft.fft2(gi)

            Ai , Bi = train_tracker(fi, Gi)
        
        else:
            ret, frame = cap.read()
            initial_frame = frame
            if not ret:
                print("本地视频已加载结束")
                break

            if len(frame.shape) == 3:
                # cv2.COLOR_BGR2GRAY: RGB->GRAY
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = frame.astype(np.float32)
            
            height , width = frame.shape

            response = []
            for scale in Scale_Filter:
                center_x = roi[0] + 0.5 * roi[2]
                center_y = roi[1] + 0.5 * roi[3]
                width = roi[2] * scale
                height = roi[3] * scale
                fi = frame[int(center_y - height / 2) : int(center_y + height / 2), int(center_x - width / 2) : int(center_x + width / 2)]
                fi = pre_processing(cv2.resize(fi, (TemplateWidth, TemplateHeight)))

                total_scale = initial_scale * scale

                Hi = Ai / Bi
                response_output_frequency_domain = Hi * np.fft.fft2(fi)
                response_output_space_domain = np.fft.ifft2(response_output_frequency_domain)
                max_response = np.max(response_output_space_domain)

                max_position = np.where(response_output_space_domain == max_response)
                y_deviation = int(np.mean(max_position[0]) - response_output_space_domain.shape[0] / 2)
                x_deviation = int(np.mean(max_position[1]) - response_output_space_domain.shape[1] / 2) 
                x = int(roi[0] + x_deviation / total_scale)
                y = int(roi[1] + y_deviation / total_scale)
                w = int(roi[2] * scale)
                h = int(roi[3] * scale)
                response.append(((x, y, w, h, max_response, scale)))

            response.sort(key = lambda x: -x[4])
            (x, y, w, h, max_response, scale) = response[0]

            roi[0] = x
            roi[1] = y
            roi[2] = w
            roi[3] = h

            target_box_position[0] = np.clip(roi[0], 0, frame.shape[1])
            target_box_position[1] = np.clip(roi[1], 0, frame.shape[0])
            target_box_position[2] = np.clip(roi[0] + roi[2], 0, frame.shape[1])
            target_box_position[3] = np.clip(roi[1] + roi[3], 0, frame.shape[0])
            target_box_position = target_box_position.astype(np.int64)

            fi = frame[target_box_position[1]:target_box_position[3], target_box_position[0]:target_box_position[2]]
            fi = pre_processing(cv2.resize(fi, (TemplateWidth, TemplateHeight)))

            x_coordinate, y_coordinate = np.meshgrid(np.arange(frame.shape[1]), np.arange(frame.shape[0]))
            center_x = roi[0] + roi[2] // 2
            center_y = roi[1] + roi[3] // 2
            gauss_response = np.exp(-((np.square(x_coordinate - center_x) + np.square(y_coordinate - center_y)) / (2 * sigma)))
            gauss_response = (gauss_response - gauss_response.min()) / (gauss_response.max() - gauss_response.min())
            gi = gauss_response[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
            gi = cv2.resize(gi, (TemplateWidth, TemplateHeight))
 
            Gi = np.fft.fft2(gi)
            Ai = lr * (Gi  * np.conjugate(np.fft.fft2(fi))) + (1 - lr) * Ai
            Bi = lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - lr) * Bi

            cv2.rectangle(initial_frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 0, 255), 1)

            cv2.imshow(video_name, initial_frame)
            generate_video.write(initial_frame)

            key= cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
    cap.release()
    generate_video.release()


video_path  = ""

# 获取视频名称
if '/' in video_path:
    video_name = video_path.split('/')[-1].split('.')[0]
elif '\\' in video_path:
    video_name = video_path.split('\\')[-1].split('.')[0]
mosse_tracker(video_path, video_name)