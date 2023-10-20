import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

TemplateSize = 256
TemplateWidth = 0
TemplateHeight = 0

WindowSize = None
BlockSize = None
BlockStride = None
CellSize = None
Padding = 2.5
Scale_h = 0.
Scale_w = 0.
sigma = 0.6
lambdar = 0.0001
lr = 0.01
X = None
Next_X = None
roi = None
Falpha = 0.
Scale_Filter = [0.95,1,1.05]


def Kernel_Generate_Circular_Matrix(x , y):
    a = np.sum(x ** 2) + np.sum(y ** 2)
    b = np.fft.ifft2(np.sum(np.multiply(np.conj(np.fft.fft2(x)), np.fft.fft2(y)), axis=0))
    c = a - 2.0 * b
    circular_matrix = np.exp(-1 / sigma ** 2 * np.abs(c) / c.size)
    return circular_matrix 

def Training(x, gaussian):
    circular_matrix = Kernel_Generate_Circular_Matrix(x, x)
    alphaf = np.fft.fft2(gaussian) / np.conj((np.fft.fft2(circular_matrix) + 0.0001))
    return alphaf

def Detecting():
    circular_matrix = Kernel_Generate_Circular_Matrix(X , Next_X)
    response_output = np.real(np.fft.ifft2(np.multiply(Falpha, np.fft.fft2(circular_matrix))))
    return response_output
    

def kcf_tracker(video_path, video_name):
    global roi , TemplateHeight , TemplateWidth , Scale_h , Scale_w , X , Falpha , Next_X

    print("运行--KcfTracker--")

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

            roi = (cv2.selectROI(video_name, initial_frame, False, False))

            x, y, w, h = roi
            center_x = x + w // 2
            center_y = y + h // 2
            roi = (center_x, center_y, w, h)

            scale = TemplateSize / float(max(w, h))
            TemplateHeight = int(h * scale) // 4 * 4 
            TemplateWidth = int(w * scale) // 4 * 4 
            
            w = int(w * Padding) // 2 * 2
            h = int(h * Padding) // 2 * 2

            x = int(center_x - w // 2)
            y = int(center_y - h // 2)
            sub_frame = frame[y : y + h , x : x + w , :]
            
            resized_frame = cv2.resize(sub_frame, (TemplateWidth, TemplateHeight))

            WindowSize = (TemplateWidth , TemplateHeight)
            BlockSize = (8, 8)
            BlockStride = (4, 4)
            CellSize = (4, 4)
            BinNum = 9
            HOG = cv2.HOGDescriptor(WindowSize, BlockSize, BlockStride, CellSize, BinNum)
            hist = HOG.compute(resized_frame, WindowSize, padding = (0, 0))

            Cell_w_num = TemplateWidth // CellSize[0] - 1
            Cell_h_num = TemplateHeight // CellSize[1] - 1
            HOG_feature = hist.reshape(Cell_w_num, Cell_h_num, 36).transpose(2, 1, 0)

            Scale_h = float(Cell_h_num) / h
            Scale_w = float(Cell_w_num) / w

            cosine_window_col = np.hanning(Cell_w_num)
            cosine_window_row = np.hanning(Cell_h_num)
            col, row = np.meshgrid(cosine_window_col, cosine_window_row)
            cosine_window = col * row

            X = HOG_feature * cosine_window
            sigma = np.sqrt(Cell_w_num * Cell_h_num) / Padding * 0.125
            Starting_ordinate, Starting_abscissa = Cell_h_num // 2, Cell_w_num // 2
            ordinate, abscissa = np.mgrid[-Starting_ordinate : -Starting_ordinate + Cell_h_num, -Starting_abscissa : -Starting_abscissa + Cell_w_num]
            gaussian = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((abscissa ** 2 + ordinate ** 2)/(2. * sigma ** 2)))

            Falpha = Training(X , gaussian)

        else:
            ret, frame = cap.read()
            initial_frame = frame
            if not ret:
                print("本地视频已加载结束")
                break
            
            center_x, center_y, w, h = roi

            Max_response = -1
            for scale in Scale_Filter:

                roi_scale = (int(center_x), int(center_y), int(w * scale), int(h * scale))

                w_scale = int(roi_scale[2] * Padding) // 2 * 2
                h_scale = int(roi_scale[3] * Padding) // 2 * 2
                x_scale = int(roi_scale[0] - w_scale // 2)
                y_scale = int(roi_scale[1] - h_scale // 2)
  
                sub_frame = frame[y_scale : y_scale + h_scale , x_scale : x_scale + w_scale , :]
                
                try:
                    resized_frame = cv2.resize(sub_frame, (TemplateWidth, TemplateHeight))
                except:
                    continue

                WindowSize = (TemplateWidth , TemplateHeight)
                BlockSize = (8, 8)
                BlockStride = (4, 4)
                CellSize = (4, 4)
                BinNum = 9
                HOG = cv2.HOGDescriptor(WindowSize, BlockSize, BlockStride, CellSize, BinNum)
                hist = HOG.compute(resized_frame, WindowSize, padding = (0, 0))

                Cell_w_num = TemplateWidth // CellSize[0] - 1
                Cell_h_num = TemplateHeight // CellSize[1] - 1
                Next_HOG_feature = hist.reshape(Cell_w_num, Cell_h_num, 36).transpose(2, 1, 0)

                Scale_h = float(Cell_h_num) / h_scale
                Scale_w = float(Cell_w_num) / w_scale
                # print("Scale_w , Scale_h" , Scale_w , Scale_h)

                # 生成余弦窗
                cosine_window_col = np.hanning(Cell_w_num)
                cosine_window_row = np.hanning(Cell_h_num)
                col, row = np.meshgrid(cosine_window_col, cosine_window_row)
                cosine_window = col * row

                Next_X = Next_HOG_feature * cosine_window

                response_output = Detecting()
                response_output_h ,  response_output_w = response_output.shape
                # print('response_output_h ,  response_output_w' , response_output_h ,  response_output_w)
                idx = np.argmax(response_output)

                # print('idx', idx)
                response_output_value = np.max(response_output)

                x_deviation , y_deviation = 0 , 0
                if response_output_value > Max_response:
                    Max_response = response_output_value
                    x_deviation = int((idx % response_output_w - response_output_w / 2) / Scale_w)
                    y_deviation = int((idx / response_output_w - response_output_h / 2) / Scale_h)
                    Best_w = int(w * scale)
                    Best_h = int(h * scale)
                    Best_Next_X = Next_X
            
            # print(x_deviation , y_deviation)
            roi = (center_x + x_deviation, center_y + y_deviation, Best_w, Best_h)
            # print('roi  1', roi)
            X = X * (1 - lr) + Best_Next_X * lr

            Cell_w_num , Cell_h_num = X.shape[2] , X.shape[1]
            sigma = np.sqrt(Cell_w_num * Cell_h_num) / Padding * 0.125
            Starting_ordinate, Starting_abscissa = Cell_h_num // 2, Cell_w_num // 2
            ordinate, abscissa = np.mgrid[-Starting_ordinate : -Starting_ordinate + Cell_h_num, -Starting_abscissa : -Starting_abscissa + Cell_w_num]
            gaussian = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((abscissa ** 2 + ordinate ** 2)/(2. * sigma ** 2)))

            Next_Flphaf = Training(Best_Next_X, gaussian)
            Falpha = Falpha * (1 - lr) + Next_Flphaf * lr

            start_x , start_y = roi[0] - roi[2] // 2 , roi[1] - roi[3] // 2

            cv2.rectangle(initial_frame, (start_x, start_y), (start_x + roi[2], start_y + roi[3]), (0, 0, 255), 1)

            cv2.imshow(video_name, initial_frame)
            generate_video.write(initial_frame)

            key= cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    generate_video.release()

video_path  = ""

if '/' in video_path:
    video_name = video_path.split('/')[-1].split('.')[0]
elif '\\' in video_path:
    video_name = video_path.split('\\')[-1].split('.')[0]
kcf_tracker(video_path, video_name)