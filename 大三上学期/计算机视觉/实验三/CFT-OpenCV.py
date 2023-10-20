import cv2
import sys
 
# 确认版本号
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
if __name__ == '__main__' :
  
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    # 选择相关滤波跟踪器
    tracker_type = tracker_types[2]
 
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.legacy.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.legacy.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.legacy.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.legacy.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.legacy.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.legacy.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.legacy.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.legacy.TrackerCSRT_create()
 
    video = cv2.VideoCapture("C:/Users/16176/Desktop/张鸾 1120202786 CV Project-3/VideoSource/Brightness.mp4")
 
    if not video.isOpened():
        print ("无法打开视频")
        sys.exit()
 
    ret, frame = video.read()
    if not ret:
        print ('无法读取视频文件')
        sys.exit()
 
    roi = cv2.selectROI(frame, False)
 
    tracker.init(frame, roi)
 
    while True:
        ret, frame = video.read()
        if not ret:
            break
         
        timer = cv2.getTickCount()
 
        # 更新滤波模板
        roi_flag , roi = tracker.update(frame)
 
        # 计算FPS 每秒播放帧数
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
 
        # 绘制目标框
        if roi_flag:
            p1 = (int(roi[0]), int(roi[1]))
            p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            cv2.putText(frame, "跟踪目标失败", (100 , 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75 , (0 , 0 , 255) , 2)
 
        # 显示相关滤波跟踪器类型
        cv2.putText(frame, tracker_type + " Tracker", (100 , 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75 , (50 , 170 , 50) , 2);
     
        # 显示FPS
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            quit()