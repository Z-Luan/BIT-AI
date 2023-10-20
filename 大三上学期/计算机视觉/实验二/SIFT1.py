import numpy as np
import cv2
import functools
from matplotlib import pyplot as plt

# SIFT算法难度不小，当时困扰了作者很长时间，故将代码注释详细写出，希望读者能快速上手。

# 容许误差范围
Tolerance = 1e-7

# 生成基础图像
# 论文作者通过实验发现, 对原始图像放大2倍, 识别效果更佳, 所以先对原始图像放大2倍(双线性插值法)
def Generate_BaseImage(image, sigma, assumed_blur):

    # cv2.resize (InputArray src, OutputArray dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR )
    # src: 输入原始图像
    # dst: 输出改变形状后的图像，图像内容没有变化
    # dsize: 如果这个参数不为0，那么就代表将原始图像缩放到这个Size(width，height)指定的大小
    #        如果这个参数为0，那么原始图像缩放之后的大小就要通过公式 dsize = Size(round(fx*src.cols), round(fy*src.rows)) 来计算：
    # fx: width方向的缩放比例，如果它是0，那么它就会按照(double)dsize.width/src.cols来计算
    # fy: height方向的缩放比例，如果它是0，那么它就会按照(double)dsize.height/src.rows来计算
    # interpolation: 指定插值方式{INTER_NEAREST - 最邻近插值 ; INTER_LINEAR - 双线性插值(默认方法)}
    # 最近邻插值法: 不需要计算新图像中像素点的数值，直接找到原始图像中对应的像素点，将原始图像中对应像素点的数值赋值给新图像中的像素点
    #              根据对应关系找到原始图像中的对应像素点公式如下：src_x = des_x * src_w/des_w ; src_y = des_y * src_h/des_h
    #              计算结果可能不是整数, 此时根据四舍五入在原始图像中找到最近的像素点进行插值
    # 最近邻插值法会破坏原始图像中像素的渐变关系
    # 双线性插值法: https://zhuanlan.zhihu.com/p/110754637
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # assumed_blur: 一张图片本身具有一定的模糊，默认为0.5, 放大2倍后的图像模糊程度为 assumed_blur * 2
    # 如果图像经过 sigma_1 的高斯核模糊，然后再经过 sigma_2 的模糊，其本质等于图像经过 sigma = sqrt(sigma_1**2 + sigma_2**2) 的高斯模糊
    # 因为放大2倍的图像已经默认经过 assumed_blur * 2 的高斯模糊了, 那想要做成结果为 sigma 的模糊, 就再进行一个 sqrt((sigma ** 2) - (2 * assumed_blur) ** 2))的高斯模糊即可 
    sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

# 计算高斯金字塔的层数
def Compute_NumOctaves(image_shape):
    # np.log: 默认以 e 为底的自然对数
    # round: 按四舍五入的规则对浮点数取整, 当整数部分以0结束时, round 一律是向下取整
    # 下式保证高斯金字塔最顶层图像的尺寸不会太小(至少大于3), 便于在空间域和尺度域中寻找局部极值点
    # min(image_shape)是图像的最短边
    return int(round(np.log(min(image_shape)) / np.log(2) - 1))

# 生成高斯金字塔的滤波器
def Generate_GaussianKernels(sigma, num_intervals):

    # num_images_per_octave: 每个组内的图像数量
    num_images_per_octave = num_intervals + 3

    k = 2 ** (1. / num_intervals)
    gaussian_kernels = np.zeros(num_images_per_octave) 
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        # 确定高斯金字塔中每组内各图像的模糊程度
        sigma_previous = (k ** (image_index - 1)) * sigma
        # 每组图像中，上一层图像已经经过 sigma_1 的高斯核模糊，然后再经过 sqrt(k * sigma_1**2 - sigma_1**2) 的模糊，其本质等于图像经过 sigma_2 的高斯模糊
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels

# 生成高斯金字塔
def Generate_GaussianPyramid(image, num_octaves, gaussian_kernels):
    gaussian_pyramid = []

    # num_octaves: 高斯金字塔的层数(组数)
    for octave_index in range(num_octaves):
        gaussian_pyramid_in_octave = []
        gaussian_pyramid_in_octave.append(image)
        # 从某一组的第1张图像一直求到第 num_images_per_octave 张图像, 就得到了该组内的所有图像, 然后把该组的倒数第三张图像缩小2倍后作为下一组的基础图像(保持尺度空间的连续性)
        for gaussian_kernel in gaussian_kernels[1:]:
            # cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType) 实现高斯低通滤波
            # src: 输入图像, 输出图像的大小和类型与 src 相同
            # ksize: 高斯内核大小, ksize.width 和 ksize.height 可以不同, 但​​它们都必须为正数和奇数, 也可以为零, 然后根据sigmaX和sigmaY计算得出
            # sigmaX: X方向上的高斯核标准偏差
            # sigmaY: Y方向上的高斯核标准差, 如果sigmaY为零, 则将其设置为等于sigmaX
            #         如果两个sigmas为零，则分别从ksize.width和ksize.height计算得出
            #         为了完全控制结果，而不管将来可能对所有这些语义进行的修改，建议指定所有ksize，sigmaX和sigmaY
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_pyramid_in_octave.append(image)
        gaussian_pyramid.append(gaussian_pyramid_in_octave)
        octave_base = gaussian_pyramid_in_octave[-3]
        image = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
    return np.array(gaussian_pyramid, dtype=object)


# 生成高斯差分金字塔
def generate_DoGPyramid(gaussian_pyramid):

    dog_pyramid = []

    for gaussian_pyramid_in_octave in gaussian_pyramid:
        dog_pyramid_in_octave = []
        # zip 如果各个迭代器的元素个数不一致, 则返回列表长度与最短的对象相同
        for first_image, second_image in zip(gaussian_pyramid_in_octave, gaussian_pyramid_in_octave[1:]):
            # cv2.subtract(src1, src2, dst=None, mask=None, dtype=None)
            # src1：作为被减数的图像数组
            # src2：作为减数的图像数组
            # dst：可选参数,输出结果保存的变量，默认值为None，如果为非None，输出图像保存到dst对应的实参中
            # mask：图像掩膜，可选参数，为8位单通道的灰度图像
            #       用于指定要更改的输出图像数组的元素，即输出图像像素只有mask对应位置元素不为0的部分才输出，否则该位置像素的所有通道分量都设置为0
            # dtype：可选参数，输出图像数组的深度，即图像单个像素值的位数(如RGB用三个字节表示，则为24位)
            dog_pyramid_in_octave.append(cv2.subtract(second_image, first_image))
        dog_pyramid.append(dog_pyramid_in_octave)
    return np.array(dog_pyramid, dtype=object)

# 计算某像素点是不是其26邻域内的极值点
def Is_Extremum(first_subimage, second_subimage, third_subimage, threshold):

    center_pixel_value = second_subimage[1, 1]
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return np.all(center_pixel_value >= first_subimage) and \
                   np.all(center_pixel_value >= third_subimage) and \
                   np.all(center_pixel_value >= second_subimage[0, :]) and \
                   np.all(center_pixel_value >= second_subimage[2, :]) and \
                   center_pixel_value >= second_subimage[1, 0] and \
                   center_pixel_value >= second_subimage[1, 2]
        elif center_pixel_value < 0:
            return np.all(center_pixel_value <= first_subimage) and \
                   np.all(center_pixel_value <= third_subimage) and \
                   np.all(center_pixel_value <= second_subimage[0, :]) and \
                   np.all(center_pixel_value <= second_subimage[2, :]) and \
                   center_pixel_value <= second_subimage[1, 0] and \
                   center_pixel_value <= second_subimage[1, 2]
    return False

def Compute_Gradient(pixel_array):

    # f'(x) = (f(x + h) - f(x - h)) / (2 * h)
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return np.array([dx, dy, ds])

def Compute_Hessian(pixel_array):

    # f''(x) = (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # h = 1: f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # h = 1: f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return np.array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

# 极值点的精确定位
def Localize_Extremum(i, j, image_index, octave_index, num_intervals, dog_pyramid_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):

    extremum_is_outside_image = False
    image_shape = dog_pyramid_in_octave[0].shape

    # num_attempts_until_convergence: 如果5次之后还不能让三个偏移量值都小于0.5，就放弃这个极值点
    for attempt_index in range(num_attempts_until_convergence):
        # 需要从 uint8 转换为 float32 以计算导数，需要将像素值重新调整到[0,1]范围内以应用Lowe阈值
        first_image, second_image, third_image = dog_pyramid_in_octave[image_index-1:image_index+2]

        # np.stack 数组堆叠
        Cube = np.stack([first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2],third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        # 极值点在x y z三个方向的梯度
        gradient = Compute_Gradient(Cube)
        # hessian 矩阵用来去除边缘响应
        hessian = Compute_Hessian(Cube)
        # numpy.linalg.lstsq 拟合最小二乘解, 得到极值点偏移量
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        # round: 按四舍五入的规则对浮点数取整, 当整数部分以0结束时, round 一律是向下取整
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        # 确保新的 3 * 3 * 3 立方体没有超出范围
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break

    if extremum_is_outside_image:
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        return None

    functionValueAtUpdatedExtremum = Cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
    
    # 消除边缘响应
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        # np.trace: 求解 array 的迹
        # np.det: 求解 array 的行列式
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det = np.linalg.det(xy_hessian)
        if xy_hessian_det > 0 and (xy_hessian_trace ** 2) / xy_hessian_det < ((eigenvalue_ratio + 1) ** 2) / eigenvalue_ratio:
            # KeyPoint()类:
            # pt(x,y): 特征点的坐标
            # octave: 得到特征点的金字塔层数
            # size(): 包含特征点图像的尺度
            # response: 响应强度
            keypoint = cv2.KeyPoint()
            # 保存特征点的位置，尺度，这里的位置是针对扩展的那个最底层金字塔的坐标而言的
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * (2 ** (octave_index + 1))
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    return None

# 计算特征点方向
# 为了使描述子具有旋转不变性，需要利用图像的局部特征为给每一个特征点分配一个主方向，并将坐标进行旋转至特征点方向
# 使用图像梯度的方法求取局部区域的稳定方向，对于在 DOG 金字塔中检测出的特征点，采集其所在高斯金字塔图像 3σ 领域窗口内像素点的梯度和方向分布特征
def Compute_Keypoint_Orientation(keypoint, octave_index, gaussian_pyramid, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):

    keypoints_orientations = []
    image_shape = gaussian_pyramid.shape

    # keypoint 中保存的是相对高斯金字塔最底层图像坐标而言特征点的位置和尺度
    scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))  
    # 按照 3 sigma 原则确定领域
    radius = int(round(radius_factor * scale))
    # 按二维高斯函数分布加权求和
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_pyramid[region_y, region_x + 1] - gaussian_pyramid[region_y, region_x - 1]
                    dy = gaussian_pyramid[region_y - 1, region_x] - gaussian_pyramid[region_y + 1, region_x]
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    # np.arctan2 的输入不仅仅是正切值, 而是要输入两个数x1, x2,或者是两者的数组, 正切值是两者的比值x1/x2, 返回弧度
                    # np.rad2deg 将角度从弧度转换为度数
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    # 加权求和
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2)) 
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    # 为了防止某个梯度方向角度因受到噪声的干扰而突变，我们还需要对梯度方向直方图进行平滑处理
    # 平滑公式: https://blog.csdn.net/sakurakawa/article/details/120833167
    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.

    # np.where(condition,x,y) 当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回x，当条件不成立时where返回y
    # np.where(condition) 当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式
    # np.roll 沿着给定轴滚动数组元素, 超出最后位置的元素将会滚动到第一个位置, 1表示水平滚动
    orientation_max = max(smooth_histogram)
    # 在进行插值时, 我们只在主方向以及辅方向所在的柱子序号 j 满足 H(j) > H(j-1) and H(j) > H(j+1) 的柱子进行插值, 三个点可以确定一条抛物线
    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # 方向直方图的峰值代表了该特征点处邻域梯度的方向，以直方图中最大值作为该特征点的主方向
            # 为了增强匹配的鲁棒性，只保留峰值大于主方向峰值 80％ 的方向作为该特征点的辅方向
            # 在实际编程实现中，对于有多个方向的特征点，把该特征点复制多份，并将辅方向值分别赋给这些复制后的特征点
            # 并且离散的梯度方向直方图要进行插值拟合处理，来求得更精确的方向角度值
            # 插值更新公式: https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < Tolerance:
                orientation = 0
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_orientations.append(new_keypoint)
    return keypoints_orientations



# 寻找尺度空间的极值点
def Find_ScaleSpaceExtrema(gaussian_pyramid, dog_pyramid, num_intervals, sigma, image_border_width, contrast_threshold=0.04):

    # np.floor: 对输入的多维数组逐元素进行向下取整
    threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)
    keypoints = []

    for octave_index, dog_pyramid_in_octave in enumerate(dog_pyramid):
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_pyramid_in_octave, dog_pyramid_in_octave[1:], dog_pyramid_in_octave[2:])):
            # (i, j) 是 3x3 array 的中心
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if Is_Extremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                        localization_result = Localize_Extremum(i, j, image_index + 1, octave_index, num_intervals, dog_pyramid_in_octave, sigma, contrast_threshold, image_border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_orientations = Compute_Keypoint_Orientation(keypoint, octave_index, gaussian_pyramid[octave_index][localized_image_index])
                            for keypoint_orientation in keypoints_orientations:
                                keypoints.append(keypoint_orientation)
    return keypoints


def Compare_Keypoints(keypoint1, keypoint2):

    # keypoint 中保存的是相对高斯金字塔最底层图像坐标而言特征点的位置和尺度
    # 位置坐标越小的像素点排序越前
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    # 尺度越大的像素点排序越前
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    # 梯度方向越小的像素点排序越前
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    # 梯度值越大的像素点排序越前
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    # octave越大的像素点排序越前
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def Remove_DuplicateKeypoints(keypoints):

    if len(keypoints) < 2:
        return keypoints

    # cmp_to_key 将比较函数转化为关键字
    # sort 默认升序排序
    # 自定义比较函数 compare(A,B) return 正数表示 A 排序在 B 右边, return 负数表示 A 排序在 B 左边, return 0 表示按循环访问顺序排列
    keypoints.sort(key = functools.cmp_to_key(Compare_Keypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints

# 将特征点位置转换到原图对应的位置
def ConvertKeypointsToInputImageSize(keypoints):

    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
        keypoint.size *= 0.5
        # & 按位与
        # ~ 取反
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)

    return converted_keypoints

# 从特征点计算倍组数，层数，尺寸
def UnpackOctave(keypoint):

    octave = keypoint.octave & 255
    # >> 将二进制数右移
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    return octave, layer, scale


# 特征点生成描述子
def Generate_Descriptors(keypoints, gaussian_pyramid, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
# window_width 子区域大小

    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = UnpackOctave(keypoint)
        gaussian_image = gaussian_pyramid[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = np.round(scale * np.array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        # deg2rad 将度数转化为弧度
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        # 子区域的像素梯度大小按 sigma = 0.5 * window_width的高斯函数加权计算
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []

        # 前两个维度增加了2以考虑边界效应
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))  

        # 计算区域半径大小, 公式参考: https://blog.csdn.net/sakurakawa/article/details/120833167
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
        # 确保计算区域位于图像内   
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))     

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                # 将坐标轴旋转为特征点的方向, 通过与旋转矩阵相乘可以得到像素点旋转后的坐标
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                # 旋转后的采样点落在子区域中的坐标
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                # print(row_rot, col_rot, row_bin, col_bin)
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            # 三线性插值
            # 公式: https://en.wikipedia.org/wiki/Trilinear_interpolation
            # np.floor: 对输入的多维数组逐元素进行向下取整
            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            # print(row_bin_floor, col_bin_floor, orientation_bin_floor)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten() 
        # 门限化
        # numpy.linalg.norm 默认二范数
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        # 为了去除光照变化的影响，需要进行归一化处理，对于图像灰度值整体漂移，图像各点的梯度是邻域像素相减得到，所以也能去除
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), Tolerance)
        # 乘以512, 在0和255之间, 并进行饱和处理, 以将float32转换为无符号字符
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')

# 计算输入图像的 SIFT 特征点和描述子
def Compute_Keypoint_Descriptor(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    # np.astype 转换数组的数据类型
    image = image.astype('float32')
    base_image = Generate_BaseImage(image, sigma, assumed_blur)
    num_octaves = Compute_NumOctaves(base_image.shape)
    gaussian_kernels = Generate_GaussianKernels(sigma, num_intervals)
    gaussian_pyramid = Generate_GaussianPyramid(base_image, num_octaves, gaussian_kernels)
    dog_pyramid = generate_DoGPyramid(gaussian_pyramid)
    keypoints = Find_ScaleSpaceExtrema(gaussian_pyramid, dog_pyramid, num_intervals, sigma, image_border_width)
    keypoints = Remove_DuplicateKeypoints(keypoints)
    keypoints = ConvertKeypointsToInputImageSize(keypoints)
    descriptors = Generate_Descriptors(keypoints, gaussian_pyramid)
    return keypoints, descriptors

# FLANN库是目前最完整的最近邻开源库
# FLANN匹配器接收两个参数：indexindexParams对象和searchParams对象
# 本实验中我们使用了5棵树的核密度树索引算法，FLANN可以并行处理此算法，
# 同时对每棵树执行50次检查或者遍历，检查次数越多，可以提供的精度也越高，但是计算成本也就更高

img1 = cv2.imread('Minion1.jpg')           
img2 = cv2.imread('Minion2.jpg') 
img1 = cv2.resize(img1, (int(img1.shape[1] / 4), int(img1.shape[0] / 4)))
img2 = cv2.resize(img2, (int(img2.shape[1] / 4), int(img2.shape[0] / 4)))

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

kp1, des1 = Compute_Keypoint_Descriptor(img1)
kp2, des2 = Compute_Keypoint_Descriptor(img2)

index_params = dict(algorithm = 0, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

Good_Match = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        Good_Match.append(m)

if len(Good_Match) > 10:

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1
        newimg[:h2, w1:w1 + w2, i] = img2

    # 画出 SIFT 特征点匹配
    for m in Good_Match:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        bgr = np.random.randint(0,255,3,dtype=np.int32)#随机颜色
        cv2.line(newimg, pt1, pt2, (np.int(bgr[0]),np.int(bgr[1]),np.int(bgr[2])))

    cv2.imshow('Match_img', newimg)
    cv2.waitKey(0)
else:
    print("没有足够多好的匹配 - %d/%d" % (len(Good_Match), 10))