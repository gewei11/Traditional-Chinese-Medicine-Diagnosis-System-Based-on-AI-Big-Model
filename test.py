import cv2
import numpy as np
import dlib
import time
import gradio as gr
from scipy import signal

# 常量
BUFFER_MAX_SIZE = 500       # 存储最近ROI平均值的数量
MAX_VALUES_TO_GRAPH = 50    # 在脉搏图中显示的最近ROI平均值的数量
MIN_HZ = 0.83       # 50 BPM - 允许的最小心率
MAX_HZ = 3.33       # 200 BPM - 允许的最大心率
MIN_FRAMES = 100    # 计算心率所需的最小帧数。值越高，计算越准确，但速度越慢。
DEBUG_MODE = False

 
# 创建并应用指定的Butterworth滤波器
def butterworth_filter(data, low, high, sample_rate, order=5):
    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)

 # 获取额头区域的感兴趣区域（ROI）
def get_forehead_roi(face_points):
    # 将点存储在Numpy数组中，以便轻松获取x和y的最小值和最大值
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)

    min_x = int(points[21, 0])
    min_y = int(min(points[21, 1], points[22, 1]))
    max_x = int(points[22, 0])
    max_y = int(max(points[21, 1], points[22, 1]))
    left = min_x
    right = max_x
    top = min_y - (max_x - min_x)
    bottom = max_y * 0.98
    return int(left), int(right), int(top), int(bottom)

 # 获取鼻子的感兴趣区域（ROI）
def get_nose_roi(face_points):
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y) 
    # 鼻子和脸颊
    min_x = int(points[36, 0])
    min_y = int(points[28, 1])
    max_x = int(points[45, 0])
    max_y = int(points[33, 1])
    left = min_x
    right = max_x
    top = min_y + (min_y * 0.02)
    bottom = max_y + (max_y * 0.02)
    return int(left), int(right), int(top), int(bottom)  
# 获取包括额头、眼睛和鼻子的感兴趣区域（ROI）。
# 注意：额头和鼻子的组合效果更好。这可能是由于此ROI包括眼睛，而眨眼会增加噪声。
def get_full_roi(face_points):
    points = np.zeros((len(face_points.parts()), 2))
    for i, part in enumerate(face_points.parts()):
        points[i] = (part.x, part.y)

    # 只保留对应于面部内部特征的点（例如嘴巴、鼻子、眼睛、眉毛）。
    # 齿尖轮廓的点被丢弃。
    min_x = int(np.min(points[17:47, 0]))
    min_y = int(np.min(points[17:47, 1]))
    max_x = int(np.max(points[17:47, 0]))
    max_y = int(np.max(points[17:47, 1])) 
    center_x = min_x + (max_x - min_x) / 2
    left = min_x + int((center_x - min_x) * 0.15)
    right = max_x - int((max_x - center_x) * 0.15)
    top = int(min_y * 0.88)
    bottom = max_y
    return int(left), int(right), int(top), int(bottom)

def sliding_window_demean(signal_values, num_windows):
    window_size = int(round(len(signal_values) / num_windows))
    demeaned = np.zeros(signal_values.shape)
    for i in range(0, len(signal_values), window_size):
        if i + window_size > len(signal_values):
            window_size = len(signal_values) - i
        curr_slice = signal_values[i: i + window_size]
        if DEBUG_MODE and curr_slice.size == 0:
            print ('Empty Slice: size={0}, i={1}, window_size={2}'.format(signal_values.size, i, window_size))
            print (curr_slice)
        demeaned[i:i + window_size] = curr_slice - np.mean(curr_slice)
    return demeaned

 # 对两个像素数组的绿色值求平均值
def get_avg(roi1, roi2):
    roi1_green = roi1[:, :, 1]
    roi2_green = roi2[:, :, 1]
    avg = (np.mean(roi1_green) + np.mean(roi2_green)) / 2.0
    return avg  
# 从列表中返回最大绝对值
def get_max_abs(lst):
    return max(max(lst), -min(lst))  
# 在GUI窗口中绘制心率图
def draw_graph(signal_values, graph_width, graph_height):
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)
    scale_factor_x = float(graph_width) / MAX_VALUES_TO_GRAPH 
    # 根据绝对值最大的值自动垂直缩放
    max_abs = get_max_abs(signal_values)
    scale_factor_y = (float(graph_height) / 2.0) / max_abs 
    midpoint_y = graph_height / 2
    for i in range(0, len(signal_values) - 1):
        curr_x = int(i * scale_factor_x)
        curr_y = int(midpoint_y + signal_values[i] * scale_factor_y)
        next_x = int((i + 1) * scale_factor_x)
        next_y = int(midpoint_y + signal_values[i + 1] * scale_factor_y)
        cv2.line(graph, (curr_x, curr_y), (next_x, next_y), color=(0, 255, 0), thickness=1)
    return graph  
# 在GUI窗口中绘制心率文本（BPM）
def draw_bpm(bpm_str, bpm_width, bpm_height):
    bpm_display = np.zeros((bpm_height, bpm_width, 3), np.uint8)
    bpm_text_size, bpm_text_base = cv2.getTextSize(bpm_str, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2.7,
                                                   thickness=2)
    bpm_text_x = int((bpm_width - bpm_text_size[0]) / 2)
    bpm_text_y = int(bpm_height / 2 + bpm_text_base)
    cv2.putText(bpm_display, bpm_str, (bpm_text_x, bpm_text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=2.7, color=(0, 255, 0), thickness=2)
    bpm_label_size, bpm_label_base = cv2.getTextSize('BPM', fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6,
                                                     thickness=1)
    bpm_label_x = int((bpm_width - bpm_label_size[0]) / 2)
    bpm_label_y = int(bpm_height - bpm_label_size[1] * 2)
    cv2.putText(bpm_display, 'BPM', (bpm_label_x, bpm_label_y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(0, 255, 0), thickness=1)
    return bpm_display 

# 在GUI窗口中绘制当前帧率
def draw_fps(frame, fps):
    cv2.rectangle(frame, (0, 0), (100, 30), color=(0, 0, 0), thickness=-1)
    cv2.putText(frame, 'FPS: ' + str(round(fps, 2)), (5, 20), fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1, color=(0, 255, 0))
    return frame  
# 在图形区域绘制文本
def draw_graph_text(text, color, graph_width, graph_height):
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)
    text_size, text_base = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=1)
    text_x = int((graph_width - text_size[0]) / 2)
    text_y = int((graph_height / 2 + text_base))
    cv2.putText(graph, text, (text_x, text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=color,
                thickness=1)
    return graph  
# 计算每分钟心跳次数（BPM）
def compute_bpm(filtered_values, fps, buffer_size, last_bpm):
    # 计算FFT
    fft = np.abs(np.fft.rfft(filtered_values))

    # 生成与FFT值对应的频率列表
    freqs = fps / buffer_size * np.arange(buffer_size / 2 + 1) 
    # 过滤掉FFT中不在[MIN_HZ, MAX_HZ]范围内的峰值，因为它们对应于不可能的心率值。
    while True:
        max_idx = fft.argmax()
        bps = freqs[max_idx]
        if bps < MIN_HZ or bps > MAX_HZ:
            if DEBUG_MODE:
                print ('BPM of {0} was discarded.'.format(bps * 60.0))
            fft[max_idx] = 0
        else:
            bpm = bps * 60.0
            break 
    # 心率不可能在两次采样之间变化超过10%，因此使用加权平均来平滑BPM与上一次BPM。
    if last_bpm > 0:
        bpm = (last_bpm * 0.9) + (bpm * 0.1) 
    return bpm  
def filter_signal_data(values, fps):
    # 确保数组中没有无穷大或NaN值
    values = np.array(values)
    np.nan_to_num(values, copy=False)


    # 通过去趋势和平滑信号
    detrended = signal.detrend(values, type='linear')
    demeaned = sliding_window_demean(detrended, 15)
    # 使用Butterworth带通滤波器过滤信号
    filtered = butterworth_filter(demeaned, MIN_HZ, MAX_HZ, fps, order=5)
    return filtered  
# 获取感兴趣区域的平均值。如果请求，还会在感兴趣区域周围绘制绿色矩形。
def get_roi_avg(frame, view, face_points, draw_rect=True):
    fh_left, fh_right, fh_top, fh_bottom = get_forehead_roi(face_points)
    nose_left, nose_right, nose_top, nose_bottom = get_nose_roi(face_points)

    # 绘制绿色矩形围绕我们的感兴趣区域（ROI）
    if draw_rect:
        cv2.rectangle(view, (fh_left, fh_top), (fh_right, fh_bottom), color=(0, 255, 0), thickness=2)
        cv2.rectangle(view, (nose_left, nose_top), (nose_right, nose_bottom), color=(0, 255, 0), thickness=2) 
    # 切出感兴趣区域（ROI）并求平均值
    fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
    nose_roi = frame[nose_top:nose_bottom, nose_left:nose_right]
    return get_avg(fh_roi, nose_roi)  
# 主函数。
def run_pulse_observer(detector, predictor, cap):
    output_path = "tszx_fusai\output.avi"
    # 尝试使用 XVID 编码器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height + 200))  # +200 for the graph

    roi_avg_values = []
    graph_values = []
    times = []
    last_bpm = 0
    graph_height = 200
    graph_width = 0
    bpm_display_width = 0
    i = 0
    bpm = 0
    bpm_value = 0
    bpm_value_flag = 0
    
    while cap.isOpened():
        ret_val, frame = cap.read()
        if not ret_val:
            print("Video capture ended or failed.")
            break
        view = np.array(frame)
        # 心率图占窗口宽度的75%。BPM占25%。
        if graph_width == 0:
            graph_width = int(view.shape[1] * 0.75)
        if bpm_display_width == 0:
            bpm_display_width = view.shape[1] - graph_width
        # 使用dlib检测人脸
        faces = detector(frame, 0)
        if len(faces) == 1:
            face_points = predictor(frame, faces[0])
            roi_avg = get_roi_avg(frame, view, face_points, draw_rect=True)
            roi_avg_values.append(roi_avg)
            times.append(time.time())
            # 缓冲区已满，因此弹出顶部值以丢弃它
            if len(times) > BUFFER_MAX_SIZE:
                roi_avg_values.pop(0)
                times.pop(0)
            curr_buffer_size = len(times)
            # 在计算脉搏之前，不要尝试计算心率，直到至少有最小帧数
            if curr_buffer_size > MIN_FRAMES:
                # 计算相关时间
                time_elapsed = times[-1] - times[0]
                fps = curr_buffer_size / time_elapsed  # 每秒帧数
                # 清理信号数据
                filtered = filter_signal_data(roi_avg_values, fps)

                graph_values.append(filtered[-1])
                if len(graph_values) > MAX_VALUES_TO_GRAPH:
                    graph_values.pop(0)
                # 绘制脉搏图
                graph = draw_graph(graph_values, graph_width, graph_height)
                # 计算并显示BPM
                bpm = compute_bpm(filtered, fps, curr_buffer_size, last_bpm)
                i += 1
                if i == 10:
                    bpm_value_flag = bpm_value // 10
                    bpm_value = 0
                    i = 0
                else:
                    bpm_value += bpm
                bpm_display = draw_bpm(str(int(round(bpm_value_flag))), bpm_display_width, graph_height)
                last_bpm = bpm
                # 显示FPS
                if DEBUG_MODE:
                    view = draw_fps(view, fps)
            else:
                # 如果没有足够的数据来计算HR，显示一个空的心率图，带有加载文本和BPM占位符
                pct = int(round(float(curr_buffer_size) / MIN_FRAMES * 100.0))
                loading_text = 'Computing pulse: ' + str(pct) + '%'
                graph = draw_graph_text(loading_text, (0, 255, 0), graph_width, graph_height)
                bpm_display = draw_bpm('--', bpm_display_width, graph_height)
        else:
            # 没有检测到人脸，因此我们必须清除值和时间的列表。否则，当再次检测到人脸时，时间戳之间会有间隙。
            del roi_avg_values[:]
            del times[:]
            graph = draw_graph_text('未检测到人脸', (0, 0, 255), graph_width, graph_height)
            bpm_display = draw_bpm('--', bpm_display_width, graph_height)

        graph = np.hstack((graph, bpm_display))
        view = np.vstack((view, graph))
        
        # 将处理后的帧写入视频文件
        out.write(view)

        cv2.waitKey(30)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path

def xinlv(input_video):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('tszx_fusai/shape_predictor_81_face_landmarks.dat')
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Video capture could not be opened.")
        return
    output_ = run_pulse_observer(detector, predictor, cap)
    return output_

    # run_pulse_observer() returns when the user has closed the window. Time to shut down.
    shut_down(cap)

# if __name__ == '__main__':
#     print(xinlv())

# Gradio界面
def display_video(input_video):
    video_path = xinlv(input_video)
    return video_path

# 使用Gradio Video组件播放视频
iface = gr.Interface(
    fn=display_video,
    inputs=None,  # 不需要输入
    outputs=gr.Video(),  # 输出为视频
    title="心率视频处理",
    description="这是一个处理并展示心率检测结果的视频应用"
)

# 启动Gradio界面
iface.launch()