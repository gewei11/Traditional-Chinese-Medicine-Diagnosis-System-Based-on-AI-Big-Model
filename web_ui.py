import gradio as gr
import openai_api_request
import torch
import librosa
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import requests
import io
import cv2
import dlib
import time
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
    output_path = "new.avi"
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
    predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Video capture could not be opened.")
        return
    output_ = run_pulse_observer(detector, predictor, cap)
    return output_

# Gradio界面
def display_video(input_video):
    video_path = xinlv(input_video)
    return video_path


#其他功能
class wang:
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)


    def build_transform(input_size):
        MEAN, STD = wang.IMAGENET_MEAN, wang.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform


    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio


    def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = wang.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images


    def load_image(image_file, input_size=448, max_num=6):
        image = Image.open(image_file).convert('RGB')
        transform = wang.build_transform(input_size=input_size)
        images = wang.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    def wang_zhen(image_path):
        path = "hf-models/Mini-InternVL-Chat-2B-V1-5"
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # 在`max_num中设置瓷砖的最大数量`
        pixel_values = wang.load_image(str(image_path), max_num=1).to(torch.bfloat16).cuda()

        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )


        question = """
        面色观察：
        您的面色整体看起来是怎样的？（红润、苍白、黄、青、黑等）
        您的面部有没有出现斑点或痘痘？
        如果有斑点或痘痘，它们主要分布在面部的哪个区域？（额头、脸颊、下巴等）

        舌象观察：
        您的舌质颜色是怎样的？（淡红、红、绛红、紫暗等）
        您的舌苔颜色如何？（薄白、白腻、黄、黄腻、灰黑等）
        舌苔主要分布在舌头的哪个部位？（全舌、舌尖、舌根等）
        您的舌体形态有什么特点吗？（胖大、瘦薄、裂纹等）

        眼睛观察：
        您的眼睛是否有血丝？
        如果有血丝，它们主要分布在眼球的哪个部位？（眼球周围、眼角等）
        您的眼白颜色看起来如何？（清澈、黄染等）

        其他观察：
        您的皮肤状态是怎样的？（干燥、湿润、油腻等）
        您的指甲状态如何？（色泽、月牙白等）
        您的身体形态是怎样的？（肥胖、消瘦、肌肉发达等）

        综合分析：
        根据这些望诊信息，您可能属于哪种体质类型？（阴虚、阳虚、气虚、湿热等）
        您可能存在哪些健康问题？（需要进一步检查的问题或初步诊断）

        建议与指导：
        根据您的情况，有哪些饮食建议？（适宜食物、禁忌食物等）
        您应该如何调整生活习惯？（作息时间、运动方式等）
        您应该如何管理情绪？（减压方法、心态调整等）
        """ 
        state = "" # 初始化
        for res in model.chat(tokenizer, pixel_values, question, generation_config):
            state += res
            yield state  # 流式输出


# 提示词设置
system_prompt = {
            "role": "system",
            "content":
            "你是一个经验丰富的老中医，知识全面丰富，你的回答准确、清晰有条理、多样化。"
        }

#闻诊功能函数
class wen_1:
    # 定义设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义检测气息平稳度和力度的函数
    def detect_breath_properties(audio, sr):
        # 计算音频信号的短时均方根能量
        frame_length = int(sr * 0.025)  # 25ms的帧长
        hop_length = int(sr * 0.01)     # 10ms的帧移
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        energy_diff = np.diff(energy, prepend=energy[0])
        breathiness_score = np.mean(np.abs(energy_diff))  # 气息变化率

        # 计算基频
        n_fft = frame_length
        f0, voiced_flag = librosa.piptrack(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        f0_mean = np.mean(f0[voiced_flag > 0]) if np.any(voiced_flag) else 0  # 基频平均值，只考虑有声部分

        # 评估气息平稳度和力度
        breath_stability = "Stable" if breathiness_score < np.percentile(energy_diff, 30) else "Unstable"
        breath_intensity = "Light" if f0_mean < 100 else "Strong"  # 假设基频低于100Hz为轻，否则为重

        return breath_stability, breath_intensity, breathiness_score, f0_mean

    def huxi(audio_path):
        try:
            # 加载音频文件
            audio, sr = librosa.load(str(audio_path), sr=None)  # 使用原始采样率加载音频
            
            # 检测气息属性
            breath_stability, breath_intensity, breathiness_score, f0_mean = wen_1.detect_breath_properties(audio, sr)
            result = (f"呼吸稳定性：{breath_stability}\n"
                    f"呼吸强度： {breath_intensity}\n"
                    f"呼吸评分（越低表示越稳定）：{breathiness_score}\n"
                    f"基频（F0）平均值（作为呼吸强度的衡量标准）：{f0_mean} Hz")
            return result
        except Exception as e:
            return f"错误："+str(e)
    def zhengchuang(audio_path):
        try:
            huxi = wen_1.huxi(audio_path)
            if (len(huxi) > 20000):
                raise gr.Error("输入长度不能超过 20000 字，请重新输入")
            messages = [system_prompt]
            messages.append({"role": "user", "content": str("请根据以下症状给出一些中医的建议，回答要求：只以专业中医的知识来给出症状分析即可.症状："+huxi)})
            complete_message = ''
            res = openai_api_request.simple_chat(messages=messages, use_stream=True)
            if (res):
                for chunk in res:
                    delta_content = chunk.choices[0].delta.content
                    complete_message += delta_content
                    # print(delta_content, end='')  # 不换行拼接输出当前块的内容
                    yield complete_message  # gradio 需要返回完整可迭代内容
            else:
                raise gr.Error("API 服务正在启动中，请稍后重试")
        except Exception as e:
            return f"错误："+str(e)

#向老中医提问获取回答
class wen_2:
    #问诊的功能，问答
    def chat(message):
        try:
            if (len(message) > 20000):
                raise gr.Error("输入长度不能超过 20000 字，请重新输入")
            messages = [system_prompt]
            messages.append({"role": "user", "content": str("请根据以下症状给出一些中医的建议，回答要求：只以专业中医的知识来给出症状分析即可.症状："+message)})
            complete_message = ''
            res = openai_api_request.simple_chat(messages=messages, use_stream=True)
            if (res):
                for chunk in res:
                    delta_content = chunk.choices[0].delta.content
                    complete_message += delta_content
                    # print(delta_content, end='')  # 不换行拼接输出当前块的内容
                    yield complete_message  # gradio 需要返回完整可迭代内容
                print(message)
                print("\nComplete message:", complete_message)
            else:
                raise gr.Error("API 服务正在启动中，请稍后重试")
        except Exception as e:
            return f"错误："+str(e)
    
def process_zhong(chatbot_result, jie2_result, jie1_result):
    try:
        messages = [system_prompt]
        combined_result = f"问诊结果: {chatbot_result}\n闻诊结果: {jie2_result}\n望诊结果: {jie1_result}"
        messages.append({"role": "user", "content": str("请根据以下症状给出一些中医的建议，回答要求：只以专业中医的知识来给出最终诊断即可."+combined_result)})
        complete_message = ''
        res = openai_api_request.simple_chat(messages=messages, use_stream=True)
        if (res):
            for chunk in res:
                delta_content = chunk.choices[0].delta.content
                complete_message += delta_content
                # print(delta_content, end='')  # 不换行拼接输出当前块的内容
                yield complete_message  # gradio 需要返回完整可迭代内容
            print("\nComplete message:", complete_message)
        else:
            raise gr.Error("API 服务正在启动中，请稍后重试")
    except Exception as e:
        return f"错误："+str(e)
    
class HealthAdviceGenerator:
    def __init__(self):
        self.doc_chunks = []
        with open('yangsheng.txt', 'r', encoding='utf-8') as file:
            self.doc_chunks = [line.strip() for line in file]
        self.index_ip = faiss.read_index('yangsheng.index')

    # L2规范化函数
    def l2_normalization(self, embedding: np.ndarray) -> np.ndarray:
        if embedding.ndim == 1:
            return embedding / np.linalg.norm(embedding)
        else:
            return embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    # 使用FAISS索引查找相关文档的函数
    def find_related_doc(self, query: str, top_k: int = 5) -> str:
        query_embedding = np.array(self.get_embedding([query]))
        query_embedding_normalized = self.l2_normalization(query_embedding)
        D, I = self.index_ip.search(query_embedding_normalized, top_k)
        top_k_similar = I.tolist()[0]
        res = ''
        for i in range(top_k):
            if i < len(top_k_similar) and top_k_similar[i] < len(self.doc_chunks):
                res += f"{self.doc_chunks[top_k_similar[i]]}" + '\n\n'
            else:
                print(f"索引错误: i={i}, top_k_similar={top_k_similar}, doc_chunks长度={len(self.doc_chunks)}")
                break
        return res

    def generate_health_advice(self, health_topic):
        related_doc = self.find_related_doc(health_topic)
        PROMPT_TEMPLATE = "基于以下已知信息，请简洁并专业地回答用户的问题。\n{DOCS}\n问题：{QUERY}\n回答："
        try:
            messages = [system_prompt]
            messages.append({'role': 'user', 'content': PROMPT_TEMPLATE.format(DOCS=related_doc, QUERY=health_topic)})
            complete_message = ''
            res = openai_api_request.simple_chat(messages=messages, use_stream=True)
            if res:
                for chunk in res:
                    delta_content = chunk.choices[0].delta.content
                    complete_message += delta_content
                    yield complete_message  # gradio 需要返回完整可迭代内容
                print("\nComplete message:", complete_message)
            else:
                raise Exception("API 服务正在启动中，请稍后重试")
        except Exception as e:
            return f"错误："+str(e)
    
# 定义API URL和请求头
API_URL = "https://ai.gitee.com/api/endpoints/shuimo-shenqiu/stable-diffusion3/inference"
headers = {
    "Authorization": "Bearer eyJpc3MiOiJodHRwczovL2FpLmdpdGVlLmNvbSIsInN1YiI6IjM5MDgwIn0._whsXGS-qZVnRUQju-p95zbCOPu_55J1kYWJw0Vztk5F2wf4bKQIe5gSSOi4dJB3Y0E-OejXglALPGvdvst4Dg",
    "Content-Type": "application/json"
}

def query(prompt):
    payload = {
        "inputs": f"Chinese herbal medicine, [{prompt}], traditional medicine, dried roots, botanical illustration, close-up view, detailed texture, medicinal plant, natural remedies, earthy tones, intricate structure, natural colors, natural lighting, traditional Chinese medicine, (masterpiece: 2), best quality, ultra highres, original, extremely detailed, perfect lighting。."
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    
    # 检查响应内容类型
    print(response.headers.get('Content-Type'))  # 打印响应的Content-Type
    print(response.content[:100])  # 打印响应内容的前100个字节

    # 确保响应内容是图像数据再进行处理
    if 'image' in response.headers.get('Content-Type'):
        image_bytes = response.content
        image = Image.open(io.BytesIO(image_bytes))
        return image
    else:
        raise ValueError("API response is not an image.")
    
def zhongcaoyao(name):
    try:
        messages = [system_prompt]
        messages.append({"role": "user", "content": str("假设你是一个中草药专家，你将对用户介绍用户提到的中草药,包括它的生成条件以及能治疗什么疾病.中草药名："+name)})
        complete_message = ''
        res = openai_api_request.simple_chat(messages=messages, use_stream=True)
        if (res):
            for chunk in res:
                delta_content = chunk.choices[0].delta.content
                complete_message += delta_content
                yield complete_message  # gradio 需要返回完整可迭代内容
            print("\nComplete message:", complete_message)
        else:
            raise gr.Error("API 服务正在启动中，请稍后重试")
    except Exception as e:
        return f"错误："+str(e)


#例子


# chatbot = gr.Chatbot(height=550, label=" InternLM 2.5")
with gr.Blocks() as demo:
    with gr.Tab("问诊服务"):
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    gr.HTML(value="<p>望：通过观察患者的面色、舌苔等,了解患者的整体身体状况。</p>")
                    gr.HTML(value="<p>请患者拍摄或上传面部状况，舌苔状况，亦可都有。</p>")
                    image = gr.Image(sources=["webcam","upload"], label="Image", type="filepath")
                    wang1 = gr.Button("开始望诊")
                    jie1 = gr.Textbox(label="望诊结果")
                    wang1.click(fn=wang.wang_zhen, inputs=image, outputs=jie1)
                    
                with gr.Column():
                    gr.HTML(value="<p>闻：通过聆听患者的呼吸、发声等声音，了解患者的呼吸系统状况。</p>")
                    gr.HTML(value="<p>请患者读以下几个字：宫、商、角、徵、羽。每个字发声不低于一秒。</p>")
                    audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio File")
                    wen1_1 = gr.Button("开始闻诊")
                    Wen1 = wen_1
                    jie2 = gr.Textbox(label="闻诊结果")
                    # examples2=["test.mp3"]
                    wen1_1.click(Wen1.zhengchuang, inputs=audio, outputs=jie2)
                    
                with gr.Column():
                    gr.HTML(value="<p>问：通过询问患者的性别、年龄、生病季节、症状、病症持续时间、病史等，了解患者的具体病情。</p>")
                    chatbot = gr.Textbox(elem_id="chat-box", label="提问", lines=3)
                    wen2_2 = gr.Button("开始问诊", visible=True)
                    jie3 = gr.Textbox(label="问诊结果",lines=5)
                    wen2_2.click(wen_2.chat, inputs=chatbot, outputs=jie3)
            with gr.Column():
                gr.HTML(value="<p>最终诊断结果</p>")
                zhong = gr.Button("查看最终诊断结果")
                wenzhen = gr.Textbox(label="问诊结果")
                # 按钮点击事件处理                
                zhong.click(
                    fn=process_zhong,
                    inputs=[chatbot, jie2, jie1],
                    outputs=wenzhen
                )
                
    with gr.Tab("养生服务"):
        with gr.Column():
            gr.HTML(value="<p>根据患者的身体状况和需求，提供个性化的养生建议。</p>")
            gr.HTML(value="<p>请输入您的身体状况和需求。</p>")
            yangsheng = gr.Textbox(elem_id="chat-box", label="养生建议", lines=5)
            gr.Examples(label='常见问题示例', examples=[["如何通过生活方式改善睡眠质量？"], ["如何通过日常活动来预防骨质疏松症？"], ["如何通过饮食来提高免疫力？"]], inputs=yangsheng)
            jianyi = gr.Button("生成养生建议")
            jie4 = gr.Textbox(label="养生建议结果")
            yang = HealthAdviceGenerator
            jianyi.click(yang.generate_health_advice, inputs=yangsheng, outputs=jie4)

    with gr.Tab("中药服务"):
        with gr.Column():
            gr.HTML(value="<p>根据用户相要了解的中药以提供图片和功效以及用法等。</p>")
            zhongyao = gr.Textbox(elem_id="chat-box", label="中药")
            gr.Examples(label='常见问题示例', examples=[["Angelica sinensis"], ["Astragalus membranaceus"], ["Codonopsis pilosula"]], inputs=zhongyao)
            zhongyaoshengcheng = gr.Button("生成中药信息")
            with gr.Row():
                jie6 = gr.Textbox(label="中药信息结果", lines=8)
                image = gr.Image()
            zhongyaoshengcheng.click(query, inputs=zhongyao, outputs=image)
            zhongyaoshengcheng.click(zhongcaoyao, inputs=zhongyao, outputs=jie6)
            
    with gr.Tab("心率检测服务"):
        # 使用Gradio Video组件播放视频
        with gr.Row():
            input_video = gr.Video(sources=["upload","webcam"], label="需要检测的视频")
            button_xinli = gr.Button("开始检测")
            button_xinli.click(display_video, inputs=input_video, outputs=gr.Video())   
                
demo.launch(show_api=False)
demo.queue()