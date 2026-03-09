import scrcpy
import cv2
import numpy as np
import threading
from queue import Queue
import time

class ScrcpyFrameGrabber:
    """适配 scrcpy-python 的帧抓取器"""
    
    def __init__(self, device_serial=None, max_fps=30, max_size=1280, bitrate=8000000):
        self.device_serial = device_serial
        self.max_fps = max_fps
        self.max_size = max_size
        self.bitrate = bitrate
        self.frame_queue = Queue(maxsize=2)
        self.client = None
        self.running = False
        self.thread = None
        self.resolution = None  # 将在连接后更新

    def on_frame(self, frame: np.ndarray):
        """回调函数：收到新画面"""
        if frame is not None:
            # 如果还没记录分辨率，现在记录
            if self.resolution is None and frame.shape[1] > 0 and frame.shape[0] > 0:
                self.resolution = (frame.shape[1], frame.shape[0])
                print(f"[scrcpy] 设备分辨率: {self.resolution}")
            
            # 更新队列
            if self.frame_queue.full():
                try: 
                    self.frame_queue.get_nowait()
                except: 
                    pass
            self.frame_queue.put(frame)

    def start(self):
        """启动 scrcpy 客户端"""
        try:
            # 初始化客户端
            self.client = scrcpy.Client(
                device=self.device_serial,
                max_size=self.max_size,
                max_fps=self.max_fps,
                bitrate=self.bitrate,
                # 锁定方向，保持画面稳定
                lock_orientation=0,  # 0=原始方向
            )
            
            # 绑定回调
            self.client.add_listener(scrcpy.EVENT_FRAME, self.on_frame)
            
            # 在子线程启动，不阻塞主逻辑
            self.running = True
            self.thread = threading.Thread(target=self._run_client, daemon=True)
            self.thread.start()
            
            # 等待一下确保连接建立
            time.sleep(2)
            print(f"[scrcpy] 启动成功，设备: {self.device_serial or '默认'}")
            return True
            
        except Exception as e:
            print(f"[scrcpy] 启动失败: {e}")
            return False

    def _run_client(self):
        """运行客户端的包装方法"""
        try:
            self.client.start()
        except Exception as e:
            print(f"[scrcpy] 客户端运行出错: {e}")
            self.running = False

    def get_frame(self):
        """获取最新帧"""
        if self.frame_queue.empty():
            return None
        return self.frame_queue.get()

    def get_frame_timeout(self, timeout=1.0):
        """带超时的获取帧"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except:
            return None

    def swipe(self, start_x, start_y, end_x, end_y, duration=0.02):
        """执行滑动操作（相对坐标 0-1）"""
        if not self.client or not self.client.resolution:
            print("[scrcpy] 客户端未就绪或无分辨率信息")
            return False
        
        try:
            # 转换为绝对坐标
            abs_start_x = int(start_x * self.client.resolution[0])
            abs_start_y = int(start_y * self.client.resolution[1])
            abs_end_x = int(end_x * self.client.resolution[0])
            abs_end_y = int(end_y * self.client.resolution[1])
            
            # 执行滑动
            self.client.control.swipe(
                abs_start_x, abs_start_y,
                abs_end_x, abs_end_y,
                duration
            )
            return True
        except Exception as e:
            print(f"[scrcpy] 滑动失败: {e}")
            return False

    def tap(self, x, y):
        """点击操作（相对坐标 0-1）"""
        if not self.client or not self.client.resolution:
            return False
        
        try:
            abs_x = int(x * self.client.resolution[0])
            abs_y = int(y * self.client.resolution[1])
            self.client.control.touch(abs_x, abs_y, scrcpy.ACTION_DOWN)
            time.sleep(0.01)
            self.client.control.touch(abs_x, abs_y, scrcpy.ACTION_UP)
            return True
        except Exception as e:
            print(f"[scrcpy] 点击失败: {e}")
            return False

    def stop(self):
        """停止抓取"""
        self.running = False
        if self.client:
            self.client.stop()
            self.client = None
        if self.thread:
            self.thread.join(timeout=2.0)