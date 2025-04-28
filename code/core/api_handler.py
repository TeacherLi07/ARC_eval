import requests
import time
import threading
from typing import Dict, Any, Optional, Tuple

DEFAULT = None

class RateLimitWaitingException(Exception):
    """当线程在等待全局速率限制解除时使用的专门异常类"""
    pass

# --- 1. 速率限制器类 ---
class ApiRateLimiter:
    """
    封装API调用速率控制和429错误处理的类。
    包含两个机制：
    1. 调用间隔控制：确保相邻API调用之间有最小时间间隔
    2. 全局速率限制：当遇到429错误时，阻塞所有线程，只允许一个重试者继续尝试
    """
    def __init__(self):
        # 调用间隔控制
        self.last_call_time = 0
        self.interval_lock = threading.Lock()

        # 全局速率限制控制 (429错误处理)
        self.rate_limit_event = threading.Event()  # 设置(set)状态表示无限制，可以调用
        self.rate_limit_event.set()  # 初始状态：允许调用
        self.retrier_lock = threading.RLock()  # 保护重试者ID的读写
        self.retrier_id = None  # 当前负责429重试的线程ID

    def wait_for_rate_limit_release(self, verbose: bool = False, timeout: float = 300) -> bool:
        """
        等待全局速率限制解除，带有超时机制。

        参数:
            verbose: 是否输出详细日志
            timeout: 最长等待时间（秒），默认5分钟

        返回:
            bool: True表示限制成功解除，False表示等待超时
        """
        if self.is_rate_limited():
            thread_name = threading.current_thread().name
            if verbose:
                print(f"线程 {thread_name} - 检测到全局速率限制，进入等待(最长{timeout}秒)...")

            # 使用wait方法的timeout参数
            result = self.rate_limit_event.wait(timeout)

            if result:
                if verbose:
                    print(f"线程 {thread_name} - 全局速率限制解除，继续执行。")
                return True
            else:
                # 超时情况下，强制解除限制
                print(f"线程 {thread_name} - 等待速率限制解除超时({timeout}秒)，强制解除全局限制...")
                with self.retrier_lock:
                    self.retrier_id = None
                    self.rate_limit_event.set()
                return False
        return True

    def wait_for_interval(self, interval: float, verbose: bool = False) -> None:
        """等待满足两次API调用之间的最小时间间隔要求"""
        with self.interval_lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time

            if time_since_last_call < interval:
                wait_time = interval - time_since_last_call
                if verbose:
                    print(f"线程 {threading.current_thread().name} - 等待 {wait_time:.2f} 秒以满足 API 调用间隔...")
                time.sleep(wait_time)

            # 更新最后调用时间
            self.last_call_time = time.time()

    def is_rate_limited(self) -> bool:
        """检查全局速率限制是否已激活"""
        return not self.rate_limit_event.is_set()

    def get_retrier_id(self) -> Optional[int]:
        """获取当前负责429重试的线程ID"""
        with self.retrier_lock:
            return self.retrier_id

    def try_become_retrier(self, thread_id: int) -> bool:
        """
        尝试将当前线程设置为429错误的重试者。
        只有当全局限制未激活时才能成为新的重试者。

        返回:
            bool: True表示(1)成功成为重试者或(2)已经是重试者，False表示其他线程是重试者
        """
        with self.retrier_lock:
            # 如果限制未激活，则激活并成为重试者
            if not self.is_rate_limited():
                self.rate_limit_event.clear()  # 激活限制
                self.retrier_id = thread_id
                thread_name = threading.current_thread().name
                print(f"线程 {thread_name} - 收到速率限制错误(429)，设置为负责重试的线程，全局限制已激活。")
                return True
            else:
                # 如果已激活，仅当自己是重试者时才返回True
                is_retrier = (self.retrier_id == thread_id)
                if is_retrier:
                    thread_name = threading.current_thread().name
                    print(f"线程 {thread_name} - 已是负责429重试的线程")
                else:
                    thread_name = threading.current_thread().name
                    print(f"线程 {thread_name} - 429错误，但另一线程({self.retrier_id})负责重试")
                return is_retrier

    def release_rate_limit(self, thread_id: int) -> bool:
        """
        解除全局速率限制，但只有当前重试者才能执行此操作。

        返回:
            bool: True表示成功解除限制，False表示非重试者无权解除
        """
        with self.retrier_lock:
            if self.is_rate_limited() and self.retrier_id == thread_id:
                thread_name = threading.current_thread().name
                print(f"线程 {thread_name} (重试者) - 解除速率限制")
                self.retrier_id = None
                self.rate_limit_event.set()
                return True
            return False

# 创建全局速率限制器实例
rate_limiter = ApiRateLimiter()

# --- 2. API请求执行函数 ---
def execute_api_request(api_url: str, payload: Dict, headers: Dict, timeout: int = 600,
                        verbose: bool = False) -> requests.Response:
    """
    执行单次API POST请求并处理HTTP错误。

    参数:
        api_url: API端点URL
        payload: 请求体内容
        headers: 请求头
        timeout: 请求超时时间（秒）
        verbose: 是否输出详细日志

    返回:
        Response对象

    抛出:
        requests.exceptions.HTTPError: 服务器返回4xx或5xx状态码
        requests.exceptions.RequestException: 其他网络或请求相关错误
    """
    thread_name = threading.current_thread().name

    if verbose:
        print(f"线程 {thread_name} - 尝试发起 API 请求...")

    response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()  # 如果状态码表示错误则抛出异常

    if verbose:
        print(f"线程 {thread_name} - API 请求成功")

    return response

# --- 3. 429错误处理函数 ---
def handle_429_error(thread_id: int, retry_delay: float, verbose: bool = False) -> bool:
    """
    处理429（速率限制）错误，决定当前线程是成为重试者还是等待解除。

    参数:
        thread_id: 当前线程ID
        retry_delay: 重试延迟时间（秒）
        verbose: 是否输出详细日志

    返回:
        bool: True表示当前线程是重试者，False表示当前线程已等待限制解除
    """
    thread_name = threading.current_thread().name
    # 尝试成为重试者
    is_retrier = rate_limiter.try_become_retrier(thread_id)

    if is_retrier:
        # 当前线程是重试者，等待后重试
        print(f"线程 {thread_name} (重试者) - 等待 {retry_delay} 秒后重试 (429)...")
        time.sleep(retry_delay)
        return True
    else:
        # 当前线程不是重试者，等待限制解除
        if verbose:
            print(f"线程 {thread_name} - 遭遇429错误，但另一线程正在处理重试，进入等待...")
        rate_limiter.wait_for_rate_limit_release(verbose)
        if verbose:
            print(f"线程 {thread_name} - 全局速率限制解除 (429等待后)，重新检查。")
        return False

# --- 4. 配置解析函数 ---
def prepare_api_request(prompt: str, config: Dict[str, Any], default_config: Dict[str, Any]) -> Tuple[Dict, Dict, str]:
    """
    准备API请求的有效载荷和请求头。

    参数:
        prompt: 发送给LLM的提示
        config: 配置字典
        default_config: 默认配置字典

    返回:
        (payload, headers, api_url): 请求体、请求头和API URL
    """
    # 构建请求体
    payload = {
        "messages": [{"role": "user", "content": prompt}],
    }

    # 动态添加配置项
    for param in ["model", "stream", "max_tokens", "temperature", "top_p", "top_k",
                  "frequency_penalty", "n", "response_format", "extra_body", "stop"]:
        value = config.get(param, DEFAULT)
        if value is not DEFAULT:
            payload[param] = value

    # 构建请求头
    headers = {
        "Authorization": f"Bearer {config.get('api_token')}",
        "Content-Type": "application/json"
    }

    # 获取API URL
    api_url = config.get("api_url", default_config["api_url"])

    return payload, headers, api_url

# --- 5. 重构后的主函数 ---
def call_llm_api(prompt: str, config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    调用LLM API，包含重试逻辑和429速率限制处理（重构版）。

    参数:
        prompt: 发送给LLM的提示
        config: 配置字典
        default_config: 默认配置字典

    返回:
        Dict: 包含API响应或错误信息的字典
    """
    # 获取配置
    max_retries = config.get("max_retries", default_config["max_retries"])
    retry_delay = config.get("retry_delay", default_config["retry_delay"])
    api_call_interval = config.get("api_call_interval", default_config["api_call_interval"])
    verbose = config.get("verbose", default_config["verbose"])

    # 线程标识
    current_thread_id = threading.get_ident()
    # 标记此次调用中当前线程是否为429重试者
    is_429_retrier = False

    # 准备请求参数
    payload, headers, api_url = prepare_api_request(prompt, config, default_config)

    # 重试计数和错误结果暂存
    attempt = 0
    result = None

    # 使用try-finally确保资源清理
    try:
        # 主循环
        while True:
            # 1. 首先检查API调用间隔
            rate_limiter.wait_for_interval(api_call_interval, verbose)

            # 2. 然后检查全局速率限制状态
            current_retrier = rate_limiter.get_retrier_id()
            # 如果当前存在速率限制，并且当前线程不是重试者，则等待
            if rate_limiter.is_rate_limited() and current_thread_id != current_retrier:
                if verbose:
                    print(f"线程 {threading.current_thread().name} - 全局速率限制已激活，且本线程不是重试者，等待解除...")
                # 等待全局限制解除，添加5分钟超时
                wait_success = rate_limiter.wait_for_rate_limit_release(verbose, timeout=300)
                if not wait_success:
                    print(f"线程 {threading.current_thread().name} - 等待超时，将作为新线程继续...")
                # 重要：解除后继续循环，重新检查所有条件
                raise RateLimitWaitingException("速率限制等待后需要重试")

            # 3. 执行API请求
            response = None
            try:
                if verbose:
                    print(f"线程 {threading.current_thread().name} - 准备发送请求...")
                # 再次确认速率限制状态（双重检查）
                if rate_limiter.is_rate_limited() and current_thread_id != rate_limiter.get_retrier_id():
                    if verbose:
                        print(f"线程 {threading.current_thread().name} - 双重检查：速率限制已激活，且本线程不是重试者，跳过请求...")
                    continue  # 立即跳过此次请求尝试

                # 发送API请求
                response = execute_api_request(api_url, payload, headers, timeout=600, verbose=verbose)

                # 请求成功，如果当前线程是429重试者，解除限制
                if is_429_retrier:
                    if verbose:
                        print(f"线程 {threading.current_thread().name} (重试者) - 请求成功，解除全局速率限制...")
                    rate_limiter.release_rate_limit(current_thread_id)
                    is_429_retrier = False

                # 返回成功结果
                return {'success': True, 'data': response.json()}

            except requests.exceptions.HTTPError as e:
                # 处理HTTP错误
                if e.response.status_code == 429:
                    # 处理429错误
                    was_retrier = handle_429_error(current_thread_id, retry_delay, verbose)
                    if was_retrier:
                        # 标记当前线程为此次调用的429重试者
                        is_429_retrier = True
                        if verbose:
                            print(f"线程 {threading.current_thread().name} - 已成为重试者，将继续尝试...")
                    else:
                        if verbose:
                            print(f"线程 {threading.current_thread().name} - 不是重试者，已等待限制解除，重新循环检查...")
                    continue  # 无论是否为重试者，都重新开始循环
                else:
                    # 处理其他HTTP错误
                    error_code = response.status_code if response is not None else -10
                    error_msg = f"API调用失败 (HTTP {error_code}): {str(e)}"
                    print(error_msg)
                    attempt += 1
                    result = {'success': False, 'error': error_msg, 'error_code': error_code}
                    # 将在循环末尾重试或退出

            except requests.exceptions.RequestException as e:
                # 处理网络或请求异常
                error_msg = f"API网络或请求失败 (尝试 {attempt+1}/{max_retries}): {str(e)}"
                print(error_msg)
                attempt += 1
                result = {'success': False, 'error': error_msg, 'error_code': -1}
                # 将在循环末尾重试或退出

            except Exception as e:
                # 处理其他意外异常
                error_msg = f"API调用中发生意外错误 (尝试 {attempt+1}/{max_retries}): {str(e)}"
                print(error_msg)
                attempt += 1
                result = {'success': False, 'error': error_msg, 'error_code': -2}
                # 将在循环末尾重试或退出

            # 4. 检查是否达到最大重试次数（对于非429错误）
            if attempt >= max_retries:
                error_msg = f"达到最大重试次数 ({max_retries})，放弃请求 {prompt[:50]}..."
                print(error_msg)

                # 确保有完整的错误信息
                if result is None:
                    result = {'success': False, 'error': error_msg, 'error_code': -3}
                elif 'error' not in result or not result['error']:
                    result['error'] = error_msg

                # 返回错误结果，finally块会处理清理
                return result

            # 5. 非429错误的重试等待
            print(f"线程 {threading.current_thread().name} - {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
            # 循环继续

    finally:
        # 6. 清理逻辑：如果当前线程是429重试者，但函数即将退出，需要释放全局限制
        if is_429_retrier and rate_limiter.is_rate_limited():
            thread_name = threading.current_thread().name
            print(f"线程 {thread_name} (重试者) - 在finally块中解除速率限制")
            rate_limiter.release_rate_limit(current_thread_id)
