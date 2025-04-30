import requests
import time
import threading
import logging
from typing import Dict, Any, Optional, Tuple, Union

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_handler")

DEFAULT = None

class RateLimitWaitingException(Exception):
    """当线程在等待全局速率限制解除时使用的专门异常类"""
    pass

# --- 1. 速率限制器类 ---
class ApiRateLimiter:
    """
    封装API调用速率控制和429错误处理的类。
    
    包含两个主要机制：
    1. 调用间隔控制：确保相邻API调用之间有最小时间间隔，防止频率过高
    2. 全局速率限制：当遇到429错误时，阻塞所有线程，只允许一个重试者线程继续尝试
    
    这种设计可以有效减少API服务的负载并处理服务方的速率限制错误。
    """
    def __init__(self):
        """初始化速率限制器，设置内部状态和锁"""
        # 调用间隔控制相关属性
        self.last_call_time = 0
        self.interval_lock = threading.Lock()

        # 全局速率限制控制相关属性
        self.rate_limit_event = threading.Event()  # 设置(set)状态表示无限制，可以调用
        self.rate_limit_event.set()  # 初始状态：允许调用
        self.retrier_lock = threading.RLock()  # 保护重试者ID的读写
        self.retrier_id = None  # 当前负责429重试的线程ID
        
        # 超时解除限制的锁，防止多线程同时尝试解除限制
        self.timeout_lock = threading.Lock()
        
        # 指数退避策略参数
        self.use_exponential_backoff = False  # 默认不使用指数级退避
        self.base_delay = 1.0  # 基础延迟时间（秒）
        self.max_delay = 60.0  # 最大延迟时间（秒）
        self.current_backoff = self.base_delay  # 当前退避时间

    def wait_for_rate_limit_release(self, verbose: bool = False, timeout: float = 300) -> bool:
        """
        等待全局速率限制解除，带有超时机制。
        
        当多个线程遇到速率限制(429)错误时，只有一个线程(重试者)负责重试，
        其他线程则调用此方法等待重试者解除限制。此方法支持超时，
        防止线程无限期等待。

        参数:
            verbose: 是否输出详细日志
            timeout: 最长等待时间（秒），默认5分钟

        返回:
            bool: True表示限制成功解除，False表示等待超时
        """
        if self.is_rate_limited():
            thread_name = threading.current_thread().name
            if verbose:
                logger.info(f"线程 {thread_name} - 检测到全局速率限制，进入等待(最长{timeout}秒)...")

            # 使用wait方法的timeout参数
            result = self.rate_limit_event.wait(timeout)

            if result:
                if verbose:
                    logger.info(f"线程 {thread_name} - 全局速率限制解除，继续执行。")
                return True
            else:
                # 使用超时锁确保只有一个线程执行超时解除操作
                if self.timeout_lock.acquire(blocking=False):
                    try:
                        # 如果仍然处于限制状态（可能在获取锁的过程中被其他线程解除了）
                        if self.is_rate_limited():
                            logger.warning(f"线程 {thread_name} - 等待速率限制解除超时({timeout}秒)，强制解除全局限制...")
                            with self.retrier_lock:
                                self.retrier_id = None
                                self.rate_limit_event.set()
                                # 重置退避时间
                                self.current_backoff = self.base_delay
                    finally:
                        self.timeout_lock.release()
                else:
                    # 其他线程正在处理超时，等待它完成
                    logger.info(f"线程 {thread_name} - 等待另一线程处理超时解除...")
                    # 再次等待，但这次时间较短
                    result = self.rate_limit_event.wait(5)
                    if not result:
                        logger.warning(f"线程 {thread_name} - 二次等待超时，视为限制已解除")
                
                # 无论如何超时后都认为可以继续
                return True
        return True

    def wait_for_interval(self, interval: float, verbose: bool = False) -> None:
        """
        等待满足两次API调用之间的最小时间间隔要求。
        
        使用锁机制确保多线程环境下对最后调用时间的安全访问和更新。
        根据上次调用时间计算需要等待的时间，确保API调用的频率不会过高。

        参数:
            interval: 两次调用之间的最小间隔时间（秒）
            verbose: 是否输出详细日志
        """
        with self.interval_lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time

            if time_since_last_call < interval:
                wait_time = interval - time_since_last_call
                if verbose:
                    logger.info(f"线程 {threading.current_thread().name} - 等待 {wait_time:.2f} 秒以满足 API 调用间隔...")
                time.sleep(wait_time)

            # 更新最后调用时间
            self.last_call_time = time.time()

    def is_rate_limited(self) -> bool:
        """
        检查全局速率限制是否已激活。
        
        返回:
            bool: True表示当前处于速率限制状态，False表示无限制
        """
        return not self.rate_limit_event.is_set()

    def get_retrier_id(self) -> Optional[int]:
        """
        获取当前负责429重试的线程ID。
        
        返回:
            Optional[int]: 如果有重试者则返回其线程ID，否则返回None
        """
        with self.retrier_lock:
            return self.retrier_id

    def try_become_retrier(self, thread_id: int) -> bool:
        """
        尝试将当前线程设置为429错误的重试者。
        
        当遇到429错误时调用此方法，线程会尝试成为负责重试的线程。
        只有当全局限制未激活时才能成为新的重试者，或者线程已经是当前的重试者。
        这种机制确保同一时间只有一个线程负责重试，避免多线程同时重试导致更多的429错误。

        参数:
            thread_id: 当前线程ID
            
        返回:
            bool: True表示(1)成功成为重试者或(2)已经是重试者，False表示其他线程是重试者
        """
        with self.retrier_lock:
            # 如果限制未激活，则激活并成为重试者
            if not self.is_rate_limited():
                self.rate_limit_event.clear()  # 激活限制
                self.retrier_id = thread_id
                thread_name = threading.current_thread().name
                logger.warning(f"线程 {thread_name} - 收到速率限制错误(429)，设置为负责重试的线程，全局限制已激活")
                # 重置退避时间
                self.current_backoff = self.base_delay
                return True
            else:
                # 如果已激活，仅当自己是重试者时才返回True
                is_retrier = (self.retrier_id == thread_id)
                thread_name = threading.current_thread().name
                if is_retrier:
                    logger.info(f"线程 {thread_name} - 已是负责429重试的线程")
                else:
                    logger.info(f"线程 {thread_name} - 429错误，但另一线程({self.retrier_id})负责重试")
                return is_retrier

    def release_rate_limit(self, thread_id: int) -> bool:
        """
        解除全局速率限制，但只有当前重试者才能执行此操作。
        
        当重试成功后，重试者线程应调用此方法解除全局限制，
        允许其他等待的线程继续发送请求。

        参数:
            thread_id: 当前线程ID
            
        返回:
            bool: True表示成功解除限制，False表示非重试者无权解除
        """
        with self.retrier_lock:
            if self.is_rate_limited() and self.retrier_id == thread_id:
                thread_name = threading.current_thread().name
                logger.info(f"线程 {thread_name} (重试者) - 解除速率限制")
                self.retrier_id = None
                self.rate_limit_event.set()
                # 重置退避时间
                self.current_backoff = self.base_delay
                return True
            return False
    
    def get_backoff_time(self, retry_delay: float = None) -> float:
        """
        获取当前的退避时间，并计算下一次的退避时间。
        
        如果启用了指数退避策略，每次调用后退避时间会按指数增长，
        但不超过设定的最大值。如果未启用，则使用固定的重试延迟。
        
        参数:
            retry_delay: 当未使用指数退避时的固定延迟时间，若未提供则使用基础延迟
            
        返回:
            float: 当前应使用的退避时间（秒）
        """
        with self.retrier_lock:
            if not self.use_exponential_backoff:
                # 不使用指数退避，返回固定延迟
                return retry_delay if retry_delay is not None else self.base_delay
            
            # 使用指数退避策略
            current = self.current_backoff
            # 计算下一次退避时间（指数增长）
            next_backoff = min(self.current_backoff * 2, self.max_delay)
            self.current_backoff = next_backoff
            return current
            
    def enable_exponential_backoff(self, enabled: bool = True) -> None:
        """
        启用或禁用指数级退避策略。
        
        参数:
            enabled: True表示启用指数级退避，False表示使用固定延迟
        """
        with self.retrier_lock:
            self.use_exponential_backoff = enabled
            # 重置当前退避时间
            self.current_backoff = self.base_delay

# 创建全局速率限制器实例
rate_limiter = ApiRateLimiter()

# --- 2. API请求执行函数 ---
def execute_api_request(api_url: str, payload: Dict, headers: Dict, timeout: int = 300,
                        verbose: bool = False) -> requests.Response:
    """
    执行单次API POST请求并处理HTTP错误。
    
    封装了HTTP请求的发送过程，统一处理请求参数和响应处理。

    参数:
        api_url: API端点URL
        payload: 请求体内容
        headers: 请求头
        timeout: 请求超时时间（秒），默认5分钟
        verbose: 是否输出详细日志

    返回:
        Response对象

    抛出:
        requests.exceptions.HTTPError: 服务器返回4xx或5xx状态码
        requests.exceptions.RequestException: 其他网络或请求相关错误
    """
    thread_name = threading.current_thread().name

    if verbose:
        logger.info(f"线程 {thread_name} - 尝试发起 API 请求...")

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()  # 如果状态码表示错误则抛出异常
    except Exception as e:
        if verbose:
            logger.error(f"线程 {thread_name} - API 请求失败: {str(e)}")
        raise  # 重新抛出异常，由调用者处理
    
    if verbose:
        logger.info(f"线程 {thread_name} - API 请求成功")

    return response

# --- 3. 429错误处理函数 ---
def handle_429_error(thread_id: int, retry_delay: float, verbose: bool = False) -> bool:
    """
    处理429（速率限制）错误，决定当前线程是成为重试者还是等待解除。
    
    当遇到429错误时，线程会尝试成为重试者或等待当前重试者解决问题。
    如果启用了指数退避策略，则使用动态增长的延迟时间；否则使用固定延迟。

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
        # 获取当前退避时间
        backoff_time = rate_limiter.get_backoff_time(retry_delay)
        # 当前线程是重试者，等待后重试
        backoff_type = "指数退避" if rate_limiter.use_exponential_backoff else "固定延迟"
        logger.warning(f"线程 {thread_name} (重试者) - 等待 {backoff_time:.2f} 秒后重试 (429 {backoff_type})...")
        time.sleep(backoff_time)
        return True
    else:
        # 当前线程不是重试者，等待限制解除
        if verbose:
            logger.info(f"线程 {thread_name} - 遭遇429错误，但另一线程正在处理重试，进入等待...")
        wait_result = rate_limiter.wait_for_rate_limit_release(verbose)
        if verbose:
            logger.info(f"线程 {thread_name} - 全局速率限制解除 (429等待后)，重新检查。等待结果: {wait_result}")
        return False

# --- 4. 配置解析函数 ---
def prepare_api_request(prompt: str, config: Dict[str, Any], default_config: Dict[str, Any]) -> Tuple[Dict, Dict, str]:
    """
    准备API请求的有效载荷和请求头。
    
    根据提供的配置和默认配置，构建API请求所需的参数。
    处理各种可能的配置项，确保请求格式正确。

    参数:
        prompt: 发送给LLM的提示
        config: 用户提供的配置字典
        default_config: 默认配置字典，在用户配置缺失时使用

    返回:
        (payload, headers, api_url): 请求体、请求头和API URL的元组
    """
    # 构建基本请求体
    payload = {
        "messages": [{"role": "user", "content": prompt}],
    }

    # 动态添加API参数配置项
    for param in ["model", "stream", "max_tokens", "temperature", "top_p", "top_k",
                  "frequency_penalty", "n", "response_format", "stop"]:
        value = config.get(param, DEFAULT)
        if value is not DEFAULT:
            payload[param] = value
    
    # 特殊处理extra_body，如果是字典则合并到payload中
    extra_body = config.get("extra_body", DEFAULT)
    if extra_body is not DEFAULT:
        if isinstance(extra_body, dict):
            payload.update(extra_body)
        else:
            payload["extra_body"] = extra_body

    # 构建请求头
    headers = {
        "Authorization": f"Bearer {config.get('api_token')}",
        "Content-Type": "application/json"
    }

    # 添加任何额外的请求头
    extra_headers = config.get("extra_headers", {})
    if extra_headers and isinstance(extra_headers, dict):
        headers.update(extra_headers)

    # 获取API URL
    api_url = config.get("api_url", default_config["api_url"])

    return payload, headers, api_url

# --- 5. 主API调用函数 ---
def call_llm_api(prompt: str, config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    调用LLM API的主函数，包含重试逻辑和429速率限制处理。
    
    作为API调用的主入口点，处理整个调用流程，包括配置准备、
    请求发送、错误处理和重试逻辑。实现了自适应的重试机制，
    能够有效处理各类错误情况，特别是429速率限制错误。
    
    参数:
        prompt: 发送给LLM的提示文本
        config: 用户提供的配置字典，控制API的行为
        default_config: 默认配置字典，在用户配置缺失时使用
        
    返回:
        Dict[str, Any]: 成功时返回{'success': True, 'data': 响应数据}
                       失败时返回{'success': False, 'error': 错误信息, 'error_code': 错误码}
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
                    logger.info(f"线程 {threading.current_thread().name} - 全局速率限制已激活，且本线程不是重试者，等待解除...")
                # 等待全局限制解除，添加5分钟超时
                wait_success = rate_limiter.wait_for_rate_limit_release(verbose, timeout=300)
                if not wait_success:
                    logger.warning(f"线程 {threading.current_thread().name} - 等待超时，将作为新线程继续...")
                # 重要：解除后继续循环，重新检查所有条件
                continue

            # 3. 执行API请求
            response = None
            try:
                if verbose:
                    logger.info(f"线程 {threading.current_thread().name} - 准备发送请求...")
                # 再次确认速率限制状态（双重检查）
                if rate_limiter.is_rate_limited() and current_thread_id != rate_limiter.get_retrier_id():
                    if verbose:
                        logger.info(f"线程 {threading.current_thread().name} - 双重检查：速率限制已激活，且本线程不是重试者，跳过请求...")
                    continue  # 立即跳过此次请求尝试

                # 发送API请求
                response = execute_api_request(api_url, payload, headers, timeout=600, verbose=verbose)

                # 请求成功，如果当前线程是429重试者，解除限制
                if is_429_retrier:
                    if verbose:
                        logger.info(f"线程 {threading.current_thread().name} (重试者) - 请求成功，解除全局速率限制...")
                    rate_limiter.release_rate_limit(current_thread_id)
                    is_429_retrier = False

                # 返回成功结果
                return {'success': True, 'data': response.json()}

            except requests.exceptions.HTTPError as e:
                # 处理HTTP错误
                status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else -1
                error_body_message = ""
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_body = e.response.json()
                        if isinstance(error_body, dict) and 'message' in error_body:
                            error_body_message = f" (API Message: {error_body['message']})"
                    except ValueError: # JSONDecodeError is a subclass of ValueError
                        error_body_message = f" (API Response Body: {e.response.text[:200]})" # Log first 200 chars if not JSON

                if status_code == 429:
                    # 处理429错误
                    was_retrier = handle_429_error(current_thread_id, retry_delay, verbose)
                    if was_retrier:
                        # 标记当前线程为此次调用的429重试者
                        is_429_retrier = True
                        if verbose:
                            logger.info(f"线程 {threading.current_thread().name} - 已成为重试者，将继续尝试...")
                    else:
                        if verbose:
                            logger.info(f"线程 {threading.current_thread().name} - 不是重试者，已等待限制解除，重新循环检查...")
                    continue  # 无论是否为重试者，都重新开始循环
                else:
                    # 处理其他HTTP错误
                    error_msg = f"API调用失败 (HTTP {status_code}): {str(e)}{error_body_message}"
                    logger.error(error_msg)
                    attempt += 1
                    result = {'success': False, 'error': error_msg, 'error_code': status_code}
                    # 将在循环末尾重试或退出

            except requests.exceptions.RequestException as e:
                # 处理网络或请求异常
                error_msg = f"API网络或请求失败 (尝试 {attempt+1}/{max_retries}): {str(e)}"
                logger.error(error_msg)
                attempt += 1
                result = {'success': False, 'error': error_msg, 'error_code': -1}
                # 将在循环末尾重试或退出

            except Exception as e:
                # 处理其他意外异常
                error_msg = f"API调用中发生意外错误 (尝试 {attempt+1}/{max_retries}): {str(e)}"
                logger.error(error_msg)
                attempt += 1
                result = {'success': False, 'error': error_msg, 'error_code': -2}
                # 将在循环末尾重试或退出

            # 4. 检查是否达到最大重试次数（对于非429错误）
            if attempt >= max_retries:
                error_msg = f"达到最大重试次数 ({max_retries})，放弃请求 {prompt[:50]}..."
                logger.error(error_msg)

                # 确保有完整的错误信息
                if result is None:
                    result = {'success': False, 'error': error_msg, 'error_code': -3}
                elif 'error' not in result or not result['error']:
                    result['error'] = error_msg

                # 返回错误结果，finally块会处理清理
                return result

            # 5. 非429错误的重试等待
            logger.info(f"线程 {threading.current_thread().name} - {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
            # 循环继续

    finally:
        # 6. 清理逻辑：如果当前线程是429重试者，但函数即将退出，需要释放全局限制
        if is_429_retrier and rate_limiter.is_rate_limited():
            thread_name = threading.current_thread().name
            logger.info(f"线程 {thread_name} (重试者) - 在finally块中解除速率限制")
            rate_limiter.release_rate_limit(current_thread_id)

def call_llm_api_with_openai_sdk(prompt: str, config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用OpenAI SDK调用LLM API的实验性函数。
    
    支持流式(stream)和非流式两种模式，在流式模式下，将累积所有的流式响应后返回。
    使用标准的OpenAI SDK错误处理方式，并与现有的速率限制机制兼容。
    
    参数:
        prompt: 发送给LLM的提示文本
        config: 用户提供的配置字典，控制API的行为
        default_config: 默认配置字典，在用户配置缺失时使用
        
    返回:
        Dict[str, Any]: 成功时返回{'success': True, 'data': 响应数据}
                       失败时返回{'success': False, 'error': 错误信息, 'error_code': 错误码}
    """
    # 尝试导入OpenAI SDK
    try:
        from openai import OpenAI, OpenAIError, RateLimitError, APITimeoutError, APIConnectionError
    except ImportError:
        error_msg = "无法导入OpenAI SDK。请确保已通过 'pip install openai' 安装了OpenAI SDK。"
        logger.error(error_msg)
        return {'success': False, 'error': error_msg, 'error_code': -999}
    
    # 获取配置
    max_retries = config.get("max_retries", default_config["max_retries"])
    retry_delay = config.get("retry_delay", default_config["retry_delay"])
    api_call_interval = config.get("api_call_interval", default_config["api_call_interval"])
    verbose = config.get("verbose", default_config["verbose"])
    
    # 获取API令牌
    api_token = config.get("api_token")
    
    # 获取是否使用流式模式
    use_streaming = config.get("stream", default_config.get("stream", False))
    
    # 准备API URL (base_url)
    api_url = config.get("api_url", default_config["api_url"])
    
    # 线程标识
    current_thread_id = threading.get_ident()
    thread_name = threading.current_thread().name
    # 标记此次调用中当前线程是否为429重试者
    is_429_retrier = False
    
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
                    logger.info(f"线程 {thread_name} - 全局速率限制已激活，且本线程不是重试者，等待解除...")
                # 等待全局限制解除，添加5分钟超时
                wait_success = rate_limiter.wait_for_rate_limit_release(verbose, timeout=300)
                if not wait_success:
                    logger.warning(f"线程 {thread_name} - 等待超时，将作为新线程继续...")
                # 重要：解除后继续循环，重新检查所有条件
                continue
            
            # 3. 执行API请求
            try:
                if verbose:
                    logger.info(f"线程 {thread_name} - 准备发送OpenAI SDK请求...")
                
                # 再次确认速率限制状态（双重检查）
                if rate_limiter.is_rate_limited() and current_thread_id != rate_limiter.get_retrier_id():
                    if verbose:
                        logger.info(f"线程 {thread_name} - 双重检查：速率限制已激活，且本线程不是重试者，跳过请求...")
                    continue  # 立即跳过此次请求尝试
                
                # 初始化OpenAI客户端
                client = OpenAI(
                    api_key=api_token,
                    base_url=api_url
                )
                
                # 准备消息
                messages = [{"role": "user", "content": prompt}]
                
                # 准备API参数
                kwargs = {}
                # 添加模型参数
                if config.get("model") is not DEFAULT:
                    kwargs["model"] = config.get("model")
                
                # 添加其他参数
                for param in ["max_tokens", "temperature", "top_p", "frequency_penalty", "n"]:
                    if config.get(param) is not DEFAULT:
                        kwargs[param] = config.get(param)
                
                # 特殊处理响应格式
                if config.get("response_format") is not DEFAULT:
                    kwargs["response_format"] = config.get("response_format")
                
                # 特殊处理停止词
                if config.get("stop") is not DEFAULT:
                    kwargs["stop"] = config.get("stop")
                
                # 处理extra_body
                extra_body = config.get("extra_body", DEFAULT)
                if extra_body is not DEFAULT:
                    if isinstance(extra_body, dict):
                        # 将extra_body合并到kwargs中
                        kwargs.update(extra_body)
                
                # 执行API调用
                if use_streaming:
                    # 流式模式
                    if verbose:
                        logger.info(f"线程 {thread_name} - 使用流式模式发送请求")
                    
                    # 创建流式请求
                    stream = client.chat.completions.create(
                        messages=messages,
                        stream=True,
                        **kwargs
                    )
                    
                    # 收集流式响应
                    collected_content = ""
                    collected_reasoning = ""
                    last_usage = None
                    
                    # 处理流式响应
                    for chunk in stream:
                        if chunk.choices and len(chunk.choices) > 0:
                            # 处理常规content
                            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                                content_piece = chunk.choices[0].delta.content
                                collected_content += content_piece
                                if verbose and len(collected_content) % 100 == 0:
                                    logger.debug(f"线程 {thread_name} - 已收集 {len(collected_content)} 字符的流式响应")
                            
                            # 处理reasoning_content（如果存在）
                            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content is not None:
                                reasoning_piece = chunk.choices[0].delta.reasoning_content
                                collected_reasoning += reasoning_piece
                                if verbose and len(collected_reasoning) % 100 == 0:
                                    logger.debug(f"线程 {thread_name} - 已收集 {len(collected_reasoning)} 字符的推理内容")
                        
                        # 更新使用情况统计信息
                        if hasattr(chunk, 'usage'):
                            last_usage = chunk.usage
                    
                    # 创建完整的响应对象
                    response_data = {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": collected_content,
                                    "reasoning_content": collected_reasoning  # 添加收集到的reasoning内容
                                },
                                "index": 0,
                                "finish_reason": "stop"  # 假设正常结束
                            }
                        ],
                        "usage": {
                            "prompt_tokens": last_usage.prompt_tokens if last_usage and hasattr(last_usage, 'prompt_tokens') else 0,
                            "completion_tokens": last_usage.completion_tokens if last_usage and hasattr(last_usage, 'completion_tokens') else 0,
                            "total_tokens": last_usage.total_tokens if last_usage and hasattr(last_usage, 'total_tokens') else 0
                        }
                    }
                    
                    # 添加completion_tokens_details（如果存在）
                    if last_usage and hasattr(last_usage, 'completion_tokens_details'):
                        details = {}
                        # 直接访问属性而不是使用.get()方法
                        if hasattr(last_usage.completion_tokens_details, 'reasoning_tokens'):
                            details['reasoning_tokens'] = last_usage.completion_tokens_details.reasoning_tokens
                        response_data["usage"]["completion_tokens_details"] = details
                else:
                    # 非流式模式
                    if verbose:
                        logger.info(f"线程 {thread_name} - 使用非流式模式发送请求")
                    
                    # 直接发送请求
                    response = client.chat.completions.create(
                        messages=messages,
                        **kwargs
                    )
                    
                    # 将OpenAI响应对象转换为字典
                    response_data = {
                        "choices": [
                            {
                                "message": {
                                    "role": response.choices[0].message.role,
                                    "content": response.choices[0].message.content
                                },
                                "index": response.choices[0].index,
                                "finish_reason": response.choices[0].finish_reason
                            }
                        ],
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    }
                
                # 请求成功，如果当前线程是429重试者，解除限制
                if is_429_retrier:
                    if verbose:
                        logger.info(f"线程 {thread_name} (重试者) - 请求成功，解除全局速率限制...")
                    rate_limiter.release_rate_limit(current_thread_id)
                    is_429_retrier = False
                
                # 返回成功结果
                return {'success': True, 'data': response_data}
                
            except RateLimitError as e:
                # 处理429错误
                error_msg = f"OpenAI SDK速率限制错误 (429): {str(e)}"
                logger.warning(error_msg)
                
                # 处理429错误
                was_retrier = handle_429_error(current_thread_id, retry_delay, verbose)
                if was_retrier:
                    # 标记当前线程为此次调用的429重试者
                    is_429_retrier = True
                    if verbose:
                        logger.info(f"线程 {thread_name} - 已成为重试者，将继续尝试...")
                else:
                    if verbose:
                        logger.info(f"线程 {thread_name} - 不是重试者，已等待限制解除，重新循环检查...")
                continue  # 无论是否为重试者，都重新开始循环
                
            except APITimeoutError as e:
                # 处理超时错误
                error_msg = f"OpenAI SDK请求超时: {str(e)}"
                logger.error(error_msg)
                attempt += 1
                result = {'success': False, 'error': error_msg, 'error_code': -4}  # 使用-4表示超时错误
                # 将在循环末尾重试或退出
                
            except APIConnectionError as e:
                # 处理连接错误
                error_msg = f"OpenAI SDK连接错误: {str(e)}"
                logger.error(error_msg)
                attempt += 1
                result = {'success': False, 'error': error_msg, 'error_code': -1}
                # 将在循环末尾重试或退出
                
            except OpenAIError as e:
                # 处理其他OpenAI错误
                error_msg = f"OpenAI SDK错误: {str(e)}"
                logger.error(error_msg)
                attempt += 1
                # 尝试获取错误代码（如果有）
                error_code = -2  # 默认错误代码
                if hasattr(e, 'status_code'):
                    error_code = e.status_code
                result = {'success': False, 'error': error_msg, 'error_code': error_code}
                # 将在循环末尾重试或退出
                
            except Exception as e:
                # 处理其他意外异常
                error_msg = f"OpenAI SDK调用中发生意外错误: {str(e)}"
                logger.error(error_msg)
                attempt += 1
                result = {'success': False, 'error': error_msg, 'error_code': -3}
                # 将在循环末尾重试或退出
            
            # 4. 检查是否达到最大重试次数（对于非429错误）
            if attempt >= max_retries:
                error_msg = f"达到最大重试次数 ({max_retries})，放弃请求 {prompt[:50]}..."
                logger.error(error_msg)
                
                # 确保有完整的错误信息
                if result is None:
                    result = {'success': False, 'error': error_msg, 'error_code': -5}
                elif 'error' not in result or not result['error']:
                    result['error'] = error_msg
                    
                # 返回错误结果，finally块会处理清理
                return result
            
            # 5. 非429错误的重试等待
            logger.info(f"线程 {thread_name} - {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
            # 循环继续
            
    finally:
        # 6. 清理逻辑：如果当前线程是429重试者，但函数即将退出，需要释放全局限制
        if is_429_retrier and rate_limiter.is_rate_limited():
            thread_name = threading.current_thread().name
            logger.info(f"线程 {thread_name} (重试者) - 在finally块中解除速率限制")
            rate_limiter.release_rate_limit(current_thread_id)
