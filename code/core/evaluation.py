import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional
import functools

from .api_handler import call_llm_api, RateLimitWaitingException, DEFAULT
from .db_utils import save_results_to_db
from .utils import extract_output_from_response

print = functools.partial(print, flush=True)

# 默认配置，可以由调用者覆盖
DEFAULT_CONFIG = {
    # API配置
    "api_url": "https://api.siliconflow.cn/v1/chat/completions",
    "api_token": "",  # 调用时必须提供

    # 模型配置
    "model": DEFAULT, # Qwen/QwQ-32B
    "stream": DEFAULT, # False
    "max_tokens": DEFAULT, # 512
    "temperature": DEFAULT, # 0.7
    "top_p": DEFAULT, # 0.7
    "top_k": DEFAULT, # 50
    "frequency_penalty": DEFAULT, # 0.5
    "n": DEFAULT, # 1
    "response_format": {"type": "text"},
    "extra_body": DEFAULT,
    "stop": DEFAULT,

    # 运行配置
    "max_workers": 12,
    "save_interval": 1,
    "retry_delay": 5,
    "max_retries": 10,
    "api_call_interval": 1.0,
    "verbose": False,
    "problem_timeout": 1200, # 单个问题处理超时时间（秒）

    # 默认prompt
    "prompt": """你是一个解决抽象推理问题的AI助手。分析下面的训练示例，理解输入到输出的变换规则。
然后对测试输入应用相同的规则，生成测试输出。
请用JSON格式回答，只包含完整的测试输出"output":
""",
    "train_input_prompt": "样例输入",
    "train_output_prompt": "样例输出",
    "test_prompt": "测试输入：",
    "end_prompt": "仔细思考，以json字典形式回答，只包含完整的预测输出\"output\":",
}


def load_problems(problems_dir: str) -> List[Dict[str, Any]]:
    """
    加载所有问题文件
    返回一个包含所有问题文件的列表，每个文件是一个字典，包含文件名和内容
    """
    problems = []
    path = Path(problems_dir)

    for file_path in path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                problem = json.load(f)
                problems.append({
                    'filename': file_path.name,
                    'content': problem
                })
        except json.JSONDecodeError:
            print(f"错误：无法解析 {file_path}，跳过...")
        except Exception as e:
            print(f"加载 {file_path} 时发生错误: {e}，跳过...")

    print(f"已加载 {len(problems)} 个问题")
    return problems

def prepare_problem_prompt(problem: Dict[str, Any], config: Dict[str, Any]) -> str:
    """准备发送给LLM的提示"""
    train_data = problem['content']['train']
    test_data = problem['content']['test']
    # 从合并后的配置中获取prompt模板
    custom_prompt = config.get("prompt", DEFAULT_CONFIG["prompt"])
    train_input_prompt = config.get("train_input_prompt", DEFAULT_CONFIG["train_input_prompt"])
    train_output_prompt = config.get("train_output_prompt", DEFAULT_CONFIG["train_output_prompt"])
    test_prompt = config.get("test_prompt", DEFAULT_CONFIG["test_prompt"])
    end_prompt = config.get("end_prompt", DEFAULT_CONFIG["end_prompt"])

    # 使用自定义提示作为基础
    prompt = custom_prompt + "\n\n"

    # 添加训练样例
    for i, train_item in enumerate(train_data):
        prompt += f"{train_input_prompt}{i+1}：\n{str(train_item['input'])}\n"
        prompt += f"{train_output_prompt}{i+1}：\n{str(train_item['output'])}\n\n"

    # 添加测试输入 (假设每个问题只有一个测试用例)
    if test_data:
        prompt += f"{test_prompt}：\n{str(test_data[0]['input'])}\n\n"

    # 添加结束语
    prompt += f"{end_prompt}\n"

    return prompt


def evaluate_problem(problem: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """评估单个问题，增加整体超时控制，且区分处理速率限制等待和其它异常"""
    filename = problem['filename']
    verbose = config.get("verbose", False)
    overall_timeout = config.get("problem_timeout", DEFAULT_CONFIG["problem_timeout"])

    if verbose:
        print(f"处理 {filename}...")

    # 使用线程和事件实现超时控制
    result_container = []
    completion_event = threading.Event()

    def process_with_timeout():
        # 添加重试循环，允许在速率限制解除后重新尝试
        retry_count = 0
        max_retries = config.get("max_retries", DEFAULT_CONFIG["max_retries"])

        while retry_count < max_retries:
            try:
                prompt = prepare_problem_prompt(problem, config)

                # 关键修改：使用专门的标记捕获速率限制等待情况
                try:
                    response = call_llm_api(prompt, config, DEFAULT_CONFIG)
                except RateLimitWaitingException:
                    # 不是真正的错误，只是指示需要重试
                    if verbose:
                        print(f"线程 {threading.current_thread().name} - 速率限制等待后重试 {filename}")
                    retry_count += 1
                    continue

                # 处理API调用失败
                if not response.get('success'):
                    # 特殊处理速率限制相关的错误
                    error_code = response.get('error_code', 0)
                    if error_code == 429 or "rate limit" in response.get('error', '').lower():
                        if verbose:
                            print(f"线程 {threading.current_thread().name} - API返回速率限制错误，重试 {filename}")
                        retry_count += 1
                        time.sleep(config.get("retry_delay", DEFAULT_CONFIG["retry_delay"])) # 使用配置的延迟
                        continue

                    print(f"获取 {filename} 的响应失败: {response.get('error')}")
                    result_container.append({
                        'filename': filename,
                        'success': False,
                        'error': response.get('error', 'API调用失败'),
                        'error_code': error_code
                    })
                    completion_event.set() # 确保事件被设置
                    return

                # 正常处理结果...
                data = response['data']
                content = data['choices'][0]['message']['content']
                reasoning_content = data['choices'][0]['message'].get('reasoning_content', '') # 假设可能存在
                prompt_token_count = data['usage'].get('prompt_tokens', 0)
                completion_token_count = data['usage'].get('completion_tokens', 0)

                if verbose:
                    print(f"{filename} 的Tokens: {prompt_token_count} + {completion_token_count} = {prompt_token_count + completion_token_count}")

                predicted_output = extract_output_from_response(response)
                actual_output = problem['content']['test'][0]['output']  # 假设只使用第一个测试用例

                # 评估结果
                is_correct = False
                if predicted_output is not None: # 确保提取到了结果再比较
                    is_correct = predicted_output == actual_output
                else:
                    if verbose:
                        print(f"{filename} - 未能成功提取答案")


                if verbose:
                    print(f"{filename} - 判断结果: {'✓ 正确' if is_correct else '✗ 错误'}")


                result_dict = {
                    'filename': filename,
                    'success': True,
                    'correct': is_correct,
                    'predicted': predicted_output,
                    'actual': actual_output,
                    'reasoning': reasoning_content,
                    'response': content,
                    'prompt_tokens': prompt_token_count,
                    'completion_tokens': completion_token_count,
                    'error': '',
                    'error_code': 0
                }
                result_container.append(result_dict)
                break  # 成功处理，退出重试循环

            except Exception as e:
                # 区分速率限制相关异常和其它异常
                if "rate limit" in str(e).lower() or "429" in str(e):
                    if verbose:
                        print(f"线程 {threading.current_thread().name} - 捕获到速率限制异常，重试 {filename}")
                    retry_count += 1
                    time.sleep(config.get("retry_delay", DEFAULT_CONFIG["retry_delay"])) # 使用配置的延迟
                else:
                    # 真正的异常，记录并退出
                    print(f"处理 {filename} 时发生意外错误: {str(e)}")
                    result_container.append({
                        'filename': filename,
                        'success': False,
                        'error': f"处理时发生异常: {str(e)}",
                        'error_code': -100
                    })
                    break  # 退出重试循环

        # 检查是否达到最大重试次数
        if retry_count >= max_retries and not result_container:
            print(f"{filename} - 达到最大重试次数 ({max_retries})，放弃请求")
            result_container.append({
                'filename': filename,
                'success': False,
                'error': f"达到最大重试次数 ({max_retries})，放弃请求",
                'error_code': -102
            })

        # 无论成功与否，都设置完成事件
        completion_event.set()

    # 启动处理线程
    process_thread = threading.Thread(target=process_with_timeout)
    process_thread.daemon = True
    process_thread.start()

    # 等待完成或超时
    if completion_event.wait(timeout=overall_timeout):
        return result_container[0] if result_container else {
            'filename': filename,
            'success': False,
            'error': "未知错误：处理完成但没有结果",
            'error_code': -103
        }
    else:
        print(f"处理 {filename} 超时(>{overall_timeout}秒)，强制终止")
        return {
            'filename': filename,
            'success': False,
            'error': f"处理超时(>{overall_timeout}秒)",
            'error_code': -101
        }

def evaluate_problem_wrapper(args):
    """
    包装函数，用于多线程池处理一个问题
    """
    problem, config, problem_idx, total_problems, print_lock = args

    with print_lock:
        print(f"\n{'='*50}")
        print(f"评估问题 {problem_idx+1}/{total_problems}: {problem['filename']}")
        print(f"{'='*50}")

    try:
        result = evaluate_problem(problem, config)

        # 打印结果状态
        with print_lock:
            if result['success']:
                status = "✓ 正确" if result['correct'] else "✗ 错误"
            else:
                status = f"! 处理失败 (错误码: {result.get('error_code', 0)})"
            print(f"\n问题 {problem_idx+1}/{total_problems} - {problem['filename']} 结果: {status}")

        return result

    except Exception as e:
        error_msg = f"处理问题 {problem['filename']} 时发生顶层错误: {str(e)}"
        with print_lock:
            print(error_msg)

        return {
            'filename': problem['filename'],
            'success': False,
            'error': error_msg,
            'error_code': -999  # 未预期的错误
        }


def run_evaluation(
    problems_dir: str,
    db_path: str = 'arc_results.db',
    run_id: str = None,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    运行评估流程的主函数，可被外部调用

    参数:
    - problems_dir: 问题文件所在目录
    - db_path: 数据库文件路径
    - run_id: 运行ID，如不提供则使用时间戳
    - config: 配置参数，覆盖默认配置

    返回:
    - 包含运行结果的字典
    """
    # 合并配置
    merged_config = DEFAULT_CONFIG.copy()
    if config:
        # 只更新显式提供的非DEFAULT值
        merged_config.update({k: v for k, v in config.items() if v is not DEFAULT})

    # 确保API令牌存在
    if not merged_config.get("api_token"):
        # 尝试从环境变量获取
        api_key_env = os.environ.get("SF_API_KEY")
        if api_key_env:
            merged_config["api_token"] = api_key_env
            print("从环境变量 SF_API_KEY 加载了 API Token")
        else:
            raise ValueError("必须在配置中提供 'api_token' 或设置 SF_API_KEY 环境变量")

    # 为本次运行生成唯一ID
    run_id = run_id if run_id else time.strftime("%Y%m%d_%H%M%S")

    # 获取多线程配置
    max_workers = int(merged_config.get("max_workers", DEFAULT_CONFIG["max_workers"]))
    save_interval = int(merged_config.get("save_interval", DEFAULT_CONFIG["save_interval"]))

    problems = load_problems(problems_dir)
    if not problems:
        print("错误：未能加载任何问题，评估中止。")
        return {
            'run_id': run_id,
            'error': '未能加载任何问题',
            'total_problems': 0,
            'successful_count': 0,
            'correct_count': 0,
            'accuracy': 0.0
        }

    results = []
    completed = 0

    print(f"本次运行ID: {run_id}")
    print(f"并行处理线程数: {max_workers}")

    # 创建线程互斥锁
    results_lock = threading.Lock()
    print_lock = threading.Lock()

    # 创建运行元数据
    run_metadata = {
        'run_id': run_id,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': merged_config.get('model'),
        'prompt': merged_config.get('prompt'),
        'problems_dir': problems_dir,
        'total_problems': len(problems),
        'config': {k: v for k, v in merged_config.items() if k != 'api_token'},  # 不保存API令牌
        'results': []  # 将在处理过程中填充
    }

    # 准备任务参数
    tasks = [(problem, merged_config, i, len(problems), print_lock)
             for i, problem in enumerate(problems)]

    # 如果是单线程模式，顺序处理
    if max_workers <= 1:
        print("以单线程模式运行...")
        for task in tasks:
            result = evaluate_problem_wrapper(task)
            results.append(result)
            run_metadata['results'].append(result)
            completed += 1

            # 定期保存中间结果
            if completed % save_interval == 0 or completed == len(problems):
                save_results_to_db(db_path, run_metadata)
                print(f"已完成: {completed}/{len(problems)} - 已保存结果到数据库 {db_path}")
    else:
        # 多线程模式
        print(f"以多线程模式运行 (max_workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_idx = {executor.submit(evaluate_problem_wrapper, task): i
                            for i, task in enumerate(tasks)}

            # 处理完成的任务
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]

                try:
                    result = future.result()
                    with results_lock:
                        results.append(result)
                        run_metadata['results'].append(result)
                        completed += 1

                        # 定期保存中间结果
                        if completed % save_interval == 0 or completed == len(problems):
                            save_results_to_db(db_path, run_metadata)
                            print(f"已完成: {completed}/{len(problems)} - 已保存结果到数据库 {db_path}")

                except Exception as e:
                    # 捕获 future.result() 可能抛出的异常
                    error_msg = f"处理问题 {problems[idx]['filename']} 时线程池发生错误: {str(e)}"
                    print(error_msg)

                    error_result = {
                        'filename': problems[idx]['filename'],
                        'success': False,
                        'error': error_msg,
                        'error_code': -998 # 线程池层面的错误
                    }

                    with results_lock:
                        results.append(error_result)
                        run_metadata['results'].append(error_result)
                        completed += 1
                        # 即使出错也尝试保存
                        if completed % save_interval == 0 or completed == len(problems):
                             save_results_to_db(db_path, run_metadata)
                             print(f"已完成: {completed}/{len(problems)} - 已保存结果(含错误)到数据库 {db_path}")


    # 计算整体统计
    successful_results = [r for r in results if r.get('success', False)]
    correct_count = sum(1 for r in successful_results if r.get('correct', False))
    prompt_tokens_sum = sum(r.get('prompt_tokens', 0) for r in successful_results)
    completion_tokens_sum = sum(r.get('completion_tokens', 0) for r in successful_results)

    # 更新运行元数据中的统计信息
    run_metadata['successful_count'] = len(successful_results)
    run_metadata['correct_count'] = correct_count
    # 避免除零错误
    total_processed = len(results) # 使用总处理数作为分母更合理
    run_metadata['accuracy'] = correct_count / total_processed if total_processed > 0 else 0.0
    # run_metadata['accuracy'] = correct_count/len(successful_results) if successful_results else 0 # 原计算方式
    run_metadata['prompt_tokens_sum'] = prompt_tokens_sum
    run_metadata['completion_tokens_sum'] = completion_tokens_sum
    run_metadata['total_tokens'] = prompt_tokens_sum + completion_tokens_sum
    run_metadata['completed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n总体结果:")
    print(f"总问题数: {len(problems)}")
    print(f"成功处理: {len(successful_results)}/{total_processed}")
    print(f"正确: {correct_count}/{total_processed} ({run_metadata['accuracy']*100:.2f}%)" if total_processed > 0 else "无处理结果")
    print(f"总prompt tokens: {prompt_tokens_sum}， 总completion tokens: {completion_tokens_sum}， token总量: {prompt_tokens_sum+completion_tokens_sum}")

    # 保存最终结果
    save_results_to_db(db_path, run_metadata)
    print(f"最终结果已保存到数据库 {db_path}")

    return run_metadata
