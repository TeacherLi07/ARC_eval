import os
import uuid
import functools
import sys
from pathlib import Path

import faulthandler
faulthandler.enable()

# 将项目根目录添加到 sys.path，以便导入 code 包
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 现在可以从 code.core 导入
from code.core.evaluation import run_evaluation, DEFAULT

# 生成一个短的随机hash后缀
# random_suffix = uuid.uuid4().hex[:8]  # 取8个字符作为短hash
api_key = os.environ.get("SF_API_KEY")

# 如果未设置环境变量，则提示用户
if not api_key:
    print("错误：请设置环境变量 SF_API_KEY")
    sys.exit(1)


print = functools.partial(print, flush=True)

# 使用相对路径或基于项目根的路径
problems_path = project_root / "data" / "ARC-AGI-1" / "data" / "evaluation"
db_path = project_root / "results"/ "newcall_test.db"

for i in range(64):
    run_id_suffix = uuid.uuid4().hex[:8] # 为每次运行生成不同的后缀
    result = run_evaluation(
        problems_dir=str(problems_path), # 转换为字符串
        db_path=str(db_path), # 转换为字符串
        run_id=f"7b_No.{i+1}",  # 添加随机后缀到原始run_id
        config = {
        # API配置 (api_token 会自动从环境变量加载)
        "api_url": "https://api.siliconflow.cn/v1/chat/completions",
        # "api_token": api_key, # 不再需要显式传递

        # 模型配置
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", # Qwen/QwQ-32B
        # "stream": DEFAULT, # 使用默认值 False
        "max_tokens": 16384, # 增加 max_tokens
        "temperature": 1.0, 
        # "top_p": DEFAULT, # 使用默认值 0.7
        # "top_k": DEFAULT, # 使用默认值 50
        "frequency_penalty": 0, # 设置 frequency_penalty
        # "n": DEFAULT, # 使用默认值 1
        # "response_format": {"type": "text"}, # 使用默认值
        # "extra_body":
        #     {
        #     "prefix": "```json\n",
        #     },
        # "stop": DEFAULT, # 使用默认值

        # 运行配置
        "max_workers": 16, # 设置 max_workers
        # "save_interval": DEFAULT, # 使用默认值 1
        # "retry_delay": DEFAULT, # 使用默认值 5
        "max_retries": 5, # 设置 max_retries
        # "api_call_interval": DEFAULT, # 使用默认值 1.0
        "verbose": False, # 启用详细输出

        # 默认prompt (使用中文prompt)
        "prompt": """你是一个擅长深度思考的AI助手，能够发现并运用规律。接下来是一道ARC-AGI测试题，包含三组输入-输出样例和一个测试输入。输入和输出以二维数组形式表示图片像素，不同数字代表不同颜色，数字之间无运算关系，有方位关系。
请你分析样例，找出输入与输出之间的统一规律，将规律应用于测试输入，给出对应的输出。常见规律包括但不限于：图像填充、补全、线条延伸、颜色标记、对称性、周期性、去噪等。
你需要全面思考所有可能性，确保规律的正确性。你有足够的时间来思考。
请用JSON格式回答，只包含完整的预测输出"output":
""",
        # "train_input_prompt" : DEFAULT, # 使用默认值
        # "train_output_prompt" : DEFAULT, # 使用默认值
        # "test_prompt" : DEFAULT, # 使用默认值
        # "end_prompt": DEFAULT # 使用默认值
        }
    )

# 英文 Prompt 的运行 (注释掉了，如果需要可以取消注释)
# run_id_suffix_eng = uuid.uuid4().hex[:8]
# result_eng = run_evaluation(
#     problems_dir=str(problems_path),
#     db_path=str(project_root / "results_eng.db"), # 使用不同的数据库文件
#     run_id=f"engprompt-7b_{run_id_suffix_eng}",
#     config = {
#     "api_url": "https://api.siliconflow.cn/v1/chat/completions",
#     "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
#     "max_tokens": 16384,
#     "frequency_penalty": 0,
#     "extra_body":
#         {
#         "prefix": "```json\n{\n", # 注意与中文prompt的prefix不同
#         },
#     "stop": ["}\n```", "}```"], # 添加 stop token
#     "max_workers": 12,
#     "max_retries": 5,
#     "verbose": False, # 可以设为 True 调试
#     "prompt": """You are an AI assistant skilled in deep thinking, capable of discovering and applying patterns. Below is an ARC-AGI test question with specific requirements:

# 1. **Problem Description**
#     - It includes three sets of input-output examples and one test input.
#     - The input and output are represented as images in the form of 2D arrays, where different numbers represent different colors, and there are no arithmetic relationships between the numbers.

# 2. **Task Objective**
#     - Analyze the examples and identify a consistent pattern between the input and output.
#     - Apply the discovered pattern to the test input and generate the corresponding output.

# 3. **Common Pattern Hints**
#     - Patterns may include but are not limited to: image filling, completion, line extension, color marking, symmetry, periodicity, noise removal, etc. You need to consider all possible patterns comprehensively to ensure correctness.

# Think carefully and respond in JSON format, containing only the complete predicted output "output".
# """,
#     "train_input_prompt" : "Train input", # 英文提示
#     "train_output_prompt" : "Train output", # 英文提示
#     "test_prompt" : "Test input:", # 英文提示
#     "end_prompt": "Think carefully and respond in JSON format, containing only the complete predicted output "output":" # 英文提示
#     }
# )

print("所有评估运行完成。")

