import json
import re
from typing import List, Optional, Dict, Any

def _is_valid_2d_int_array(obj):
    """检查对象是否为列表的列表，且内部元素均为整数。"""
    if not isinstance(obj, list):
        return False
    for row in obj:
        if not isinstance(row, list):
            return False
        # 允许空行（空列表）
        for item in row:
            if not isinstance(item, int):
                return False
    return True

def extract_output_from_response(response: Dict[str, Any]) -> Optional[List[List[int]]]:
    """
    从LLM响应中提取预测的输出网格。
    尝试查找并解析响应内容中最后一个符合二维整数数组格式的部分。
    优先查找独立的二维数组，然后回退查找 JSON 对象中的 'output' 键。
    """
    if not response or not response.get('success') or 'data' not in response:
        return None

    data = response['data']

    if 'choices' not in data or not data['choices']:
        return None

    content = data['choices'][0]['message']['content']

    # 正则表达式查找所有看起来像二维整数数组的模式
    # Handles optional whitespace, negative numbers, empty lists.
    pattern = r"""
        \[                      # Start outer list '['
        \s*
        (?:                     # Optional group for the first inner list
            \[                  # Start inner list '['
            \s*
            (?:-?\d+\s*(?:,\s*-?\d+\s*)*)? # Optional numbers inside (int, optional neg)
            \s*
            \]                  # End inner list ']'
            \s*
        )?                      # Make the first inner list optional (allows `[]`)
        (?:                     # Group for subsequent inner lists prefixed by a comma
            ,\s*                # Comma separator
            \[                  # Start inner list '['
            \s*
            (?:-?\d+\s*(?:,\s*-?\d+\s*)*)? # Optional numbers inside
            \s*
            \]                  # End inner list ']'
            \s*
        )*                      # Zero or more subsequent inner lists
        \]                      # End outer list ']'
    """

    last_valid_array = None

    try:
        # 使用 re.finditer 查找所有匹配项
        matches = list(re.finditer(pattern, content, re.VERBOSE))

        # 从后往前检查匹配项，尝试解析和验证
        for match in reversed(matches):
            potential_match_str = match.group(0)
            try:
                # 尝试解析匹配到的字符串
                parsed_array = json.loads(potential_match_str)

                # 验证是否为二维整数数组
                if _is_valid_2d_int_array(parsed_array):
                    last_valid_array = parsed_array
                    # 找到最后一个有效的，就跳出循环
                    break

            except json.JSONDecodeError:
                # 解析失败，忽略此匹配，继续尝试前一个
                continue
            except Exception as e:
                # 其他验证错误
                # print(f"验证数组时出错: {str(e)} - 匹配: {potential_match_str[:100]}...") # 可选的调试信息
                continue

    except Exception as e:
        print(f"正则表达式查找或迭代时出错: {str(e)}")

    # 如果通过正则找到了有效的数组，返回它
    if last_valid_array is not None:
        return last_valid_array

    # --- 回退逻辑: 尝试解析 JSON 对象中的 'output' 键 ---
    try:
        # 查找可能包含 JSON 的代码块 (例如 ```json ... ```)
        json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        json_str = None
        if json_block_match:
             json_str = json_block_match.group(1)
        else:
            # 如果没有 ```json ... ```, 查找第一个 '{' 到最后一个 '}' 作为可能的 JSON 对象
            # 注意：这可能不是一个健壮的 JSON 对象查找方法，但与原始逻辑类似
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                 # 尝试解析这个潜在的 JSON 字符串
                 potential_json = content[start_idx:end_idx]
                 try:
                     # 尝试加载以确认它是有效的 JSON
                     json.loads(potential_json)
                     json_str = potential_json
                 except json.JSONDecodeError:
                     json_str = None # 不是有效的 JSON

        if json_str:
             result = json.loads(json_str)
             if 'output' in result and _is_valid_2d_int_array(result['output']):
                 # 即使在 JSON 对象中找到，也只返回这一个
                 return result['output']

    except Exception as e:
        # JSONDecodeError or other errors during fallback
        # print(f"提取输出时出错 (JSON对象回退方法): {str(e)}") # 避免过多日志
        pass # 如果正则和回退都失败，则静默处理

    # 如果所有方法都失败，返回 None
    return None
