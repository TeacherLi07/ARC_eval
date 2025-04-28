import json
import re
import logging
from typing import List, Optional, Dict, Any, Tuple

# 配置日志记录，与api_handler保持一致的风格
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("utils")

def _is_valid_2d_int_array(obj) -> bool:
    """
    检查对象是否为有效的二维整数数组。
    
    参数:
        obj: 待检查的对象
        
    返回:
        bool: 如果对象是二维整数数组则返回True，否则返回False
    """
    if not isinstance(obj, list):
        return False
        
    # 检查是否为空数组
    if not obj:
        return True  # 允许空的二维数组 []
        
    for row in obj:
        if not isinstance(row, list):
            return False
        # 允许空行（空列表）
        for item in row:
            if not isinstance(item, int):
                return False
                
    return True

def _filter_array_chars(content: str) -> str:
    """
    过滤字符串，只保留可能在二维数组中出现的字符：
    方括号、数字、逗号、负号、空白字符
    
    参数:
        content: 原始文本内容
        
    返回:
        str: 过滤后只包含可能在数组中的字符的字符串
    """
    result = []
    for char in content:
        if char in '[],-0123456789 \t\n':
            result.append(char)
    return ''.join(result)

def _find_arrays(content: str) -> List[str]:
    """
    使用高效的手动解析方法在文本内容中查找可能的二维数组。
    
    参数:
        content: 要搜索的文本内容
        
    返回:
        List[str]: 匹配到的可能是二维数组的字符串列表
    """
    # 快速检查：如果没有足够的方括号，肯定不包含二维数组
    if content.count('[') < 2 or content.count(']') < 2:
        return []
    
    # 过滤内容，只保留可能在数组中的字符
    filtered_content = _filter_array_chars(content)
    
    results = []
    i = 0
    content_len = len(filtered_content)
    
    while i < content_len:
        # 寻找外部数组开始
        if filtered_content[i] == '[':
            # 查找下一个非空白字符
            j = i + 1
            while j < content_len and filtered_content[j].isspace():
                j += 1
            
            # 如果下一个非空白字符不是'['，说明这不是二维数组的开始
            if j >= content_len or filtered_content[j] != '[':
                i += 1
                continue
            
            # 可能是二维数组的开始，记录起始位置
            start_pos = i
            depth = 1  # 括号嵌套深度
            i += 1
            
            # 查找匹配的结束括号
            while i < content_len and depth > 0:
                if filtered_content[i] == '[':
                    depth += 1
                elif filtered_content[i] == ']':
                    depth -= 1
                i += 1
                
            # 找到匹配的结束位置
            if depth == 0:
                array_str = filtered_content[start_pos:i]
                # 基本格式验证：确保至少有两层括号
                if array_str.count('[') >= 2 and array_str.count(']') >= 2:
                    # 尝试解析确保是有效的二维数组
                    try:
                        parsed = json.loads(array_str)
                        if _is_valid_2d_int_array(parsed):
                            results.append(array_str)
                    except:
                        # 如果解析失败，可能需要清理更多格式问题
                        # 这里可以尝试进一步清理，但简单起见我们先跳过
                        pass
        else:
            i += 1
            
    return results

def _extract_json_objects(content: str) -> List[Dict]:
    """
    从文本内容中提取可能的JSON对象。
    
    参数:
        content: 要搜索的文本内容
        
    返回:
        List[Dict]: 提取并成功解析的JSON对象列表
    """
    results = []
    
    # 1. 首先查找代码块中的JSON
    json_blocks = re.findall(r'```(?:json)?\s*(\{[^`]*?\})\s*```', content, re.DOTALL)
    for json_str in json_blocks:
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict):
                results.append(obj)
        except (json.JSONDecodeError, Exception):
            pass
    
    # 2. 然后搜索可能的独立JSON对象
    # 使用更健壮的嵌套花括号匹配方法
    depth = 0
    start_idx = -1
    for i, char in enumerate(content):
        if char == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start_idx != -1:
                # 找到了完整的JSON对象
                try:
                    json_str = content[start_idx:i+1]
                    obj = json.loads(json_str)
                    if isinstance(obj, dict):
                        results.append(obj)
                except (json.JSONDecodeError, Exception):
                    pass
                            
    return results

def extract_output_from_response(response: Dict[str, Any]) -> Optional[List[List[int]]]:
    """
    从LLM响应中提取预测的输出网格。
    
    算法策略:
    1. 首先检查响应的有效性
    2. 使用高效的手动解析方法查找所有形如二维数组的文本
    3. 解析这些文本，查找有效的二维整数数组
    4. 如无法找到，尝试从响应中提取JSON对象，检查其中的'output'字段
    
    参数:
        response: LLM的响应字典，预期格式为包含data->choices->message->content路径
        
    返回:
        Optional[List[List[int]]]: 找到的二维整数数组，如未找到则返回None
    """
    # 1. 安全地提取内容，增强健壮性
    try:
        if not response or not isinstance(response, dict):
            logger.warning("提供的响应为空或不是字典")
            return None
            
        if not response.get('success', False):
            logger.debug("响应标记为不成功")
            return None
            
        data = response.get('data')
        if not data or not isinstance(data, dict):
            logger.debug("响应中无有效数据字段")
            return None
            
        choices = data.get('choices')
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            logger.debug("响应中无有效选择字段")
            return None
            
        message = choices[0].get('message')
        if not message or not isinstance(message, dict):
            logger.debug("响应中无有效消息字段")
            return None
            
        content = message.get('content')
        if not content or not isinstance(content, str):
            logger.debug("响应中无有效内容字段")
            return None
            
    except Exception as e:
        logger.warning(f"提取响应内容时出错: {str(e)}")
        return None
    
    # 2. 使用高效方法查找可能的二维数组
    array_candidates = _find_arrays(content)
    
    # 3. 从后向前尝试解析，查找有效的二维整数数组
    for candidate in reversed(array_candidates):
        try:
            array = json.loads(candidate)
            if _is_valid_2d_int_array(array):
                logger.debug("找到有效的二维数组")
                return array
        except:
            continue
    
    # 4. 如果直接解析方法失败，尝试提取JSON对象
    json_objects = _extract_json_objects(content)
    for obj in json_objects:
        if 'output' in obj and _is_valid_2d_int_array(obj['output']):
            logger.debug("在JSON对象的'output'字段中找到有效的二维数组")
            return obj['output']
    
    # 5. 如果所有方法都失败，返回None
    logger.debug("未能在响应中找到有效的二维数组")
    return None
