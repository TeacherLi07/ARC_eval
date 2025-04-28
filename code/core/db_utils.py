import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union

# 配置日志记录，保持与api_handler和utils一致的风格
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_utils")

def create_db_tables(db_path: str) -> bool:
    """
    创建必要的数据库表结构，如果不存在的话。
    
    参数:
        db_path: 数据库文件的路径
        
    返回:
        bool: 表示操作是否成功
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 创建运行元数据表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                completed_at TEXT,
                model TEXT,
                prompt TEXT,
                problems_dir TEXT,
                total_problems INTEGER,
                successful_count INTEGER,
                correct_count INTEGER,
                accuracy REAL,
                prompt_tokens_sum INTEGER,
                completion_tokens_sum INTEGER,
                total_tokens INTEGER,
                config TEXT
            )
            ''')

            # 创建结果表 - 包含 error_code 字段
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                filename TEXT,
                success INTEGER,
                correct INTEGER,
                error TEXT,
                error_code INTEGER,
                predicted TEXT,
                actual TEXT,
                reasoning TEXT,
                response TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
            ''')

            conn.commit()
            logger.debug(f"成功创建或确认数据库表结构：{db_path}")
            return True
            
    except sqlite3.Error as e:
        logger.error(f"创建数据库表时发生错误: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"创建数据库表时发生意外错误: {str(e)}")
        return False

def _validate_run_metadata(run_metadata: Dict[str, Any]) -> Tuple[bool, str]:
    """
    验证运行元数据的有效性。
    
    参数:
        run_metadata: 要验证的运行元数据字典
        
    返回:
        (valid, message): 包含验证结果和错误信息（如果有）的元组
    """
    if not run_metadata:
        return False, "运行元数据不能为空"
        
    # 检查必要的字段
    required_fields = ['run_id']
    for field in required_fields:
        if field not in run_metadata:
            return False, f"运行元数据缺少必要字段: {field}"
            
    # 验证run_id不为空
    if not run_metadata.get('run_id'):
        return False, "run_id不能为空"
        
    # 验证结果列表（如果存在）
    results = run_metadata.get('results', [])
    if results and not isinstance(results, list):
        return False, "results必须是列表类型"
        
    return True, ""

def save_results_to_db(db_path: str, run_metadata: Dict[str, Any]) -> bool:
    """
    将本次运行的结果保存到SQLite数据库。
    
    该函数会检查必要的字段，验证数据有效性，并在事务中进行所有操作，
    确保数据一致性。如果数据库不存在，则会创建新的数据库和表结构。
    
    参数:
        db_path: 数据库文件的路径
        run_metadata: 包含运行结果和元数据的字典
        
    返回:
        bool: 表示操作是否成功
    """
    # 验证元数据有效性
    valid, message = _validate_run_metadata(run_metadata)
    if not valid:
        logger.error(f"无效的运行元数据: {message}")
        return False
        
    # 确保数据库表存在
    if not create_db_tables(db_path):
        logger.error("无法创建数据库表，保存操作中止")
        return False

    try:
        # 使用with语句管理连接
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 确保事务一致性
            conn.execute("BEGIN TRANSACTION")
            
            # 检查是否已存在相同run_id的记录
            run_id = run_metadata.get('run_id')
            cursor.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,))
            exists = cursor.fetchone() is not None

            # 准备运行元数据
            run_data = (
                run_id,
                run_metadata.get('timestamp'),
                run_metadata.get('completed_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                run_metadata.get('model'),
                run_metadata.get('prompt'),
                run_metadata.get('problems_dir'),
                run_metadata.get('total_problems'),
                run_metadata.get('successful_count', 0),
                run_metadata.get('correct_count', 0),
                run_metadata.get('accuracy', 0.0),
                run_metadata.get('prompt_tokens_sum', 0),
                run_metadata.get('completion_tokens_sum', 0),
                run_metadata.get('total_tokens', 0),
                json.dumps(run_metadata.get('config', {}), ensure_ascii=False)
            )

            # 插入或更新运行元数据
            if exists:
                logger.info(f"更新已存在的运行记录: {run_id}")
                cursor.execute('''
                UPDATE runs
                SET timestamp = ?, completed_at = ?, model = ?, prompt = ?,
                    problems_dir = ?, total_problems = ?, successful_count = ?,
                    correct_count = ?, accuracy = ?, prompt_tokens_sum = ?,
                    completion_tokens_sum = ?, total_tokens = ?, config = ?
                WHERE run_id = ?
                ''', run_data[1:] + (run_data[0],))

                # 删除原有结果，稍后重新插入
                cursor.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
            else:
                logger.info(f"创建新的运行记录: {run_id}")
                cursor.execute('''
                INSERT INTO runs (
                    run_id, timestamp, completed_at, model, prompt, problems_dir,
                    total_problems, successful_count, correct_count, accuracy,
                    prompt_tokens_sum, completion_tokens_sum, total_tokens, config
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', run_data)

            # 保存每个问题的结果
            results = run_metadata.get('results', [])
            if results:
                logger.debug(f"保存 {len(results)} 条结果记录")
                for result in results:
                    try:
                        # 尝试将预测和实际结果序列化为JSON，如果不是有效类型则使用空列表
                        predicted = json.dumps(result.get('predicted', []), ensure_ascii=False) 
                        actual = json.dumps(result.get('actual', []), ensure_ascii=False)
                    except (TypeError, ValueError) as e:
                        logger.warning(f"序列化结果数据失败，使用空值: {str(e)}")
                        predicted = json.dumps([])
                        actual = json.dumps([])

                    result_data = (
                        run_id,
                        result.get('filename', ''),
                        1 if result.get('success') else 0,
                        1 if result.get('correct') else 0,
                        result.get('error', ''),
                        result.get('error_code', 0),
                        predicted,
                        actual,
                        result.get('reasoning', ''),
                        result.get('response', ''),
                        result.get('prompt_tokens', 0),
                        result.get('completion_tokens', 0)
                    )

                    cursor.execute('''
                    INSERT INTO results (
                        run_id, filename, success, correct, error, error_code, predicted, actual,
                        reasoning, response, prompt_tokens, completion_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', result_data)

            # 提交事务
            conn.commit()
            logger.info(f"已成功保存运行记录到数据库: {db_path}")
            return True
            
    except sqlite3.Error as e:
        logger.error(f"数据库操作错误: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"保存结果时发生意外错误: {str(e)}")
        return False


