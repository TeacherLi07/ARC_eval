import sqlite3
import json
from datetime import datetime
from typing import Dict, Any

def create_db_tables(db_path: str) -> None:
    """创建必要的数据库表结构"""
    conn = sqlite3.connect(db_path)
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

    # 创建结果表 - 增加 error_code 字段
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
    conn.close()

def save_results_to_db(db_path: str, run_metadata: Dict[str, Any]) -> None:
    """
    将本次运行的结果保存到SQLite数据库
    如果数据库不存在，则创建新数据库和表
    """
    # 确保数据库表存在
    create_db_tables(db_path)

    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 检查是否已存在相同run_id的记录
    cursor.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_metadata.get('run_id'),))
    exists = cursor.fetchone() is not None

    # 准备运行元数据
    run_data = (
        run_metadata.get('run_id'),
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
        cursor.execute('''
        UPDATE runs
        SET timestamp = ?, completed_at = ?, model = ?, prompt = ?,
            problems_dir = ?, total_problems = ?, successful_count = ?,
            correct_count = ?, accuracy = ?, prompt_tokens_sum = ?,
            completion_tokens_sum = ?, total_tokens = ?, config = ?
        WHERE run_id = ?
        ''', run_data[1:] + (run_data[0],))

        # 删除原有结果，稍后重新插入
        cursor.execute("DELETE FROM results WHERE run_id = ?", (run_metadata.get('run_id'),))
    else:
        cursor.execute('''
        INSERT INTO runs (
            run_id, timestamp, completed_at, model, prompt, problems_dir,
            total_problems, successful_count, correct_count, accuracy,
            prompt_tokens_sum, completion_tokens_sum, total_tokens, config
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', run_data)

    # 保存每个问题的结果
    for result in run_metadata.get('results', []):
        result_data = (
            run_metadata.get('run_id'),
            result.get('filename'),
            1 if result.get('success') else 0,
            1 if result.get('correct') else 0,
            result.get('error', ''),
            result.get('error_code', 0),  # 添加错误代码
            json.dumps(result.get('predicted'), ensure_ascii=False),
            json.dumps(result.get('actual'), ensure_ascii=False),
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

    # 提交事务并关闭连接
    conn.commit()
    conn.close()
