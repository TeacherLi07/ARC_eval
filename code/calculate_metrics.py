import os
import sqlite3
import json
import argparse
from collections import Counter
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Set, Any

# 配置日志记录，与其他模块保持一致的风格
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("calculate_metrics")

def connect_db(db_path: str) -> sqlite3.Connection:
    """连接到SQLite数据库"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        logger.info(f"成功连接到数据库: {db_path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"连接数据库时发生错误: {str(e)}")
        raise

def get_run_ids(conn: sqlite3.Connection) -> List[str]:
    """获取数据库中所有的run_id"""
    cursor = conn.cursor()
    cursor.execute("SELECT run_id FROM runs ORDER BY timestamp")
    return [row[0] for row in cursor.fetchall()]

def get_run_info(conn: sqlite3.Connection, run_id: str) -> Dict[str, Any]:
    """获取特定运行的元数据"""
    cursor = conn.cursor()
    cursor.execute("""
    SELECT run_id, timestamp, model, total_problems, successful_count, 
           correct_count, accuracy
    FROM runs WHERE run_id = ?
    """, (run_id,))
    row = cursor.fetchone()
    if not row:
        return {}
    
    return {
        'run_id': row[0],
        'timestamp': row[1],
        'model': row[2],
        'total_problems': row[3],
        'successful_count': row[4],
        'correct_count': row[5],
        'accuracy': row[6]
    }

def get_results_by_filenames(conn: sqlite3.Connection, filenames: Set[str], run_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    按文件名获取多个运行的结果
    
    参数:
        conn: 数据库连接
        filenames: 文件名集合
        run_ids: 运行ID列表
    
    返回:
        Dict[str, List[Dict]]: 以文件名为键，结果列表为值的字典
    """
    cursor = conn.cursor()
    results_by_filename = {filename: [] for filename in filenames}
    
    # 将文件名和运行ID转换为列表
    filenames_list = list(filenames)
    
    # 构建参数化查询的占位符
    placeholders_filenames = ','.join(['?'] * len(filenames_list))
    placeholders_run_ids = ','.join(['?'] * len(run_ids))
    
    # 一次性查询所有符合条件的结果
    query = f"""
    SELECT filename, run_id, success, correct, predicted, actual
    FROM results 
    WHERE filename IN ({placeholders_filenames}) 
    AND run_id IN ({placeholders_run_ids})
    """
    
    # 执行批量查询
    cursor.execute(query, filenames_list + run_ids)
    
    # 处理查询结果
    for row in cursor.fetchall():
        filename = row[0]
        run_id = row[1]
        
        try:
            predicted = json.loads(row[4]) if row[4] else None
            actual = json.loads(row[5]) if row[5] else None
        except json.JSONDecodeError:
            logger.warning(f"解析JSON失败: {filename}, {run_id}")
            predicted = None
            actual = None
        
        result = {
            'run_id': run_id,
            'success': bool(row[2]),
            'correct': bool(row[3]),
            'predicted': predicted,
            'actual': actual
        }
        
        results_by_filename[filename].append(result)
    
    return results_by_filename

def get_problem_filenames(conn: sqlite3.Connection) -> Set[str]:
    """获取所有问题文件名"""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT filename FROM results")
    return {row[0] for row in cursor.fetchall()}

def calculate_pass_at_k(results_by_filename: Dict[str, List[Dict[str, Any]]], k: int) -> float:
    """
    计算pass@k指标
    
    参数:
        results_by_filename: 按文件名组织的结果字典
        k: 尝试次数上限
        
    返回:
        float: pass@k的比例
    """
    total_problems = 0
    passed_problems = 0
    
    for filename, results in results_by_filename.items():
        # 只考虑有足够结果的问题
        if len(results) >= k:
            total_problems += 1
            # 限制为前k个结果
            limited_results = results[:k]
            # 检查是否至少有一个结果是正确的
            if any(r['correct'] for r in limited_results):
                passed_problems += 1
    
    if total_problems == 0:
        return 0.0
    
    return passed_problems / total_problems

def calculate_majority_vote_at_k(results_by_filename: Dict[str, List[Dict[str, Any]]], k: int) -> float:
    """
    计算maj@k指标（多数投票）
    
    参数:
        results_by_filename: 按文件名组织的结果字典
        k: 参与投票的尝试次数上限
        
    返回:
        float: maj@k的比例
    """
    total_problems = 0
    correct_votes = 0
    
    for filename, results in results_by_filename.items():
        # 只考虑有足够结果的问题
        if len(results) >= k:
            total_problems += 1
            # 限制为前k个结果
            limited_results = results[:k]
            
            # 获取预测结果并进行投票
            predicted_outputs = []
            actual_output = None
            
            for result in limited_results:
                if result['success'] and result['predicted'] is not None:
                    # 将预测结果转换为JSON字符串以便于比较
                    pred_json = json.dumps(result['predicted'], sort_keys=True)
                    predicted_outputs.append(pred_json)
                
                # 保存实际输出（假设所有结果的actual字段都相同）
                if actual_output is None and result['actual'] is not None:
                    actual_output = result['actual']
            
            # 进行多数投票
            if predicted_outputs and actual_output is not None:
                vote_counter = Counter(predicted_outputs)
                majority_vote, _ = vote_counter.most_common(1)[0]  # 获取出现最多的预测
                
                # 将实际输出也转换为相同格式的JSON字符串
                actual_json = json.dumps(actual_output, sort_keys=True)
                
                # 比较多数投票结果和实际输出
                if majority_vote == actual_json:
                    correct_votes += 1
    
    if total_problems == 0:
        return 0.0
    
    return correct_votes / total_problems

def main(db_path: str, k_values: List[int] = [1, 8, 16, 32, 64]):
    """主函数，计算并显示各种评估指标"""
    conn = connect_db(db_path)
    
    try:
        # 获取所有运行ID
        run_ids = get_run_ids(conn)
        if not run_ids:
            logger.error("数据库中没有找到任何运行记录")
            return
        
        logger.info(f"找到 {len(run_ids)} 个运行记录")
        
        # 获取所有问题文件名
        filenames = get_problem_filenames(conn)
        logger.info(f"找到 {len(filenames)} 个不同的问题文件")
        
        # 获取每个文件名对应的结果
        results_by_filename = get_results_by_filenames(conn, filenames, run_ids)
        
        # 计算并显示每个k值的指标
        print("\n===== 评估指标结果 =====")
        for k in k_values:
            if k <= len(run_ids):
                pass_at_k = calculate_pass_at_k(results_by_filename, k)
                maj_at_k = calculate_majority_vote_at_k(results_by_filename, k)
                
                print(f"\nk = {k}:")
                print(f"  pass@{k}: {pass_at_k:.4f} ({int(pass_at_k * len(filenames))}/{len(filenames)})")
                print(f"  maj@{k}: {maj_at_k:.4f} ({int(maj_at_k * len(filenames))}/{len(filenames)})")
        
        # 显示单独运行的准确率
        print("\n===== 单次运行准确率 =====")
        for run_id in run_ids:  
            run_info = get_run_info(conn, run_id)
            if run_info:
                print(f"Run ID: {run_info['run_id']}")
                print(f"  准确率: {run_info['accuracy']:.4f} ({run_info['correct_count']}/{run_info['total_problems']})")
        
        
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算ARC评估指标")
    parser.add_argument("--db", default="results/newcall_test.db", 
                        help="数据库文件路径")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 8, 16, 32, 64], 
                        help="要计算的k值列表，例如 --k 1 8 16 32 64")
    
    args = parser.parse_args()
    
    # 确保数据库文件路径是绝对路径或相对于项目根目录的路径
    db_path = args.db
    if not os.path.isabs(db_path):
        project_root = Path(__file__).resolve().parent.parent
        db_path = os.path.join(project_root, db_path)
    
    main(db_path, args.k)
