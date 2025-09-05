"""
多模型Tokenizer对比与训练时间估算工具
整合了TokenizerComparator和TrainingTimeEstimator功能
"""

import sys
import os
import re

# 添加当前目录到路径，确保可以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 延迟导入模块，减少启动时间
_tokenizer_comparator = None
_training_time_estimator = None

def get_tokenizer_comparator():
    """延迟加载TokenizerComparator"""
    global _tokenizer_comparator
    if _tokenizer_comparator is None:
        try:
            from TokenizerComparator import TokenizerComparator
            _tokenizer_comparator = TokenizerComparator()
        except Exception as e:
            print(f"加载 TokenizerComparator 失败: {e}")
            return None
    return _tokenizer_comparator

def get_training_time_estimator():
    """延迟加载TrainingTimeEstimator"""
    global _training_time_estimator
    if _training_time_estimator is None:
        try:
            from TrainingTimeEstimator import TrainingTimeEstimator
            _training_time_estimator = TrainingTimeEstimator()
        except Exception as e:
            print(f"加载 TrainingTimeEstimator 失败: {e}")
            return None
    return _training_time_estimator

def resource_path(relative_path):
    """获取资源的绝对路径"""
    # 首先检查当前目录下是否存在该文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(current_dir, relative_path)
    if os.path.exists(local_path):
        return local_path
    
    # 如果当前目录下不存在，则检查打包环境
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def is_frozen():
    """检查是否在打包环境中运行"""
    return getattr(sys, 'frozen', False)

def ask_for_restart():
    """询问用户是否重新开始"""
    choice = input("\n是否进行下一次时间估算? (Y/N): ").strip().lower()
    if choice in ['y', 'yes']:
        print("\n" + "="*60)
        print("开始新一轮估算...")
        print("="*60)
        return True
    elif choice in ['n', 'no']:
        print("感谢使用，再见!")
        return False
    else:
        print("请输入 Y 或 N")
        return ask_for_restart()

def parse_dataset_size(size_str):
    """
    解析带单位的数据集大小字符串，返回KB值
    
    Args:
        size_str: 带单位的数据集大小字符串 (如: "10GB", "500MB", "1.5TB")
    
    Returns:
        数据集大小的KB值
    """
    # 使用正则表达式匹配数字和单位
    match = re.match(r"^\s*([\d.]+)\s*([KMGTP]?B?)\s*$", size_str, re.IGNORECASE)
    if not match:
        raise ValueError("无效的数据集大小格式")
    
    size = float(match.group(1))
    unit = match.group(2).upper()
    
    # 根据单位转换为KB
    if unit in ["KB", "K"]:
        return size
    elif unit in ["MB", "M"]:
        return size * 1024
    elif unit in ["GB", "G"]:
        return size * 1024 * 1024
    elif unit in ["TB", "T"]:
        return size * 1024 * 1024 * 1024
    else:
        # 默认单位为KB
        return size

def get_target_dataset_size():
    """获取用户输入的目标数据集大小（带单位）"""
    while True:
        try:
            size_input = input("\n请输入您计划使用的数据集大小(可带单位: KB/MB/GB/TB): ").strip()
            size_kb = parse_dataset_size(size_input)
            if size_kb > 0:
                return size_kb
            else:
                print("数据集大小必须大于0")
        except ValueError as e:
            print(f"输入错误: {e}")

def scale_tokens_by_dataset_size(total_tokens_dict, default_dataset_size_kb, target_dataset_size_kb):
    """
    根据目标数据集大小缩放token数量
    
    Args:
        total_tokens_dict: 原始token数量字典
        default_dataset_size_kb: 默认数据集大小(KB)
        target_dataset_size_kb: 目标数据集大小(KB)
    
    Returns:
        缩放后的token数量字典
    """
    # 计算缩放比例
    scale_factor = target_dataset_size_kb / default_dataset_size_kb
    
    # 缩放每个模型的token数量
    scaled_tokens_dict = {}
    for model_name, token_count in total_tokens_dict.items():
        scaled_tokens_dict[model_name] = int(token_count * scale_factor)
    
    return scaled_tokens_dict

def format_size_kb(size_kb):
    """将KB大小格式化为更易读的格式"""
    if size_kb >= 1024 * 1024 * 1024:  # TB
        return f"{size_kb / (1024 * 1024 * 1024):.2f} TB"
    elif size_kb >= 1024 * 1024:  # GB
        return f"{size_kb / (1024 * 1024):.2f} GB"
    elif size_kb >= 1024:  # MB
        return f"{size_kb / 1024:.2f} MB"
    else:  # KB
        return f"{size_kb:.2f} KB"

def run_tokenizer_comparison():
    """运行tokenizer比较并返回结果"""
    # 延迟加载比较器
    comparator = get_tokenizer_comparator()
    if comparator is None:
        return None, None, None, None, None
    
    # 确保配置已加载
    comparator.load_config()
    
    # 检查数据集选项是否为空
    if not comparator.dataset_options:
        print("配置文件中没有定义数据集!")
        return None, None, None, None, None
    
    # 使用第一个数据集，不再让用户选择
    dataset_choice_idx = 0
    
    # 输出使用的数据集信息
    dataset_name = comparator.dataset_options[dataset_choice_idx]['name']
    
    # 提供交互式模型选择
    print("\n请选择要比较的模型 (可多选，用逗号分隔):")
    for i, model in enumerate(comparator.model_options, 1):
        note_parts = []
        if model.get('trust_remote_code', False):
            note_parts.append("需要trust_remote_code")
        if model.get('auth_required', False):
            note_parts.append("需要认证")
        note = f" ({', '.join(note_parts)})" if note_parts else ""
        print(f"{i}. {model['name']}{note}")
    
    try:
        model_choices = input("\n请输入模型编号 (例如: 1,2,3): ").split(',')
        selected_model_indices = []
        for choice in model_choices:
            choice_idx = int(choice.strip()) - 1
            if 0 <= choice_idx < len(comparator.model_options):
                selected_model_indices.append(choice_idx)
        
        # 如果选择了"全部模型"，则包含所有模型
        all_model_idx = next((i for i, m in enumerate(comparator.model_options) if m.get('is_all_option', False)), -1)
        if all_model_idx != -1 and (all_model_idx + 1) in [int(c.strip()) for c in model_choices]:
            selected_model_indices = [i for i, m in enumerate(comparator.model_options) if not m.get('is_all_option', False)]
    except ValueError:
        return None, None, None, None, None
    
    if not selected_model_indices:
        return None, None, None, None, None
    
    # 加载数据集文本
    dataset_texts = comparator.load_dataset_texts(dataset_choice_idx)
    if not dataset_texts:
        print("无法加载数据集!")
        return None, None, None, None, None
    
    # 加载tokenizer
    successful_models = comparator.load_tokenizers(selected_model_indices)
    if not successful_models:
        print("未能加载任何tokenizer!")
        return None, None, None, None, None
    
    # 进行处理
    results = comparator.process_dataset(dataset_texts, successful_models)
    
    # 获取模型名称和总token数
    model_names = []
    total_tokens_dict = {}
    
    for model_key in successful_models:
        if model_key in comparator.tokenizers:
            model_name = comparator.tokenizers[model_key].model_name
            model_names.append(model_name)
            
            # 从结果中获取总token数
            if model_name in results['models']:
                total_tokens = results['models'][model_name]['total_tokens']
                total_tokens_dict[model_name] = total_tokens
    
    # 获取默认数据集大小(KB)
    default_dataset_size_kb = results['total_bytes'] / 1024
    
    return results, model_names, total_tokens_dict, default_dataset_size_kb, selected_model_indices

def run_training_time_estimation(total_tokens_dict):
    """运行训练时间估算"""
    try:
        # 延迟加载估算器
        estimator = get_training_time_estimator()
        if estimator is None:
            print("无法加载训练时间估算器")
            return
        estimator.running_train(total_tokens_dict)
    except Exception as e:
        print(f"训练时间估算出错: {e}")

def main():
    """主函数，整合两个模块的功能"""
    try:
        # 设置当前工作目录
        if is_frozen():
            os.chdir(os.path.dirname(sys.executable))
        
        while True:
            print("="*60)
            print("       多模型Tokenizer对比与训练时间估算工具")
            print("="*60)
            
            # 运行tokenizer比较
            results, model_names, total_tokens_dict, default_dataset_size_kb, selected_model_indices = run_tokenizer_comparison()
            
            if not results or not total_tokens_dict:
                print("Tokenizer比较失败，无法继续估算训练时间")
                if not ask_for_restart():
                    break
                continue
            
            # 检查是否选择了"全部模型"
            comparator = get_tokenizer_comparator()
            all_model_idx = next((i for i, m in enumerate(comparator.model_options) if m.get('is_all_option', False)), -1)
            is_all_selected = all_model_idx != -1 and all_model_idx in selected_model_indices
            
            # 显示默认数据集信息
            print(f"\n默认数据集大小: {format_size_kb(default_dataset_size_kb)}")
            
            # 获取用户输入的目标数据集大小
            target_dataset_size_kb = get_target_dataset_size()
            
            # 根据目标数据集大小缩放token数量
            scaled_tokens_dict = scale_tokens_by_dataset_size(
                total_tokens_dict, default_dataset_size_kb, target_dataset_size_kb
            )
            
            # 显示缩放后的token数量
            print(f"\n目标数据集大小: {format_size_kb(target_dataset_size_kb)}")
            
            # 运行训练时间估算
            run_training_time_estimation(scaled_tokens_dict)
            
            # 询问用户是否重新开始
            if not ask_for_restart():
                break
                
    except Exception as e:
        print(f"程序初始化失败: {e}")
        input("按任意键退出...")  # 等待用户按键，以便查看错误信息

if __name__ == "__main__":
    main()