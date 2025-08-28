"""
多模型Tokenizer对比与训练时间估算工具
整合了TokenizerComparator和TrainingTimeEstimator功能
"""

import sys
import os
import time
import logging
import traceback
import yaml

# 立即设置基本日志配置，确保最早捕获错误
def setup_early_logging():
    """设置早期日志配置"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app_debug.log"),
            logging.StreamHandler()
        ]
    )

# 立即设置日志
setup_early_logging()
logger = logging.getLogger(__name__)
logger.info("程序开始启动...")

# 添加当前目录到路径，确保可以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # 尝试导入主要模块
    logger.info("尝试导入模块...")
    from TokenizerComparator import TokenizerComparator
    from TrainingTimeEstimator import TrainingTimeEstimator
    logger.info("模块导入成功")
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    logger.error(traceback.format_exc())
    print(f"导入模块失败: {e}")
    print("请确保TokenizerComparator.py和TrainingTimeEstimator.py在同一目录下")
    input("按任意键退出...")
    sys.exit(1)

def resource_path(relative_path):
    """获取资源的绝对路径"""
    try:
        # PyInstaller创建临时文件夹，将路径存储在_MEIPASS中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def is_frozen():
    """检查是否在打包环境中运行"""
    return getattr(sys, 'frozen', False)

def ask_for_restart():
    """询问用户是否重新开始"""
    max_attempts = 3
    attempts = 0
    
    while attempts < max_attempts:
        try:
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
                attempts += 1
        except Exception as e:
            logging.error(f"获取用户输入时出错: {e}")
            attempts += 1
            if attempts >= max_attempts:
                print("输入错误次数过多，程序将退出。")
                return False
    
    return False

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def run_tokenizer_comparison(comparator):
    """运行tokenizer比较并返回结果"""
    # 提供交互式数据集选择
    print("\n请选择数据集:")
    for i, dataset in enumerate(comparator.dataset_options, 1):
        print(f"{i}. {dataset['name']}")
    
    try:
        dataset_choice_idx = int(input("\n请输入数据集编号: ")) - 1
        if dataset_choice_idx < 0 or dataset_choice_idx >= len(comparator.dataset_options):
            print("无效的选择!")
            return None, None, None
    except ValueError:
        print("请输入有效的数字!")
        return None, None, None
    
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
            print("已选择所有模型")
    except ValueError:
        print("请输入有效的数字!")
        return None, None, None
    
    if not selected_model_indices:
        print("未选择有效模型!")
        return None, None, None
    
    # 加载数据集文本
    print("正在加载数据集...")
    dataset_texts = comparator.load_dataset_texts(dataset_choice_idx)
    if not dataset_texts:
        print("无法加载数据集!")
        return None, None, None
    
    print(f"成功加载 {len(dataset_texts)} 个样本")
    
    # 加载tokenizer
    print("正在加载tokenizer...")
    successful_models = comparator.load_tokenizers(selected_model_indices)
    if not successful_models:
        print("未能加载任何tokenizer!")
        return None, None, None
    
    # 进行处理
    print("开始处理数据集...")
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
    
    return results, model_names, total_tokens_dict

def display_comparison_results(results, model_names, total_tokens_dict):
    """显示比较结果"""
    try:
        if not results:
            print("没有比较结果可显示")
            return
        
        print(f"\n总字节数: {results['total_bytes']:,}")
        print(f"样本数量: {results['sample_count']:,}")
        
        print("\n各模型tokenization统计:")
        print("-" * 60)
        
        for model_name in model_names:
            if model_name in results['models']:
                model_data = results['models'][model_name]
                
                if model_data['error_count'] > 0:
                    print(f"{model_name:20s}: 错误数: {model_data['error_count']}")
                else:
                    compression_ratio = results['total_bytes'] / model_data['total_tokens']
                    print(f"{model_name:20s}: {model_data['total_tokens']:,} tokens, "
                          f"压缩比: {compression_ratio:.2f} 字节/token")
    except Exception as e:
        logging.error(f"显示比较结果时出错: {e}")
        logging.error(traceback.format_exc())
        print(f"显示比较结果时出错: {e}")

def run_training_time_estimation(total_tokens_dict):
    """运行训练时间估算"""
    try:
        estimator = TrainingTimeEstimator()
        estimator.running_train(total_tokens_dict)  # 使用新的running_train方法
    except Exception as e:
        logging.error(f"训练时间估算出错: {e}")
        logging.error(traceback.format_exc())
        print(f"训练时间估算出错: {e}")
        # 即使出错也停留在结果页面
        input("按回车键返回主菜单...")

def validate_paths(config):
    """验证配置中的路径是否存在"""
    for dataset in config.get('datasets', []):
        if 'local_path' in dataset:
            abs_path = resource_path(dataset['local_path'])
            if not os.path.exists(abs_path):
                logging.warning(f"数据集本地路径不存在: {abs_path}")
    
    for model in config.get('models', []):
        if 'local_path' in model:
            abs_path = resource_path(model['local_path'])
            if not os.path.exists(abs_path):
                logging.warning(f"模型本地路径不存在: {abs_path}")

def main():
    """主函数，整合两个模块的功能"""
    # 设置日志
    setup_logging()
    
    try:
        # 设置当前工作目录
        if is_frozen():
            # 如果是打包环境，设置工作目录为EXE所在目录
            os.chdir(os.path.dirname(sys.executable))
        
        logging.info("程序启动成功")

        try:
            config_path = resource_path("config.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                validate_paths(config)
            else:
                logging.warning("配置文件不存在，跳过路径验证")
        except Exception as e:
            logging.warning(f"路径验证失败: {e}")
        
        while True:  # 添加循环，支持重复运行
            try:
                print("="*60)
                print("       多模型Tokenizer对比与训练时间估算工具")
                print("="*60)
                
                # 初始化比较器
                comparator = TokenizerComparator()
                
                # 运行tokenizer比较
                print("\n1. 运行Tokenizer比较...")
                start_time = time.time()
                results, model_names, total_tokens_dict = run_tokenizer_comparison(comparator)
                end_time = time.time()
                print(f"Tokenizer比较完成，耗时: {end_time - start_time:.2f} 秒")
                
                if not results or not total_tokens_dict:
                    print("Tokenizer比较失败，无法继续估算训练时间")
                    # 询问用户是否重新开始
                    if not ask_for_restart():
                        break
                    continue
                
                # 显示比较结果
                print("\n2. Tokenizer比较结果:")
                display_comparison_results(results, model_names, total_tokens_dict)
                
                 # 运行训练时间估算
                print("\n3. 运行训练时间估算...")
                run_training_time_estimation(total_tokens_dict)
                
                # 添加结果页面停留功能
                print("\n" + "="*60)
                print("           结果展示")
                print("="*60)
                
                # 显示tokenizer比较结果
                print("\nTokenizer比较结果:")
                display_comparison_results(results, model_names, total_tokens_dict)
                
                # 询问用户是否重新开始
                if not ask_for_restart():
                    # 添加最终停留
                    print("\n程序执行完毕，结果如上所示。")
                    input("按回车键退出程序...")
                    break
                    
            except Exception as e:
                logging.error(f"程序运行出错: {e}")
                logging.error(traceback.format_exc())
                print(f"程序运行出错: {e}")
                # 询问用户是否重新开始
                if not ask_for_restart():
                    # 添加错误情况下的停留
                    input("按回车键退出程序...")
                    break
                    
    except Exception as e:
        logging.error(f"程序初始化失败: {e}")
        logging.error(traceback.format_exc())
        print(f"程序初始化失败: {e}")
        input("按回车键退出...")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 创建紧急日志文件
        with open("crash_log.txt", "w") as f:
            f.write(f"程序崩溃: {e}\n")
            import traceback
            f.write(traceback.format_exc())
        
        # 也尝试使用日志记录
        try:
            logging.error(f"程序崩溃: {e}")
            logging.error(traceback.format_exc())
        except:
            pass  # 如果日志系统也失败了，至少我们有crash_log.txt
            
        input("程序发生错误，按任意键退出...")



