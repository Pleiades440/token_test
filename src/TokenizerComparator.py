import os
import sys
import yaml
from transformers import AutoTokenizer
import datasets
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pyarrow as pa
import time
import logging
import traceback
import ijson
import json

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

def resource_path(relative_path):
    """获取资源的绝对路径"""
    try:
        # PyInstaller创建临时文件夹，将路径存储在_MEIPASS中
        base_path = sys._MEIPASS
    except Exception:
        # 如果不是打包环境，或者需要访问外部文件，使用当前工作目录
        base_path = os.getcwd()
    
    path = os.path.join(base_path, relative_path)
    return path

def is_frozen():
    """检查是否在打包环境中运行"""
    return getattr(sys, 'frozen', False)

class TokenizerComparator:
    def __init__(self, config_path="config.yaml"):
        # 先初始化logger
        self.logger = logging.getLogger(__name__)
        
        # 处理配置文件路径
        if is_frozen():
            # 如果是打包环境，使用资源路径
            config_path = resource_path(config_path)
        elif not os.path.isabs(config_path):
            # 如果不是打包环境，确保使用绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, config_path)
        
        self.tokenizers = {}
        self.load_config(config_path)
    
    def load_config(self, config_path):
        """从YAML文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # 数据集配置
            self.dataset_options = config.get('datasets', [])
            
            # 模型配置
            self.model_options = config.get('models', [])
            
            self.logger.info("配置加载成功!")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    

    def _load_local_dataset(self, path):
        """从本地路径加载数据集 - 简化版，支持JSONL"""
        try:
            # 处理路径（保持原样）
            if is_frozen():
                path = resource_path(path)
            elif not os.path.isabs(path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(current_dir, path)

            path = os.path.normpath(path)
            self.logger.info(f"尝试加载数据集: {path}")

            if not os.path.exists(path):
                self.logger.error(f"路径不存在: {path}")
                return None

            texts = []  # 在外部定义 texts，确保所有分支都能访问
            
            # 处理 JSONL 文件
            if os.path.isfile(path) and path.endswith('.jsonl'):
                self.logger.info(f"检测到JSONL文件: {path}")
                line_count = 0
                error_count = 0
                
                try:
                    # 使用文本模式打开文件，逐行读取
                    with open(path, 'r', encoding='utf-8') as file:
                        for line in file:
                            line_count += 1
                            line = line.strip()
                            if not line:  # 跳过空行
                                continue
                                
                            try:
                                # 直接使用json.loads解析每行
                                obj = json.loads(line)
                                # 尝试从对象中提取文本
                                extracted_text = self._extract_text_from_json(obj)
                                if extracted_text:
                                    texts.append(extracted_text)
                                else:
                                    # 如果提取失败，记录一条警告并将整个对象转换为字符串作为后备
                                    self.logger.warning(f"第{line_count}行: 从JSON对象中提取文本失败，使用后备方案: {str(obj)[:200]}...")
                                    texts.append(str(obj))
                                    
                            except json.JSONDecodeError as e:
                                error_count += 1
                                # 记录错误但继续处理其他行
                                self.logger.warning(f"第{line_count}行: JSON解析错误: {e}")
                                # 将原始行作为文本添加
                                texts.append(line)
                                
                except Exception as e:
                    self.logger.error(f"读取JSONL文件时出错: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    return None

                self.logger.info(f"从JSONL文件中成功加载 {len(texts)} 个文本样本，共处理 {line_count} 行，遇到 {error_count} 个错误")
                return texts

            # 处理普通JSON文件
            elif os.path.isfile(path) and path.endswith('.json'):
                self.logger.info("处理标准JSON格式文件")
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 记录数据结构以帮助调试
                self.logger.info(f"JSON数据结构: {type(data)}")
                if isinstance(data, list) and len(data) > 0:
                    self.logger.info(f"第一个元素的类型: {type(data[0])}")
                    if isinstance(data[0], dict):
                        self.logger.info(f"第一个元素的键: {list(data[0].keys())}")
                
                # 处理数组格式的JSON
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if i < 3:  # 记录前几个元素的信息
                            extracted = self._extract_text_from_json(item)
                            self.logger.info(f"第{i+1}个元素提取的文本: {extracted[:100] if extracted else '无文本提取'}...")
                        
                        text = self._extract_text_from_json(item)
                        if text:
                            texts.append(text)
                # 处理对象格式的JSON
                elif isinstance(data, dict):
                    text = self._extract_text_from_json(data)
                    if text:
                        texts.append(text)
                else:
                    self.logger.error(f"不支持的JSON格式: {type(data)}")
                    return None
                
                self.logger.info(f"成功加载 {len(texts)} 个文本样本")
                return texts
                
            # 处理其他文件格式（如TXT）
            elif os.path.isfile(path) and path.endswith('.txt'):
                self.logger.info(f"处理文本文件: {path}")
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        texts = f.read().splitlines()
                    self.logger.info(f"从文本文件中成功加载 {len(texts)} 行文本")
                    return texts
                except Exception as e:
                    self.logger.error(f"读取文本文件时出错: {e}")
                    return None
                    
            else:
                self.logger.error(f"不支持的文件格式: {path}")
                return None
                
        except Exception as e:
            self.logger.error(f"加载本地数据集时出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    

    def _extract_text_from_json(self, item):
        """从JSON对象中提取文本内容"""
        if isinstance(item, dict):
            # 处理问答格式的数据 (instruction + input + output)
            if all(key in item for key in ['instruction', 'input', 'output']):
                # 组合instruction、input和output字段
                parts = []
                
                # 添加instruction
                if item.get('instruction'):
                    parts.append(str(item['instruction']))
                
                # 添加input (如果有内容)
                if item.get('input'):
                    parts.append(str(item['input']))
                
                # 添加output (如果有内容)
                if item.get('output'):
                    parts.append(str(item['output']))
                
                if parts:
                    return "\n".join(parts)
            
            # 原有的其他提取逻辑保持不变
            text_fields = ['text', 'content', 'article', 'body', 'question', 'input', 'output', 
                        'prompt', 'context', 'answer', 'choices', 'option', 'options', 'instruction']
            
            for field in text_fields:
                if field in item and item[field]:
                    value = item[field]
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, list):
                        # 如果是列表，尝试连接所有字符串元素
                        text_parts = []
                        for part in value:
                            if isinstance(part, str):
                                text_parts.append(part)
                            elif isinstance(part, dict):
                                # 如果是字典，递归提取
                                nested_text = self._extract_text_from_json(part)
                                if nested_text:
                                    text_parts.append(nested_text)
                        if text_parts:
                            return " ".join(text_parts)
            
            # 如果没有找到标准字段，尝试所有字符串值
            for key, value in item.items():
                if isinstance(value, str) and value:
                    return value
                elif isinstance(value, list) and value and isinstance(value[0], str):
                    # 如果是字符串列表，连接它们
                    return " ".join(value)
                    
            # 如果还没有找到，尝试嵌套查找
            for value in item.values():
                if isinstance(value, dict):
                    nested_text = self._extract_text_from_json(value)
                    if nested_text:
                        return nested_text
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    # 如果是字典列表，递归处理每个字典
                    text_parts = []
                    for sub_item in value:
                        nested_text = self._extract_text_from_json(sub_item)
                        if nested_text:
                            text_parts.append(nested_text)
                    if text_parts:
                        return " ".join(text_parts)
                        
        elif isinstance(item, str) and item:
            return item
        elif isinstance(item, list) and item and isinstance(item[0], str):
            # 如果是字符串列表，连接它们
            return " ".join(item)
            
        return None
    
    def _load_dataset_online(self, dataset_info):
        """从Hugging Face加载在线数据集"""
        try:
            dataset_name = dataset_info['name']
            config_name = dataset_info.get('config', None)
            split = dataset_info.get('split', 'train')
            
            print(f"从Hugging Face加载数据集: {dataset_name}")
            
            # 加载数据集
            if config_name:
                dataset = datasets.load_dataset(dataset_name, config_name, split=split)
            else:
                dataset = datasets.load_dataset(dataset_name, split=split)
            
            # 尝试获取文本列名
            text_columns = [col for col in dataset.column_names if col in ['text', 'content', 'article']]
            if text_columns:
                return dataset[text_columns[0]]
            else:
                # 如果没有找到标准文本列，返回第一列
                return dataset[dataset.column_names[0]]
                
        except Exception as e:
            print(f"加载在线数据集时出错: {e}")
            return None
    
    def load_dataset_texts(self, dataset_index):
        """加载选定的数据集并返回文本列表"""
        if dataset_index < 0 or dataset_index >= len(self.dataset_options):
            self.logger.error(f"无效的数据集选择: {dataset_index}")
            return None
            
        dataset_info = self.dataset_options[dataset_index]
        self.logger.info(f"正在加载数据集: {dataset_info['name']}")
        
        # 检查是否有本地路径
        if "local_path" in dataset_info:
            local_path = dataset_info["local_path"]
            
            # 处理路径 - 使用资源路径函数
            abs_local_path = resource_path(local_path)
            self.logger.info(f"数据集本地路径: {abs_local_path}")
            
            # 检查路径是否存在
            if os.path.exists(abs_local_path):
                self.logger.info("从本地路径加载数据集...")
                return self._load_local_dataset(abs_local_path)
            else:
                self.logger.warning(f"数据集本地路径不存在: {abs_local_path}")
                # 如果本地路径不存在，尝试在线加载
                if "hf_path" in dataset_info:
                    self.logger.info("尝试从Hugging Face加载数据集...")
                    return self._load_dataset_online(dataset_info)
                else:
                    self.logger.error("无法加载数据集: 本地路径不存在且未提供在线路径")
                    return None
        elif "hf_path" in dataset_info:
            # 在线加载
            return self._load_dataset_online(dataset_info)
        else:
            self.logger.error("数据集配置错误: 既没有本地路径也没有在线路径")
            return None
        
    
    # 修改load_tokenizers方法，添加更多错误处理和日志
    def load_tokenizers(self, selected_model_indices):
        """加载选定的tokenizer"""
        self.logger.info("正在加载tokenizer...")
        
        successful_models = []
        
        for model_index in selected_model_indices:
            if model_index < 0 or model_index >= len(self.model_options):
                continue
                
            model_info = self.model_options[model_index]
            
            # 跳过"全部模型"选项
            if model_info.get("is_all_option", False):
                continue
                
            # 跳过需要认证但用户未登录的模型
            if model_info.get('auth_required', False):
                self.logger.info(f"跳过 {model_info['name']} (需要认证)")
                continue
                
            try:
                # 优先尝试本地路径
                if "local_path" in model_info:
                    local_path = model_info["local_path"]
                    
                    # 处理路径
                    if is_frozen() and not os.path.isabs(local_path):
                        local_path = resource_path(local_path)
                    
                    if os.path.exists(local_path):
                        self.logger.info(f"从本地加载 {model_info['name']} 的tokenizer...")
                        tokenizer = AutoTokenizer.from_pretrained(
                            local_path,
                            trust_remote_code=model_info.get("trust_remote_code", False)
                        )
                    else:
                        # 在线加载
                        self.logger.info(f"从网络加载 {model_info['name']} 的tokenizer...")
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_info['model_name'],
                            trust_remote_code=model_info.get('trust_remote_code', False)
                        )
                else:
                    # 在线加载
                    self.logger.info(f"从网络加载 {model_info['name']} 的tokenizer...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_info['model_name'],
                        trust_remote_code=model_info.get('trust_remote_code', False)
                    )
                
                # 确保所有tokenizer都有pad_token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"
                
                # 处理ChatGLM的特殊情况
                if model_info['name'] == "智谱 (ChatGLM)":
                    if not hasattr(tokenizer, 'vocab_size'):
                        if hasattr(tokenizer, 'sp_model'):
                            tokenizer.vocab_size = tokenizer.sp_model.get_piece_size()
                        elif hasattr(tokenizer, 'vocab'):
                            tokenizer.vocab_size = len(tokenizer.vocab)
                        else:
                            tokenizer.vocab_size = 65024
                            self.logger.warning(f"无法确定 {model_info['name']} 的词汇表大小，使用默认值 {tokenizer.vocab_size}")
                
                tokenizer.model_name = model_info['name']
                self.tokenizers[model_index] = tokenizer
                successful_models.append(model_index)
                self.logger.info(f"已加载 {model_info['name']} 的tokenizer")
            
            except Exception as e:
                error_msg = f"加载 {model_info['name']} 的tokenizer时出错: {e}"
                self.logger.error(error_msg)
                # 提供更详细的错误信息
                import traceback
                self.logger.error(traceback.format_exc())
                # 提供更具体的错误信息
                if "file" in str(e).lower() or "path" in str(e).lower():
                    self.logger.error("这可能是路径问题，请检查模型文件是否完整")
        
        return successful_models
    
    def process_dataset(self, dataset_texts, selected_model_indices):
        """处理数据集并比较tokenization结果"""
        if not dataset_texts:
            print("数据集为空!")
            return None
        
        total_bytes = sum(len(text.encode('utf-8')) for text in dataset_texts)
        sample_count = len(dataset_texts)
        
        print(f"需要处理 {total_bytes} 字节的文本，共 {sample_count} 个样本")
        
        # 初始化结果字典
        results = {
            'total_bytes': total_bytes,
            'sample_count': sample_count,
            'models': {}
        }
        
        # 为每个模型初始化计数器
        for model_index in selected_model_indices:
            if model_index in self.tokenizers:
                model_name = self.tokenizers[model_index].model_name
                results['models'][model_name] = {
                    'total_tokens': 0,
                    'error_count': 0
                }
        
        # 使用tqdm显示进度条
        print("开始处理数据，请稍候...")
        with tqdm(total=total_bytes, unit='B', unit_scale=True, desc="处理进度") as progress_bar:
            # 处理每个文本样本
            for text in dataset_texts:
                text_bytes = len(text.encode('utf-8'))
                
                # 对每个模型进行tokenization
                for model_index in selected_model_indices:
                    if model_index in self.tokenizers:
                        model_name = self.tokenizers[model_index].model_name
                        
                        try:
                            tokens = self.tokenizers[model_index].encode(text, truncation=False)
                            results['models'][model_name]['total_tokens'] += len(tokens)
                        except Exception as e:
                            results['models'][model_name]['error_count'] += 1
                            print(f"处理文本时出错 ({model_name}): {e}")
                
                # 更新进度条
                progress_bar.update(text_bytes)
        
        return results
    
    def display_results(self, results):
        """显示比较结果"""
        print("\n" + "="*60)
        print("               Tokenizer 比较结果")
        print("="*60)
        
        print(f"总字节数: {results['total_bytes']:,}")
        print(f"样本数量: {results['sample_count']:,}")
        print("-"*60)
        
        # 计算并显示每个模型的结果
        for model_name, model_results in results['models'].items():
            total_tokens = model_results['total_tokens']
            compression_ratio = results['total_bytes'] / total_tokens if total_tokens > 0 else 0
            error_count = model_results['error_count']
            
            print(f"{model_name}:")
            print(f"  总token数: {total_tokens:,}")
            print(f"  压缩比率: {compression_ratio:.2f} 字节/token")
            print(f"  错误数量: {error_count}")
            print()
    
    def run(self):
        """运行比较工具"""
        print("="*60)
        print("       多模型Tokenizer对比分析工具")
        print("="*60)
        
        # 选择数据集
        print("\n请选择数据集:")
        for i, dataset in enumerate(self.dataset_options, 1):
            print(f"{i}. {dataset['name']}")
        
        try:
            dataset_choice_idx = int(input("\n请输入数据集编号: ")) - 1
            if dataset_choice_idx < 0 or dataset_choice_idx >= len(self.dataset_options):
                print("无效的选择!")
                return
        except ValueError:
            print("请输入有效的数字!")
            return
        
        # 选择模型
        print("\n请选择要比较的模型 (可多选，用逗号分隔):")
        for i, model in enumerate(self.model_options, 1):
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
                if 0 <= choice_idx < len(self.model_options):
                    selected_model_indices.append(choice_idx)
            
            # 如果选择了"全部模型"，则包含所有模型
            all_model_idx = next((i for i, m in enumerate(self.model_options) if m.get('is_all_option', False)), -1)
            if all_model_idx != -1 and (all_model_idx + 1) in [int(c.strip()) for c in model_choices]:
                selected_model_indices = [i for i, m in enumerate(self.model_options) if not m.get('is_all_option', False)]
                print("已选择所有模型")
        except ValueError:
            print("请输入有效的数字!")
            return
        
        if not selected_model_indices:
            print("未选择有效模型!")
            return
        
        # 加载数据集文本
        dataset_texts = self.load_dataset_texts(dataset_choice_idx)
        if not dataset_texts:
            print("无法加载数据集!")
            return
        
        # 加载tokenizer
        successful_models = self.load_tokenizers(selected_model_indices)
        if not successful_models:
            print("未能加载任何tokenizer!")
            return
        
        # 处理数据集并比较结果
        results = self.process_dataset(dataset_texts, successful_models)
        
        if results:
            # 显示结果
            self.display_results(results)
        else:
            print("处理数据集时出错!")

# 运行程序
if __name__ == "__main__":
    # 检查必要的库是否已安装
    try:
        import transformers
        import datasets
        import pandas
        import yaml
        import pyarrow
        from tqdm import tqdm
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("请使用以下命令安装: pip install transformers datasets pandas tqdm pyyaml pyarrow")
        sys.exit(1)
    
    # 使用配置文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在")
        sys.exit(1)
    
    comparator = TokenizerComparator(config_path)
    comparator.run()