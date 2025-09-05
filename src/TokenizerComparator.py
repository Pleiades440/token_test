import os
import sys
import yaml
from tokenizers import Tokenizer
import json
from tqdm import tqdm

def resource_path(relative_path):
    """获取资源的绝对路径 - 只使用内置资源"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def is_frozen():
    """检查是否在打包环境中运行"""
    return getattr(sys, 'frozen', False)

class TokenizerComparator:
    def __init__(self, config_path="config.yaml"):
        # 处理配置文件路径 - 只使用内置配置文件
        if is_frozen():
            config_path = resource_path(config_path)
        elif not os.path.isabs(config_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, config_path)
        
        self.config_path = config_path
        self.tokenizers = {}
        # 延迟加载配置
        self.dataset_options = []
        self.model_options = []
        self.default_dataset = None
    
    def load_config(self):
        """从YAML文件加载配置（延迟加载）"""
        if not self.dataset_options:  # 如果配置未加载，则加载
            try:
                # 确保配置文件存在
                if not os.path.exists(self.config_path):
                    print(f"配置文件不存在: {self.config_path}")
                    # 设置默认值
                    self.dataset_options = []
                    self.model_options = []
                    self.default_dataset = {
                        'name': 'Default Dataset',
                        'local_path': 'data/default_dataset.jsonl'
                    }
                    return
                
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    
                # 数据集配置 - 只保留第一个数据集
                self.dataset_options = config.get('datasets', [])
                if self.dataset_options:
                    self.default_dataset = self.dataset_options[0]
                else:
                    self.default_dataset = {
                        'name': 'Default Dataset',
                        'local_path': 'data/default_dataset.jsonl'
                    }
                
                # 模型配置
                self.model_options = config.get('models', [])
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                # 设置默认值
                self.dataset_options = []
                self.model_options = []
                self.default_dataset = {
                    'name': 'Default Dataset',
                    'local_path': 'data/default_dataset.jsonl'
                }
    
    def _load_local_dataset(self, path):
        """从本地路径加载数据集 - 简化版，支持JSONL"""
        # 处理路径 - 只使用内置资源
        if is_frozen():
            path = resource_path(path)
        elif not os.path.isabs(path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, path)

        path = os.path.normpath(path)

        if not os.path.exists(path):
            print(f"路径不存在: {path}")
            return None

        texts = []
        
        # 处理 JSONL 文件
        if os.path.isfile(path) and path.endswith('.jsonl'):
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                            
                        try:
                            obj = json.loads(line)
                            extracted_text = self._extract_text_from_json(obj)
                            if extracted_text:
                                texts.append(extracted_text)
                            else:
                                texts.append(str(obj))
                                
                        except json.JSONDecodeError:
                            texts.append(line)
                
            except Exception as e:
                print(f"加载JSONL文件失败: {e}")
                return None

            return texts

        # 处理普通JSON文件
        elif os.path.isfile(path) and path.endswith('.json'):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 处理数组格式的JSON
                if isinstance(data, list):
                    for item in data:
                        text = self._extract_text_from_json(item)
                        if text:
                            texts.append(text)
                # 处理对象格式的JSON
                elif isinstance(data, dict):
                    text = self._extract_text_from_json(data)
                    if text:
                        texts.append(text)
                else:
                    return None
                
                return texts
            except Exception as e:
                print(f"加载JSON文件失败: {e}")
                return None
            
        # 处理其他文件格式（如TXT）
        elif os.path.isfile(path) and path.endswith('.txt'):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    texts = f.read().splitlines()
                return texts
            except Exception as e:
                print(f"加载TXT文件失败: {e}")
                return None
                
        else:
            return None
    
    def _extract_text_from_json(self, item):
        """从JSON对象中提取文本内容"""
        if isinstance(item, dict):
            # 处理问答格式的数据 (instruction + input + output)
            if all(key in item for key in ['instruction', 'input', 'output']):
                parts = []
                
                if item.get('instruction'):
                    parts.append(str(item['instruction']))
                
                if item.get('input'):
                    parts.append(str(item['input']))
                
                if item.get('output'):
                    parts.append(str(item['output']))
                
                if parts:
                    return "\n".join(parts)
            
            text_fields = ['text', 'content', 'article', 'body', 'question', 'input', 'output', 
                        'prompt', 'context', 'answer', 'choices', 'option', 'options', 'instruction']
            
            for field in text_fields:
                if field in item and item[field]:
                    value = item[field]
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, list):
                        text_parts = []
                        for part in value:
                            if isinstance(part, str):
                                text_parts.append(part)
                            elif isinstance(part, dict):
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
                    return " ".join(value)
                    
            # 如果还没有找到，尝试嵌套查找
            for value in item.values():
                if isinstance(value, dict):
                    nested_text = self._extract_text_from_json(value)
                    if nested_text:
                        return nested_text
                elif isinstance(value, list) and value and isinstance(value[0], dict):
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
            return " ".join(item)
            
        return None
    
    def load_dataset_texts(self, dataset_index=0):
        """加载指定索引的数据集并返回文本列表"""
        # 确保配置已加载
        self.load_config()
        
        if dataset_index < 0 or dataset_index >= len(self.dataset_options):
            return None
        
        dataset_info = self.dataset_options[dataset_index]
        
        # 检查是否有本地路径
        if "local_path" in dataset_info:
            local_path = dataset_info["local_path"]
            
            # 处理路径 - 只使用内置资源
            abs_local_path = resource_path(local_path)
            
            # 检查路径是否存在
            if os.path.exists(abs_local_path):
                return self._load_local_dataset(abs_local_path)
        
        return None
    
    def load_tokenizers(self, selected_model_indices):
        """加载选定的 tokenizer (使用 tokenizers 库)"""
        # 确保配置已加载
        self.load_config()
        
        successful_models = []
        for model_index in selected_model_indices:
            if model_index < 0 or model_index >= len(self.model_options):
                continue
            model_info = self.model_options[model_index]
            # 跳过"全部模型"选项和需要认证的模型
            if model_info.get("is_all_option", False) or model_info.get('auth_required', False):
                continue
            try:
                tokenizer = None
                # 优先尝试本地路径
                if "local_path" in model_info:
                    local_path = model_info["local_path"]
                    if is_frozen() and not os.path.isabs(local_path):
                        local_path = resource_path(local_path)
                    # 检查本地是否存在 tokenizer.json
                    tokenizer_json_path = os.path.join(local_path, "tokenizer.json")
                    if os.path.exists(tokenizer_json_path):
                        tokenizer = Tokenizer.from_file(tokenizer_json_path)
                    else:
                        # 如果本地没有 tokenizer.json，尝试从 Hugging Face 下载
                        try:
                            if model_info.get('trust_remote_code', False):
                                print(f"警告: tokenizers 库可能不支持 trust_remote_code，尝试加载 {model_info['name']} 可能失败。")
                            tokenizer = Tokenizer.from_pretrained(model_info['model_name'])
                        except Exception as e:
                            print(f"从网络加载 {model_info['name']} 的 tokenizer 失败: {e}")
                            continue
                else:
                    # 没有本地路径，尝试在线加载
                    try:
                        tokenizer = Tokenizer.from_pretrained(model_info['model_name'])
                    except Exception as e:
                        print(f"从网络加载 {model_info['name']} 的 tokenizer 失败: {e}")
                        continue

                if tokenizer is None:
                    print(f"无法加载 {model_info['name']} 的 tokenizer")
                    continue

                tokenizer.model_name = model_info['name']
                self.tokenizers[model_index] = tokenizer
                successful_models.append(model_index)
            except Exception as e:
                print(f"加载 {model_info['name']} 的 tokenizer 时出错: {e}")
        return successful_models
    
    def process_dataset(self, dataset_texts, selected_model_indices):
        """处理数据集并比较 tokenization 结果 (使用 tokenizers 库)"""
        if not dataset_texts:
            return None
        total_bytes = sum(len(text.encode('utf-8')) for text in dataset_texts)
        sample_count = len(dataset_texts)
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
        with tqdm(total=total_bytes, unit='B', unit_scale=True, desc="处理进度") as progress_bar:
            for text in dataset_texts:
                text_bytes = len(text.encode('utf-8'))
                for model_index in selected_model_indices:
                    if model_index in self.tokenizers:
                        model_name = self.tokenizers[model_index].model_name
                        try:
                            # 使用 tokenizers 库的 encode 方法
                            encoding = self.tokenizers[model_index].encode(text)
                            results['models'][model_name]['total_tokens'] += len(encoding.tokens)
                        except Exception as e:
                            results['models'][model_name]['error_count'] += 1
                            print(f"处理文本时出错 ({model_name}): {e}")
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
        # 确保配置已加载
        self.load_config()
        
        print("="*60)
        print("       多模型Tokenizer对比分析工具")
        print("="*60)
        
        # 直接使用默认数据集，不再让用户选择
        print(f"\n使用默认数据集: {self.default_dataset['name']}")
        
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
        except ValueError:
            print("请输入有效的数字!")
            return
        
        if not selected_model_indices:
            print("未选择有效模型!")
            return
        
        # 加载数据集文本
        dataset_texts = self.load_dataset_texts(0)
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
            return results
            print()
        else:
            print("处理数据集时出错!")
            return None

# 运行程序
if __name__ == "__main__":
    # 检查必要的库是否已安装
    try:
        import yaml
        from tqdm import tqdm
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("请使用以下命令安装: pip install yaml tqdm")
        sys.exit(1)
    
    # 使用配置文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在")
        sys.exit(1)
    
    comparator = TokenizerComparator(config_path)
    comparator.run()