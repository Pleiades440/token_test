import yaml
import os

class TrainingTimeEstimator:
    def __init__(self, config_path="config.yaml"):
        # 加载配置文件
        self.config_path = config_path
        self.model_configs = None  # 延迟加载
        
        # 默认使用华为昇腾910B显卡
        self.default_gpu = "华为昇腾910B"
    
    def load_config(self):
        """从YAML文件加载配置（延迟加载）"""
        if self.model_configs is None:
            try:
                if not os.path.isabs(self.config_path):
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    config_path = os.path.join(current_dir, self.config_path)
                else:
                    config_path = self.config_path
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    
                self.model_configs = config.get('models', [])
            except Exception:
                self.model_configs = []
    
    def format_time(self, seconds):
        """将秒转换为天、小时、分钟、秒的格式"""
        if seconds <= 0:
            return "0秒"
        
        days = seconds // (24 * 3600)
        seconds %= (24 * 3600)
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        
        time_parts = []
        if days > 0:
            time_parts.append(f"{int(days)}天")
        if hours > 0:
            time_parts.append(f"{int(hours)}小时")
        if minutes > 0:
            time_parts.append(f"{int(minutes)}分钟")
        if seconds > 0 or len(time_parts) == 0:
            time_parts.append(f"{int(seconds)}秒")
        
        return " ".join(time_parts)
    
    def estimate_training_time(self, token_count, model_name, fine_tune_method, epochs):
        """估算训练时间"""
        # 确保配置已加载
        self.load_config()
        
        # 查找模型配置
        model_config = None
        for config in self.model_configs:
            if config.get('name') == model_name and not config.get('is_all_option', False):
                model_config = config
                break
        
        if not model_config:
            raise ValueError(f"找不到模型配置: {model_name}")
        
        # 获取指定微调方法的配置
        method_config = model_config.get(fine_tune_method, {})
        if not method_config:
            raise ValueError(f"模型 {model_name} 没有 {fine_tune_method} 微调配置")
        
        throughput = method_config.get('throughput', 0)
        world_size = method_config.get('world_size', 1)
        
        if throughput <= 0:
            raise ValueError(f"无效的吞吐量配置: {throughput}")
        
        # 计算总token处理量
        total_tokens = token_count * epochs
        
        # 计算训练时间 (秒)
        training_time_seconds = total_tokens / (throughput * world_size)
        
        return training_time_seconds, world_size, throughput
    
    def get_fine_tune_method(self):
        """获取用户选择的微调方法"""
        print("\n请选择微调方法:")
        print("1. 全参数微调")
        print("2. LoRA微调")
        
        while True:
            try:
                choice = int(input("请输入选项编号 (1 或 2): "))
                if choice == 1:
                    return "full", "全参数微调"
                elif choice == 2:
                    return "lora", "LoRA微调"
                else:
                    print("请输入 1 或 2")
            except ValueError:
                print("请输入有效的数字")
    
    def running_train(self, total_tokens_dict):
        """简化的训练时间估算函数，与run.py配合使用"""
        if not total_tokens_dict:
            print("没有可用的token数量数据!")
            return
        
        # 获取用户输入
        print("\n请配置训练参数:")
        
        # 选择微调方法
        method, method_display = self.get_fine_tune_method()
        
        # 获取epoch数
        while True:
            try:
                epochs = int(input("请输入训练epoch数: "))
                if epochs > 0:
                    break
                else:
                    print("epoch数必须大于0")
            except ValueError:
                print("请输入有效的整数")
        
        # 为每个模型估算训练时间
        print(f"\n使用 {self.default_gpu} 的训练时间估算结果 ({method_display}):")
        print("-" * 80)
        
        for model_name, total_tokens in total_tokens_dict.items():
            try:
                # 计算总训练token数
                total_training_tokens = total_tokens * epochs
                
                # 估算训练时间
                training_seconds, world_size, throughput = self.estimate_training_time(
                    total_tokens, model_name, method, epochs
                )
                
                formatted_time = self.format_time(training_seconds)
                print(f"{model_name:20s}: {formatted_time} "
                      f"(GPU数量: {world_size}, 吞吐量: {throughput} tokens/sec/GPU, "
                      f"总训练token数: {total_training_tokens:,})")
                
            except Exception as e:
                print(f"{model_name:20s}: 估算错误 - {e}")
        
        print("-" * 80)
    
    def run(self):
        """独立运行估算工具"""
        # 确保配置已加载
        self.load_config()
        
        print("=" * 60)
        print("          深度学习训练时长估算工具")
        print("=" * 60)
        print("注意: 此估算基于配置文件中的吞吐量和GPU数量")
        print("=" * 60)
        
        # 获取token数量
        while True:
            try:
                token_count = float(input("请输入训练数据的token数量: "))
                if token_count > 0:
                    break
                else:
                    print("token数量必须大于0")
            except ValueError:
                print("请输入有效的数字")
        
        # 显示可用模型
        print("\n可用模型:")
        valid_models = []
        for i, model in enumerate(self.model_configs, 1):
            if not model.get('is_all_option', False):
                print(f"{i}. {model['name']}")
                valid_models.append(model)
        
        # 选择模型
        while True:
            try:
                model_choice = int(input("\n请选择模型编号: "))
                if 1 <= model_choice <= len(valid_models):
                    selected_model = valid_models[model_choice - 1]
                    break
                else:
                    print(f"请输入1到{len(valid_models)}之间的数字")
            except ValueError:
                print("请输入有效的数字")
        
        # 选择微调方法
        method, method_display = self.get_fine_tune_method()
        
        # 获取epoch数
        while True:
            try:
                epochs = int(input("请输入训练epoch数: "))
                if epochs > 0:
                    break
                else:
                    print("epoch数必须大于0")
            except ValueError:
                print("请输入有效的整数")
        
        # 计算训练时间
        try:
            training_seconds, world_size, throughput = self.estimate_training_time(
                token_count, selected_model['name'], method, epochs
            )
            formatted_time = self.format_time(training_seconds)
            
            # 显示结果
            print("\n" + "=" * 60)
            print("训练估算结果:")
            print(f"模型: {selected_model['name']}")
            print(f"微调方法: {method_display}")
            print(f"GPU类型: {self.default_gpu}")
            print(f"GPU数量: {world_size}")
            print(f"吞吐量: {throughput} tokens/sec/GPU")
            print(f"Token数量: {token_count:,.0f}")
            print(f"Epoch数: {epochs}")
            print(f"预计训练时长: {formatted_time}")
            print("=" * 60)
            
        except ValueError as e:
            print(f"计算错误: {e}")

# 运行程序
if __name__ == "__main__":
    estimator = TrainingTimeEstimator()
    estimator.run()