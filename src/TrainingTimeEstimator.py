import math

class TrainingTimeEstimator:
    def __init__(self):
        # 显卡实际训练吞吐量 (tokens/sec/GPU)
        # 这些值基于实际基准测试和经验估计
        self.gpu_throughput = {
            "NVIDIA H100": 3500,    # 基于H100在Llama2 70B上的表现
            "NVIDIA A100": 2000,    # 基于A100的实际训练吞吐量
            "NVIDIA RTX 4090": 800, # 基于消费级GPU的实际表现
            "Huawei Ascend 910B": 2800, # 基于华为官方数据和基准测试
        }
        
        # 集群规模 (GPU数量)
        self.cluster_config = {
            "single": 1,
            "single_node": 8,
            "8_nodes": 64,
            "16_nodes": 128,
            "32_nodes": 256,
            "64_nodes": 512,
            "128_nodes": 1024
        }
        
        # 效率因子 (考虑通信开销、数据加载等)
        self.efficiency_factors = {
            "single": 0.95,      # 单卡几乎没有通信开销
            "single_node": 0.85, # 单节点内NVLink/PCIe通信
            "multi_node": 0.65   # 多节点间网络通信
        }
    
    def format_time(self, seconds):
        """将秒转换为天、小时、分钟、秒的格式"""
        if seconds <= 0:
            return "0秒"
        
        # 计算各个时间单位
        days = seconds // (24 * 3600)
        seconds %= (24 * 3600)
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        
        # 构建时间字符串
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
    
    def estimate_training_time(self, token_count, gpu_type, cluster_size, epochs):
        """估算训练时间"""
        # 获取GPU的吞吐量
        if gpu_type not in self.gpu_throughput:
            raise ValueError(f"不支持的GPU类型: {gpu_type}")
        
        gpu_tokens_per_sec = self.gpu_throughput[gpu_type]
        
        # 获取集群规模
        if cluster_size not in self.cluster_config:
            raise ValueError(f"不支持的集群规模: {cluster_size}")
        
        gpu_count = self.cluster_config[cluster_size]
        
        # 确定效率因子
        if gpu_count == 1:
            efficiency = self.efficiency_factors["single"]
        elif gpu_count <= 8:  # 单节点
            efficiency = self.efficiency_factors["single_node"]
        else:  # 多节点
            efficiency = self.efficiency_factors["multi_node"]
        
        # 计算总token处理量
        total_tokens = token_count * epochs
        
        # 计算训练时间 (秒)
        # 使用实际吞吐量而非理论FLOPs
        training_time_seconds = total_tokens / (gpu_tokens_per_sec * gpu_count * efficiency)
        
        return training_time_seconds
    
    def running_train(self, total_tokens_dict):
        """简化的训练时间估算函数，与run.py配合使用"""
        # 获取用户输入
        print("\n请配置训练参数:")
        
        # 显示GPU选项
        self.display_gpu_options()
        gpu_choice = self.get_gpu_choice()
        
        # 显示集群选项
        self.display_cluster_options()
        cluster_choice = self.get_cluster_choice()
        
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
        print("\n训练时间估算结果:")
        print("-" * 80)
        
        for model_name, total_tokens in total_tokens_dict.items():
            try:
                # 计算总训练token数
                total_training_tokens = total_tokens * epochs
                
                # 估算训练时间
                training_seconds = self.estimate_training_time(
                    total_training_tokens, gpu_choice, cluster_choice, 1
                )
                
                formatted_time = self.format_time(training_seconds)
                print(f"{model_name:20s}: {formatted_time} "
                      f"(总训练token数: {total_training_tokens:,})")
                
            except Exception as e:
                print(f"{model_name:20s}: 估算错误 - {e}")
        
        print("-" * 80)
    
    def display_gpu_options(self):
        """显示GPU选项"""
        print("请选择GPU类型:")
        print("-" * 50)
        
        gpus = list(self.gpu_throughput.keys())
        for i, gpu in enumerate(gpus, 1):
            print(f"{i}. {gpu} ({self.gpu_throughput[gpu]} tokens/sec/GPU)")
        
        print("-" * 50)
    
    def get_gpu_choice(self):
        """获取用户选择的GPU"""
        all_gpus = list(self.gpu_throughput.keys())
        
        while True:
            try:
                choice = int(input("请输入GPU编号: "))
                if 1 <= choice <= len(all_gpus):
                    return all_gpus[choice - 1]
                else:
                    print(f"请输入1到{len(all_gpus)}之间的数字")
            except ValueError:
                print("请输入有效的数字")
    
    def display_cluster_options(self):
        """显示集群选项"""
        print("\n请选择集群规模:")
        print("-" * 30)
        options = list(self.cluster_config.keys())
        for i, option in enumerate(options, 1):
            print(f"{i}. {option} ({self.cluster_config[option]}个GPU)")
        print("-" * 30)
    
    def get_cluster_choice(self):
        """获取用户选择的集群规模"""
        options = list(self.cluster_config.keys())
        
        while True:
            try:
                choice = int(input("请输入集群规模编号: "))
                if 1 <= choice <= len(options):
                    return options[choice - 1]
                else:
                    print(f"请输入1到{len(options)}之间的数字")
            except ValueError:
                print("请输入有效的数字")
    
    def run(self):
        """独立运行估算工具"""
        print("=" * 60)
        print("          深度学习训练时长估算工具")
        print("=" * 60)
        print("注意: 此估算基于实际基准测试和经验数据，")
        print("      考虑了通信开销和数据加载等因素")
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
        
        # 显示并选择GPU
        self.display_gpu_options()
        gpu_choice = self.get_gpu_choice()
        
        # 显示并选择集群规模
        self.display_cluster_options()
        cluster_choice = self.get_cluster_choice()
        
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
            training_seconds = self.estimate_training_time(
                token_count, gpu_choice, cluster_choice, epochs
            )
            formatted_time = self.format_time(training_seconds)
            
            # 显示结果
            print("\n" + "=" * 60)
            print("训练估算结果:")
            print(f"Token数量: {token_count:,.0f}")
            print(f"GPU类型: {gpu_choice}")
            print(f"集群规模: {cluster_choice} ({self.cluster_config[cluster_choice]}个GPU)")
            print(f"Epoch数: {epochs}")
            print(f"预计训练时长: {formatted_time}")
            
            # 额外显示每日处理token量
            daily_tokens = self.gpu_throughput[gpu_choice] * self.cluster_config[cluster_choice] * 86400
            if cluster_choice != "single":
                efficiency = self.efficiency_factors["multi_node"] if self.cluster_config[cluster_choice] > 8 else self.efficiency_factors["single_node"]
                daily_tokens *= efficiency
            print(f"每日处理token量: {daily_tokens:,.0f}")
            print("=" * 60)
            
        except ValueError as e:
            print(f"计算错误: {e}")

# 运行程序
if __name__ == "__main__":
    estimator = TrainingTimeEstimator()
    estimator.run()