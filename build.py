import os
import sys
import subprocess
import shutil
import time
import stat

def remove_readonly(func, path, excinfo):
    """处理只读文件的删除"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def safe_remove(path):
    """安全删除文件或目录"""
    if os.path.exists(path):
        if os.path.isfile(path):
            try:
                os.remove(path)
            except PermissionError:
                # 如果是文件，尝试修改权限后删除
                os.chmod(path, stat.S_IWRITE)
                os.remove(path)
        else:
            # 如果是目录，使用特殊处理
            shutil.rmtree(path, onerror=remove_readonly)

def build_exe():
    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 清理旧文件 - 使用安全删除方法
    print("清理旧文件...")
    safe_remove('dist')
    safe_remove('build')
    safe_remove('TokenAnalyzer.spec')
    
    # 确保dist目录存在
    dist_dir = os.path.join(project_root, 'dist')
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    
    # 检查数据目录是否存在
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, 'models')
    
    if not os.path.exists(data_dir):
        print(f"警告: 数据目录不存在: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(models_dir):
        print(f"警告: 模型目录不存在: {models_dir}")
        os.makedirs(models_dir, exist_ok=True)
    
    # 使用PyInstaller打包（文件夹模式）
    cmd = [
        'pyinstaller',
        '--add-data', 'data;data',
        '--add-data', 'models;models',
        '--add-data', 'src/config.yaml;.',
        '--hidden-import', 'transformers.models.auto',
        '--hidden-import', 'transformers.models.gpt2',
        '--hidden-import', 'transformers.models.bert',
        '--hidden-import', 'transformers.tokenization_utils',
        '--hidden-import', 'transformers.file_utils',
        '--hidden-import', 'datasets',
        '--hidden-import', 'pyarrow',
        '--hidden-import', 'yaml',
        '--hidden-import', 'torch',
        '--hidden-import', 'torch._C',
        '--hidden-import', 'torch._VF',
        '--hidden-import', 'torch.distributed',
        '--hidden-import', 'torch.distributed.rpc',
        '--hidden-import', 'torch.distributed.optim',
        '--hidden-import', 'torch.distributed.algorithms',
        '--hidden-import', 'torch.multiprocessing',
        '--hidden-import', 'torch._dynamo', 
        '--hidden-import', 'torch._inductor', 
        '--hidden-import', 'requests',
        '--hidden-import', 'urllib3',
        '--hidden-import', 'chardet',
        '--hidden-import', 'idna',
        '--hidden-import', 'numpy',
        '--hidden-import', 'tqdm',
        '--hidden-import', 'tokenizers',
        '--collect-all', 'tokenizers',
        '--collect-all', 'transformers',
        '--onedir',
        '--console',  # 改为使用控制台模式，避免stdin问题
        '--optimize', '2',
        '--name', 'TokenAnalyzer',
        'src/run.py'
    ]
    
    print("Building application...")
    print("Command:", ' '.join(cmd))
    
    # 切换到项目根目录执行命令
    os.chdir(project_root)
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Build successful!")
        
        # 复制所有数据文件
        if os.path.exists(data_dir):
            dist_data_dir = os.path.join(dist_dir, 'TokenAnalyzer', 'data')
            if os.path.exists(dist_data_dir):
                safe_remove(dist_data_dir)
            shutil.copytree(data_dir, dist_data_dir)
            print("数据文件已复制到dist目录")
        
        # 复制所有模型文件
        if os.path.exists(models_dir):
            dist_models_dir = os.path.join(dist_dir, 'TokenAnalyzer', 'models')
            if os.path.exists(dist_models_dir):
                safe_remove(dist_models_dir)
            shutil.copytree(models_dir, dist_models_dir)
            print("模型文件已复制到dist目录")
        
        # 确保配置文件在正确的位置
        dist_config_path = os.path.join(dist_dir, 'TokenAnalyzer', 'config.yaml')
        if os.path.exists('src/config.yaml') and not os.path.exists(dist_config_path):
            shutil.copy2('src/config.yaml', dist_config_path)
            print("配置文件已复制到dist目录")
        
        print("\n打包完成! 请查看dist/TokenAnalyzer目录")
        print("运行dist/TokenAnalyzer/TokenAnalyzer.exe来启动程序")
        
        # 创建批处理文件用于调试（即使使用控制台模式也保留）
        debug_bat = os.path.join(dist_dir, 'TokenAnalyzer', 'debug.bat')
        with open(debug_bat, 'w') as f:
            f.write('@echo off\n')
            f.write('echo 正在启动TokenAnalyzer...\n')
            f.write('echo 如果程序崩溃，此处会显示错误信息\n')
            f.write('echo.\n')
            f.write('TokenAnalyzer.exe\n')
            f.write('echo.\n')
            f.write('echo 程序已退出，按任意键关闭窗口...\n')
            f.write('pause >nul\n')
        print("已创建调试批处理文件: debug.bat")
        
    else:
        print("Build failed:")
        print(result.stdout)
        print(result.stderr)

if __name__ == '__main__':
    start_time = time.time()
    build_exe()
    end_time = time.time()
    print(f"Build process took {end_time - start_time:.2f} seconds")