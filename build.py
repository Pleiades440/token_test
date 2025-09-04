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
    
    # # 创建临时目录，只包含需要的数据文件
    # temp_data_dir = os.path.join(project_root, 'temp_data')
    # if os.path.exists(temp_data_dir):
    #     safe_remove(temp_data_dir)
    # os.makedirs(temp_data_dir)
    
    # # 复制mmlu_dev文件夹到临时目录
    # mmlu_dev_src = os.path.join(data_dir, 'mmlu_dev')
    # mmlu_dev_dst = os.path.join(temp_data_dir, 'mmlu_dev')
    # if os.path.exists(mmlu_dev_src):
    #     shutil.copytree(mmlu_dev_src, mmlu_dev_dst)
    #     print("已复制mmlu_dev数据文件到临时目录")
    # else:
    #     print(f"警告: mmlu_dev目录不存在: {mmlu_dev_src}")
    
    # 使用PyInstaller打包（文件夹模式）
    cmd = [
        'pyinstaller',
        '--add-data', f'{data_dir};data',
        '--add-data', f'{models_dir};models',
        '--add-data', 'src/config.yaml;.',
        '--hidden-import', 'yaml',
        '--hidden-import', 'tqdm',
        '--hidden-import', 'tokenizers',
        '--hidden-import', 'requests',
        '--hidden-import', 'urllib3',
        '--hidden-import', 'chardet',
        '--hidden-import', 'idna',
        '--collect-all', 'tokenizers',
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
        
        # 清理临时数据目录
        safe_remove(temp_data_dir)
        
        # 确保配置文件在正确的位置
        dist_config_path = os.path.join(dist_dir, 'TokenAnalyzer', 'config.yaml')
        if os.path.exists('src/config.yaml') and not os.path.exists(dist_config_path):
            shutil.copy2('src/config.yaml', dist_config_path)
            print("配置文件已复制到dist目录")
        
        # 修改代码中的资源路径获取方式
        # 更新run.py中的resource_path函数
        run_py_path = os.path.join(dist_dir, 'TokenAnalyzer', 'run.py')
        if os.path.exists(run_py_path):
            with open(run_py_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 修改resource_path函数，优先检查当前目录下的data和models文件夹
            new_resource_path = '''
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
'''
            
            # 替换resource_path函数
            import re
            old_pattern = r'def resource_path\(relative_path\):\s*""".*?"""\s*.*?return os\.path\.join\(base_path, relative_path\)'
            content = re.sub(old_pattern, new_resource_path, content, flags=re.DOTALL)
            
            with open(run_py_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("已修改run.py中的资源路径获取方式")
        
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
        # 清理临时数据目录
        safe_remove(temp_data_dir)

if __name__ == '__main__':
    start_time = time.time()
    build_exe()
    end_time = time.time()
    print(f"Build process took {end_time - start_time:.2f} seconds")