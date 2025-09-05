import os
import sys
import subprocess
import shutil
import time
import stat
import re

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
    
    # 清理旧文件

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
    
    # 创建临时目录，只包含需要的数据文件
    temp_data_dir = os.path.join(project_root, 'temp_data')
    if os.path.exists(temp_data_dir):
        safe_remove(temp_data_dir)
    os.makedirs(temp_data_dir)
    
    # 复制mmlu_dev文件夹到临时目录
    mmlu_dev_src = os.path.join(data_dir, 'mmlu_dev')
    mmlu_dev_dst = os.path.join(temp_data_dir, 'mmlu_dev')
    if os.path.exists(mmlu_dev_src):
        shutil.copytree(mmlu_dev_src, mmlu_dev_dst)
        print("已复制mmlu_dev数据文件到临时目录")
    else:
        print(f"警告: mmlu_dev目录不存在: {mmlu_dev_src}")
    
    # 使用PyInstaller打包（单文件模式）
    cmd = [
        'pyinstaller',
        '--add-data', f'{temp_data_dir};data',
        '--add-data', f'{models_dir};models',
        '--add-data', 'src/config.yaml;.',
        '--hidden-import', 'yaml',
        '--hidden-import', 'tqdm',
        '--hidden-import', 'tokenizers',
        # 排除不必要的模块
        '--exclude-module', 'numpy',
        '--exclude-module', 'pandas',
        '--exclude-module', 'matplotlib',
        '--exclude-module', 'scipy',
        '--exclude-module', 'sklearn',
        '--exclude-module', 'PIL',
        '--exclude-module', 'torch',
        '--exclude-module', 'tensorflow',
        '--collect-all', 'tokenizers',
        '--onefile',  # 单文件模式
        '--console',  # 控制台模式
        '--optimize', '2',
        '--name', 'TokenAnalyzer',
        'src/run.py'
    ]
    
    # 获取 build.py 所在目录的绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建 UPX 目录路径
    upx_dir = os.path.join(base_dir, "upx")
    # 检查 UPX 可执行文件是否存在
    upx_exe = os.path.join(upx_dir, "upx.exe" if os.name == 'nt' else "upx")
    if os.path.exists(upx_exe):
        cmd.extend(['--upx-dir', upx_dir])
        print("使用 UPX 压缩")
    else:
        print(f"UPX 未找到于: {upx_exe}")
    
    print("Building application...")
    print("Command:", ' '.join(cmd))
    
    # 切换到项目根目录执行命令
    os.chdir(project_root)
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Build successful!")
        
        # 清理临时数据目录
        safe_remove(temp_data_dir)
        
        print("\n打包完成! 请查看dist目录中的TokenAnalyzer.exe")
        print("运行dist/TokenAnalyzer.exe来启动程序")
        
        # 创建批处理文件用于调试
        debug_bat = os.path.join(dist_dir, 'debug.bat')
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