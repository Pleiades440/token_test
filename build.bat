@echo off
chcp 65001 > nul
echo 正在安装必要的依赖...
pip install transformers datasets pandas tqdm pyyaml pyarrow pyinstaller

echo 正在打包应用...
python build.py

if exist dist\TokenAnalyzer.exe (
    echo 打包成功! 请查看dist目录中的TokenAnalyzer.exe
) else (
    echo 打包失败，请查看上面的错误信息
)

pause