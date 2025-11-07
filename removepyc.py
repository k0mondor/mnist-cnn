import shutil

# 删除 __pycache__ 文件夹
if os.path.exists('__pycache__'):
    shutil.rmtree('__pycache__')
    print("Python cache removed.")
