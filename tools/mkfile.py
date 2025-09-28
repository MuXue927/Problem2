import os

def create_txt_file(base_path='./', prefix='dataset_',  file_name='Windows PowerShell.txt', count=30):
    """
        在指定文件夹下创建文本文件

        参数:
        base_path (str): 基础路径，默认为当前目录
        prefix (str): 文件夹名称前缀
        file_name (str): 文本文件的名称
        count (int): 当前目录下，文件夹的总数量
    """
    # 确保基础路径存在
    os.makedirs(base_path, exist_ok=True)

    for i in range(1, count + 1):  # 遍历当前目录下的所有文件夹
        folder_name = f"{prefix}{i}"
        # 当前文件夹的完整路径
        full_path = os.path.join(base_path, folder_name)
        # 要创建的文本文件的完整路径
        file_path = os.path.join(full_path, file_name)
        if not os.path.exists(file_path):
            file = open(file_path, 'w', encoding='utf-8')
            file.close()
            print(f"在文件夹 {full_path} 中成功创建文本文件: {file_name}")
        else:
            print(f"在文件夹 {full_path} 中已经存在文本文件: {file_name}")


if __name__ == "__main__":
    par_dir = os.path.join('logs-cg', 'multi-periods', 'medium')
    create_txt_file(base_path=par_dir)
