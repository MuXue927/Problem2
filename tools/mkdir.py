import os

def create_folder(base_path='./', prefix='dataset_', count=30):
    """
        在指定路径下创建带编号的多个文件夹

        参数:
        base_path (str): 基础路径，默认为当前目录
        prefix (str): 文件夹名称前缀
        count (int): 要创建的文件夹数量
    """
    # 确保基础路径存在
    os.makedirs(base_path, exist_ok=True)

    for i in range(1, count + 1):
        folder_name = f"{prefix}{i}"
        full_path = os.path.join(base_path, folder_name)
        try:
            os.mkdir(full_path)
            print(f"在路径 {base_path} 中成功创建文件夹: {folder_name}")
        except FileExistsError:
            print(f"在路径 {base_path} 中文件夹 {folder_name} 已存在")

if __name__ == "__main__":
    par_dir = os.path.join('logs-alns', 'multi-periods', 'medium')
    create_folder(base_path=par_dir)
