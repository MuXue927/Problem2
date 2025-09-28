示例目录 (examples)

本目录包含并行/并发示例，演示如何在 Windows 环境下使用 Python 的 multiprocessing 与 concurrent.futures。

Files:
- multiprocessing_example.py - 使用 multiprocessing.Process、Pool、Queue，展示 mp.cpu_count() 与进程间通信。
- concurrent_futures_example.py - 使用 concurrent.futures 的 ThreadPoolExecutor 与 ProcessPoolExecutor，展示 map、submit、as_completed 的不同用法。

运行示例（在项目根 d:\Gurobi_code\Problem2 下运行）：

```powershell
python .\examples\multiprocessing_example.py
python .\examples\concurrent_futures_example.py
```

注意（Windows）:
- 使用 multiprocessing 时请确保可创建进程的代码放在 if __name__ == '__main__' 下。
- 在容器或受限环境下，mp.cpu_count() 可能不反映容器配额。可结合 os.sched_getaffinity(0) 或 psutil 来获得更准确的可用核数。

小提示：如果你希望我把这些示例集成到测试或 make 命令中，我可以继续添加任务或 tests。
