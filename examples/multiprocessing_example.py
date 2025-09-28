"""
multiprocessing_example.py
演示 Python 的 multiprocessing 模块的基本用法（适用于 Windows）。
包含：
- mp.cpu_count() 的使用
- 使用 Process 启动子进程
- 使用 Pool 并行处理（含 __main__ 保护）
- 使用 Queue 在进程间传递结果

运行（在项目根目录下）：
python examples\multiprocessing_example.py

注意（Windows）：必须把可创建进程的代码放在
if __name__ == '__main__':
    ...
之下，否则会产生无限递归。
"""
import multiprocessing as mp
import time


def worker_square(x, out_q=None):
    """示例工作函数：计算平方并把结果放入队列（如果提供）。"""
    pid = mp.current_process().pid
    time.sleep(0.1)  # 模拟工作
    res = (x, x * x, pid)
    if out_q is not None:
        out_q.put(res)
    return res


def demo_process_and_queue():
    """演示 Process 与 Queue 的基本用法"""
    print('\n--- demo: Process + Queue ---')
    q = mp.Queue()
    processes = []
    inputs = [1, 2, 3, 4]

    for i in inputs:
        p = mp.Process(target=worker_square, args=(i, q))
        p.start()
        processes.append(p)

    # 收集结果
    results = []
    for _ in processes:
        results.append(q.get())

    for p in processes:
        p.join()

    print('Results from processes (x, x^2, pid):', results)


def demo_pool_map():
    """演示 Pool.map 和 mp.cpu_count() 的使用"""
    print('\n--- demo: Pool.map ---')
    cpu_count = mp.cpu_count()
    print(f'Logical CPU count (mp.cpu_count()): {cpu_count}')

    # 池大小：保留一个给系统或限制为 cpu_count
    pool_size = max(1, cpu_count - 1)
    print(f'Using pool size = {pool_size}')

    with mp.Pool(processes=pool_size) as pool:
        inputs = list(range(10))
        start = time.time()
        results = pool.map(worker_square, inputs)
        end = time.time()

    print('Pool results (x, x^2, pid):')
    for r in results:
        print(r)
    print(f'Elapsed: {end - start:.3f}s')


if __name__ == '__main__':
    # Windows: 必须放在此保护下
    print('multiprocessing example, PID:', mp.current_process().pid)
    demo_process_and_queue()
    demo_pool_map()
    print('\nDone.')
