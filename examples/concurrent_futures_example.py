"""
concurrent_futures_example.py
演示 Python 的 concurrent.futures 模块（ThreadPoolExecutor 与 ProcessPoolExecutor）的基本用法。
包含：
- 使用 ThreadPoolExecutor 处理 I/O 绑定任务
- 使用 ProcessPoolExecutor 处理 CPU 绑定任务
- 使用 as_completed、map、submit + result

运行：
python examples\concurrent_futures_example.py

"""
import concurrent.futures as cf
import multiprocessing as mp
import time


def io_task(x):
    """模拟 I/O 绑定任务（睡眠）"""
    time.sleep(0.1)
    return (x, 'io', mp.current_process().pid)


def cpu_task(x):
    """模拟 CPU 密集型任务"""
    s = 0
    for i in range(1000000):
        s += (i % (x + 1))
    return (x, s, mp.current_process().pid)


def demo_threadpool():
    print('\n--- demo: ThreadPoolExecutor (I/O bound) ---')
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        inputs = list(range(10))
        results = list(ex.map(io_task, inputs))
    print('ThreadPool results:', results)


def demo_processpool():
    print('\n--- demo: ProcessPoolExecutor (CPU bound) ---')
    cpu_count = mp.cpu_count()
    workers = max(1, cpu_count - 1)
    print(f'Process workers: {workers}')

    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        inputs = list(range(1, 8))
        # 使用 submit + as_completed
        futures = [ex.submit(cpu_task, x) for x in inputs]
        for f in cf.as_completed(futures):
            try:
                print('Result from future:', f.result())
            except Exception as e:
                print('Future exception:', e)


if __name__ == '__main__':
    demo_threadpool()
    demo_processpool()
    print('\nDone.')
