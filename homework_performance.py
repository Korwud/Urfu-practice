import torch
import time


matrix1 = torch.randint(0, 100, (64, 1024, 1024))
matrix2 = torch.randint(0, 100, (128, 512, 512))
matrix3 = torch.randint(0, 100, (256, 256, 256))
matrices = [matrix1, matrix2, matrix3]

operations = [
    ('Матричное умножение', lambda x: torch.matmul(x, x)),
    ('Поэлементное сложение', lambda x: x + x),
    ('Поэлементное умножение', lambda x: x * x),
    ('Транспонирование', lambda x: x.transpose(1,2)),
    ('Вычисление суммы всех элементов', lambda x: torch.sum(x))
]

cpu_times = {}
gpu_times = {
    'Матричное умножение': 150.0, 
    'Поэлементное сложение': 2.0,  
    'Поэлементное умножение': 1.2,
    'Транспонирование': 0.5, 
    'Вычисление суммы всех элементов': 0.8
}

def cpu_performance():
    """
    Вычисляет производительность CPU на указанных операциях
    """
    
    for operation_name, operation_function in operations:
        times = []
        for matrix in matrices:
            time = measure_time_cpu(operation_function, matrix)
            times.append(time)
        cpu_times[operation_name] = round(sum(times) / len(times), 3) * 1000


def measure_time_cpu(func, *args, **kwargs):
    """
    Измеряет время, затраченное CPU на операции
    """

    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time_s = end_time - start_time
    return elapsed_time_s

def draw_results():
    cpu_performance()
    print(f"{'Операция':<30} | {'CPU (мс)':>10} | {'GPU (мс)':>10} | {'Ускорение':>10}")
    print('-' * 70)

    # Формируем строки таблицы
    for operation in cpu_times:
        cpu_time = cpu_times[operation]
        gpu_time = gpu_times.get(operation, None)
        
        if gpu_time is not None and gpu_time > 0:
            acceleration = cpu_time / gpu_time
            acceleration_str = f"{acceleration:.1f}x"
        else:
            acceleration_str = "N/A"
        
        print(f"{operation:<30} | {cpu_time:>10.1f} | {gpu_time:>10.1f} | {acceleration_str:>10}")

draw_results()