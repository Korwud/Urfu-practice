import torch

def tensors_creation():
    # Тензор размером 3x4, заполненный случайными числами от 0 до 1
    rand_tensor = torch.rand((3, 4))
    print(rand_tensor)

    # Тензор размером 2x3x4, заполненный нулями
    zeros_tensor = torch.zeros((2, 3, 4))
    print(zeros_tensor)

    # Тензор размером 5x5, заполненный единицами
    ones_tensor = torch.ones((5, 5))
    print(ones_tensor)

    # Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
    reshape_tensor = torch.arange(16)
    reshape_tensor = reshape_tensor.reshape((4, 4))
    print(reshape_tensor)


def operations_with_tensors():
    A = torch.rand((3, 4))
    B = torch.rand((4, 3))

    # Транспонирование тензора A
    transposed = A.T
    print(transposed)

    # Матричное умножение A и B
    matrix_multiplication = A @ B
    print(matrix_multiplication)

    # Поэлементное умножение A и транспонированного B
    multiplication = A * B.T
    print(multiplication)

    # Вычислите сумму всех элементов тензора A
    tensor_sum = A.sum()
    print(tensor_sum)


def indexing_and_slicing():
    tensor = torch.randint(0, 10, (5, 5, 5))

    # Первую строку
    first_row = tensor[0, :, :]
    print(first_row)

    # Последний столбец
    last_column = tensor[:, :, -1]
    print(last_column)

    # Подматрицу размером 2x2 из центра тензора
    submatrix = tensor[2, 2:4, 2:4]
    print(submatrix)

    # Все элементы с четными индексами
    even_elements = tensor[::2, ::2, ::2]
    print(even_elements)


def work_with_forms():
    tensor = torch.arange(24)

    # 2x12
    tensor1 = tensor.reshape((2, 12))
    print(tensor1)

    # 3x8
    tensor2 = tensor.reshape((3, 8))
    print(tensor2)

    # 4x6
    tensor3 = tensor.reshape((4, 6))
    print(tensor3)

    # 2x3x4
    tensor4 = tensor.reshape((2, 3, 4))
    print(tensor4)

    # 2x2x2x3
    tensor5 = tensor.reshape((2, 2, 2, 3))
    print(tensor5)

tensors_creation()
operations_with_tensors()
indexing_and_slicing()
work_with_forms()