import torch

def simple_calculations_with_grad():
    # Создайте тензоры x, y, z с requires_grad=True
    x = torch.tensor(3.0, requires_grad=True)
    y = torch.tensor(4.0, requires_grad=True)
    z = torch.tensor(5.0, requires_grad=True)

    # Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
    f = x**2 + y**2 + z**2 + 2 * x * y * z

    # Найдите градиенты по всем переменным
    f.backward()

    print("Градиент по x:", x.grad)
    print("Градиент по y:", y.grad)
    print("Градиент по z:", z.grad)

    # Проверьте результат аналитически
    # df/dx = 2x + 2yz = 2 * 3 + 2 * 4 * 5 = 46
    # df/dy = 2y + 2xz = 2 * 4 + 2 * 3 * 5 = 38
    # df/dz = 2z + 2xy = 2 * 5 + 2 * 3 * 4 = 34


def loss_function_grad():
    # Реализуйте функцию MSE (Mean Squared Error):
    # MSE = (1/n) * Σ(y_pred - y_true)^2
    # где y_pred = w * x + b (линейная функция)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])

    w = torch.rand(1, requires_grad=True)
    b = torch.rand(1, requires_grad=True)

    y_pred = w * x + b

    n = x.shape[0]
    mse = (1 / n) * torch.sum((y_pred - y_true) ** 2)

    mse.backward()

    # Найдите градиенты по w и b
    print("Градиент по w:", w.grad.item())
    print("Градиент по b:", b.grad.item())


def chain_rule():
    # Реализуйте составную функцию: f(x) = sin(x^2 + 1)
    x = torch.tensor(4.5, requires_grad=True)
    f = torch.sin(x**2 + 1)

    # Найдите градиент df/dx
    f.backward(retain_graph=True)
    print("Градиент по x:", x.grad)

    # Проверьте результат с помощью torch.autograd.grad
    grad_x, = torch.autograd.grad(f, x)
    print("Градиент по x с autograd:", grad_x)

#simple_calculations_with_grad()
#loss_function_grad()
chain_rule()