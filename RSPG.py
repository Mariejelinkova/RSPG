from abc import abstractmethod, ABC
import numpy as np
import copy
from itertools import product
from scipy.optimize import linprog, minimize
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt

class Function(ABC):
    def __init__(self, *args, **kwargs):
        self.projection_fn = lambda x: x

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass

    def estimate_gradient(self, x, ksi, m, sample_size=20, return_var=False):
        gradient_estimates = []
        eye = np.eye(x.shape[0])
        for _ in range(m):
            grad = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                eps = np.random.normal(0, ksi, (sample_size))
                est = [(self(x + eye[i] * h) - self(x)) / h for h in eps]

                grad[i] = np.mean(est, axis=0)[i]
            gradient_estimates.append(grad)

        if return_var:
            return np.mean(gradient_estimates, axis=0), np.var(gradient_estimates, axis=1)
        else:
            return np.mean(gradient_estimates, axis=0)

    def projection(self, x):
        return self.projection_fn(x)


class Quadratic(Function):
    U = np.linalg.qr(np.random.normal(0, 1, (1000, 100)))[0]
    b = np.random.normal(0, 1, 1000)
    def __call__(self, x):
        return (x ** 2) / 2

    def gradient(self, x):
        return x


class AffineConstrainedQuadratic(Quadratic):
    def __init__(self, U, b):
        super().__init__()
        self.projection_fn = lambda x: b + U.dot(U.T).dot(x - b)


class Farming(Function):
    def __init__(self):
        super().__init__()
        self.c = np.array([150, 230, 260, 238, 210, -170, -150, -36, -10])
        self.A = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [-2.5, 0, 0, -1, 0, 1, 0, 0, 0],
                           [0, -3, 0, 0, -1, 0, 1, 0, 0],
                           [0, 0, -20, 0, 0, 0, 0, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0]])
        self.b = np.array([500, -200, -240, 0, 6000])

        solution = linprog(self.c, A_ub=self.A, b_ub=self.b)
        self.optimal_val = solution['fun']
        self.optimal_x = solution['x']

    def __call__(self, x):
        return self.c @ x.T

    def gradient(self, x):
        raise NotImplementedError('Gradient of the function is not available.')

    def estimate_gradient(self, x, ksi, m, sample_size=5, return_var=False):
        gradient_estimates = []
        eye = np.eye(x.shape[0])
        for _ in range(m):
            grad = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                eps = np.random.normal(0, ksi, (sample_size))
                est = [(self(x + eye[i] * h) - self(x)) / h for h in eps]

                if isinstance(est[0], np.ndarray):
                    grad[i] = np.mean(est, axis=0)[i]
                else:
                    grad[i] = np.mean(est)
            gradient_estimates.append(grad)

        if return_var:
            return np.mean(gradient_estimates, axis=0), np.var(gradient_estimates, axis=1)
        else:
            return np.mean(gradient_estimates, axis=0)

    def projection(self, x):
        jac = lambda x: self.estimate_gradient(x, 1, 20)
        loss = lambda x_: np.linalg.norm((x - x_))
        cons = {'type': 'ineq',
                'fun': lambda x_: self.b - np.dot(self.A, x_),
                'jac': lambda x_: -self.A}
        bounds = [(0, None) for _ in range(x.shape[0])]

        x_proj = minimize(loss, x, jac='cs', constraints=cons, bounds=bounds, method='SLSQP', options={'disp': False})['x']
        return x_proj


class FarmingStochastic(Farming):
    def __init__(self):
        super().__init__()
        self.c = np.concatenate([
            np.array([150, 230, 260]),
            1/3 * np.array([238, 210, -170, -150, -36, -10]),
            1/3 * np.array([238, 210, -170, -150, -36, -10]),
            1/3 * np.array([238, 210, -170, -150, -36, -10])
        ])
        #                   x1  x2  x3  y11 y21 p11 p21 p31 p41 y12 y22 p12 p22 p32 p42 y13 y23 p13 p23 p33 p43
        self.A = np.array([[1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                           [-3, 0,  0, -1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                           [-2.5,0,  0,  0,  0,  0,  0,  0,  0,  -1,  0, 1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                           [-2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  -1,  0, 1,  0,  0,  0],
                           [0,-3.6,  0,  0,  -1,  0, 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                           [0,  -3,  0,  0,  0,  0,  0,  0,  0,  0,  -1,  0, 1,  0,  0,  0,  0,  0,  0,  0,  0],
                           [0,-2.4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  -1,  0, 1,  0,  0],
                           [0,  0,-24,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                           [0,  0,-20,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0],
                           [0,  0,-16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1],
                           [0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                           [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
                           [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0]])
        self.b = np.array([500, -200, -200, -200, -240, -240, -240, 0, 0, 0, 6000, 6000, 6000])

        solution = linprog(self.c, A_ub=self.A, b_ub=self.b)
        self.optimal_val = solution['fun']
        self.optimal_x = solution['x']


def pg_algorithm(x, gammas, f):
    for gamma in gammas:
        y = x - gamma * f.gradient(x)
        x = f.projection(y)
        print(f'Loss: {np.mean((x - f.projection(0)) ** 2)}')
    return x


def rspg_algorithm(x_0, gammas, f, S, sigma, m):
    candidate_solutions = np.zeros((S, x_0.shape[0]))
    gradient_norms = np.zeros(S)
    all_losses = []
    all_R = []
    for i in range(S):
        losses = []
        R = np.random.randint(0, len(gammas))
        all_R.append(R)
        x = copy.deepcopy(x_0)
        for gamma in tqdm(gammas[:R]):
            y = x - gamma * f.estimate_gradient(x, sigma, m)
            x = f.projection(y)
            loss = np.sqrt((f(x) - f.optimal_val) ** 2)
            losses.append(loss)
        print(f'\tLoss: {loss}')
        all_losses.append(losses)
        candidate_solutions[i] = x
        gradient_norms[i] = loss
    best_grad = np.argmin(gradient_norms)
    return candidate_solutions[best_grad], all_losses, all_R


def estimate_parameters(f):
    points = np.zeros((10, data_shape))
    grad_estimates = np.zeros((10, data_shape))
    var_estimates = []
    for i in range(10):
        random_point = np.random.normal(0, 1, data_shape)
        points[i] = random_point

        grad, var_grads = f.estimate_gradient(random_point, 1, 20, sample_size=20, return_var=True)
        var_est = np.sqrt(np.mean(var_grads))
        var_estimates.append(var_est)
        grad_estimates[i] = grad
    sigma_estimate = np.max(var_estimates)

    L_estimates = []
    for i, j in product(list(range(10)), list(range(10))):
        grad_norm = np.linalg.norm(grad_estimates[i] - grad_estimates[j])
        L_estimate = grad_norm / np.linalg.norm(points[i] - points[j])
        L_estimates.append(L_estimate)
    L_estimate = np.nanmean(L_estimates)

    return L_estimate, sigma_estimate


if __name__ == '__main__':
    func = FarmingStochastic()
    data_shape = 21

    M = 15000

    _, sigma = estimate_parameters(func)
    alpha = 1
    L = 0.5

    x_0 = linprog(c=np.zeros(func.A.shape[1]), A_ub=func.A, b_ub=func.b)['x']
    D = np.sqrt((func(x_0) - func.optimal_val) / L)

    m = int(np.ceil(min([M, max([1, (sigma * np.sqrt(6 * M)) / (4 * L * D)])])))

    x_opt, losses, Rs = rspg_algorithm(x_0, [1 / (2 * L)] * int(np.floor(M / m)), func, 5, 0.1, m)
    print(f'Final loss: {np.sqrt((func(x_opt) - func.optimal_val) ** 2)}')
    print()
    print(f'===============')
    print(f'Wheat planted:\t\t {x_opt[0]:.0f} ar\n'
          f'Corn planted:\t\t {x_opt[1]:.0f} ar\n'
          f'Beats planted:\t\t {x_opt[2]:.0f} ar\n')
    print('-----------------------------------------------------------------------------------------------------')
    print(tabulate([[f'{x_opt[3]:.0f}', f'{x_opt[4]:.0f}', f'{x_opt[5]:.0f}', f'{x_opt[6]:.0f}', f'{x_opt[7]:.0f}', f'{x_opt[8]:.0f}'],
                    [f'{x_opt[9]:.0f}', f'{x_opt[10]:.0f}', f'{x_opt[11]:.0f}', f'{x_opt[12]:.0f}', f'{x_opt[13]:.0f}', f'{x_opt[14]:.0f}'],
                    [f'{x_opt[15]:.0f}', f'{x_opt[16]:.0f}', f'{x_opt[17]:.0f}', f'{x_opt[18]:.0f}', f'{x_opt[19]:.0f}', f'{x_opt[20]:.0f}']],
                   headers=['Wheat bought', 'Corn bought', 'Wheat sold', 'Corn sold', 'Beats sold (<=6000)', 'Beats sold (>6000)']))
    print(f'Expected Profit:\t{-func(x_opt):.0f}$')

    plt.figure(figsize=(16, 9))
    plt.axhline(y=0, label="Optimal value (loss=0)")
    plt.xlabel("Number of steps")
    plt.ylabel("Loss")
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    Rs = sorted(Rs, reverse=True)
    losses = sorted(losses, key=lambda x: len(x), reverse=True)
    for i, (R, loss) in enumerate(zip(Rs, losses)):
        last_loss = loss[-1]
        idx = len(loss) - 1
        plt.plot(loss, c=colors[i])
        plt.scatter(x=idx, y=loss[-1], c=colors[i])
        plt.annotate(f'R={R}', xy=(idx, loss[-1]), xycoords='data', xytext=(-10, 10), textcoords='offset points')
    plt.legend()
    plt.savefig('loss.pdf')

