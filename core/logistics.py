"""Транспортная задача: балансировка и построение матрицы ограничений."""
import numpy as np
from typing import Tuple


def balance_task(
    costs: np.ndarray, supply: np.ndarray, demand: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Балансирует транспортную задачу: если Σзапасов ≠ Σспроса,
    добавляет фиктивного поставщика/потребителя с нулевыми затратами.
    Возвращает (costs, supply, demand, was_balanced).
    """
    total_s, total_d = np.sum(supply), np.sum(demand)
    if abs(total_s - total_d) < 1e-6:
        return costs, supply, demand, False

    if total_s > total_d:
        diff = total_s - total_d
        demand = np.append(demand, diff)
        costs = np.hstack([costs, np.zeros((costs.shape[0], 1))])
    else:
        diff = total_d - total_s
        supply = np.append(supply, diff)
        costs = np.vstack([costs, np.zeros((1, costs.shape[1]))])

    return costs, supply, demand, True


def build_lp_matrices(
    costs: np.ndarray, supply: np.ndarray, demand: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Строит c, A_eq, b_eq для транспортной задачи."""
    n_s, n_c = costs.shape
    c = costs.flatten()

    A_eq, b_eq = [], []
    for i in range(n_s):
        row = np.zeros(n_s * n_c)
        for j in range(n_c):
            row[i * n_c + j] = 1
        A_eq.append(row); b_eq.append(supply[i])
    for j in range(n_c):
        row = np.zeros(n_s * n_c)
        for i in range(n_s):
            row[i * n_c + j] = 1
        A_eq.append(row); b_eq.append(demand[j])

    return c, np.array(A_eq), np.array(b_eq)