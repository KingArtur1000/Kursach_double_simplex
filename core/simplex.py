"""Двухфазный симплекс-метод с логированием."""
import numpy as np
from typing import Callable, Optional, List, Tuple


LogFn = Callable[[str, Optional[str]], None]


def simplex_iterations(
    A: np.ndarray, b: np.ndarray, c: np.ndarray,
    basis: List[int], log: LogFn, phase: int, max_iter: int = 500
) -> Tuple[bool, List[int], Optional[float]]:
    """Итерации симплекс-метода. A: m×n, b: m, c: n, basis: m индексов."""
    m, n = A.shape
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    basis = list(basis)

    for iteration in range(1, max_iter + 1):
        B = A[:, basis]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            log("  ⚠ Вырожденный базис — прерывание.", "error")
            return False, basis, None

        x_B = B_inv @ b
        c_B = c[basis]
        z = c_B @ x_B
        y = c_B @ B_inv
        reduced = c - y @ A
        non_basic = [j for j in range(n) if j not in basis]

        log(f"\n  ┌── Итерация {iteration} (фаза {phase}) ──────────────", "iteration")
        log("  │ Базис: " + ", ".join(f"x{basis[i]+1}" for i in range(m)))
        log("  │ Решение: " + ", ".join(f"x{basis[i]+1}={x_B[i]:.3f}" for i in range(m)))
        log(f"  │ z = {z:.6f}")

        if non_basic:
            shown = non_basic[:12]
            rc_str = ", ".join(f"Δ{j+1}={reduced[j]:+.4f}" for j in shown)
            if len(non_basic) > 12:
                rc_str += ", ..."
            log("  │ Оценки: " + rc_str)

        # Оптимальность
        entering, min_rc = None, -1e-9
        for j in non_basic:
            if reduced[j] < min_rc:
                min_rc = reduced[j]
                entering = j

        if entering is None:
            log("  │ ✓ Все Δ ≥ 0 → ОПТИМУМ", "success")
            log("  └──────────────────────────────────────", "iteration")
            return True, basis, z

        log(f"  │ → Вводим x{entering+1} (Δ={reduced[entering]:.4f})", "info")
        d = B_inv @ A[:, entering]
        ratios = [(x_B[i] / d[i], i) for i in range(m) if d[i] > 1e-9]

        if not ratios:
            log("  │ ✗ Задача неограничена", "error")
            log("  └──────────────────────────────────────", "iteration")
            return False, basis, None

        min_ratio, leaving_idx = min(ratios)
        leaving = basis[leaving_idx]
        log(f"  │ → Выводим x{leaving+1} (min отн. = {min_ratio:.4f})", "info")
        log("  └──────────────────────────────────────", "iteration")

        basis[leaving_idx] = entering

    log(f"  ⚠ Лимит итераций ({max_iter})", "error")
    return False, basis, None


def solve_two_phase_simplex(
    c_orig: np.ndarray, A_eq: np.ndarray, b_eq: np.ndarray, log: LogFn
) -> Optional[Tuple[np.ndarray, float]]:
    """Запускает двухфазный симплекс. Возвращает (x, z*) или None."""
    m, n = A_eq.shape
    A = A_eq.astype(float).copy()
    b = b_eq.astype(float).copy()

    for i in range(m):
        if b[i] < 0:
            A[i] = -A[i]
            b[i] = -b[i]

    log("╔══════════════════════════════════════════════════════════╗", "header")
    log("║          ДВУХФАЗНЫЙ СИМПЛЕКС-МЕТОД                       ║", "header")
    log("╚══════════════════════════════════════════════════════════╝", "header")
    log("")
    log(f"Размерность: {m} ограничений, {n} переменных")
    log("Целевая функция:  min z = " +
        " + ".join(f"{c_orig[j]:.2f}·x{j+1}" for j in range(n)))
    log("")

    # ─── ФАЗА 1 ───
    log("─" * 60, "phase1")
    log("ФАЗА 1. Поиск начального допустимого базисного решения", "phase1")
    log("─" * 60, "phase1")

    A1 = np.hstack([A, np.eye(m)])
    c1 = np.concatenate([np.zeros(n), np.ones(m)])
    basis = list(range(n, n + m))

    ok, basis, z1 = simplex_iterations(A1, b, c1, basis, log, phase=1)
    if not ok or z1 > 1e-6:
        log(f"\n❌ Фаза 1 не сошлась (W={z1:.6f})", "error")
        return None
    log(f"\n✅ W = {z1:.6f} ⇒ допустимый базис найден.", "success")

    # Вывод искусственных из базиса
    redundant = []
    for idx in range(m):
        if basis[idx] >= n:
            B = A1[:, basis]
            try:
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                redundant.append(idx); continue
            row = B_inv[idx, :] @ A1
            swapped = False
            for j in range(n):
                if j not in basis and abs(row[j]) > 1e-9:
                    basis[idx] = j; swapped = True; break
            if not swapped:
                redundant.append(idx)

    keep = [i for i in range(m) if i not in redundant]
    A_red = A[keep, :]
    b_red = b[keep]
    basis_red = [basis[i] for i in keep]

    # ─── ФАЗА 2 ───
    log("")
    log("─" * 60, "phase2")
    log("ФАЗА 2. Оптимизация исходной целевой функции", "phase2")
    log("─" * 60, "phase2")

    ok, basis_final, z2 = simplex_iterations(
        A_red, b_red, c_orig.astype(float).copy(), basis_red, log, phase=2
    )
    if not ok:
        return None
    log(f"\n✅ ОПТИМУМ: z* = {z2:.6f}", "success")

    B = A_red[:, basis_final]
    x_B = np.linalg.solve(B, b_red)
    x = np.zeros(n)
    for i, b_idx in enumerate(basis_final):
        x[b_idx] = x_B[i]
    return x, z2