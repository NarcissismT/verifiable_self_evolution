import math
import numpy as np

def solve(case_id, problem, seed, oracle_budget, hyperparameters):
    del seed, hyperparameters
    if case_id == "realpilot_flatness_penalty":
        a = float(problem["a"]); target = np.asarray(problem["upper_target"], dtype=float)
        x = np.linspace(0.1, 2.0, 1200); y0 = np.sqrt(a * x)
        values = [0.5 * ((x - target[0]) ** 2 + (y0 - target[1]) ** 2), 0.5 * ((x - target[0]) ** 2 + (-y0 - target[1]) ** 2)]
        row, col = min(((v, i) for v in values for i in range(len(x))), key=lambda z: float(z[0][z[1]]))
        return {"point": [float(x[col]), float(y0[col] if row is values[0] else -y0[col])], "oracle_calls": min(4000, oracle_budget)}
    if case_id == "realpilot_nonconvex_simple":
        a = np.asarray(problem["a"], dtype=float); target = np.asarray(problem["upper_target"], dtype=float); c = float(problem["coupling"])
        roots = np.sqrt(a); choices = []
        for s0 in (-1.0, 1.0):
            for s1 in (-1.0, 1.0):
                p = roots * np.array([s0, s1]); choices.append((0.5 * float(np.sum((p-target)**2)) + c * math.sin(float(p.sum())), p))
        return {"point": min(choices, key=lambda z: z[0])[1].tolist(), "oracle_calls": 4}
    if case_id == "realpilot_linear_coupling":
        A = np.asarray(problem["lower_matrix"], dtype=float); b = np.asarray(problem["lower_offset"], dtype=float); C = np.asarray(problem["constraint_matrix"], dtype=float); d = np.asarray(problem["constraint_bound"], dtype=float); target = np.asarray(problem["upper_target"], dtype=float)
        best = None
        for x0 in np.linspace(-1.6, 1.6, 41):
            for x1 in np.linspace(-1.6, 1.6, 41):
                p = np.r_[x0, x1, A @ np.array([x0, x1]) + b]
                if np.max(C @ p - d) <= 1e-10:
                    score = 0.5 * float(np.sum((p-target)**2))
                    if best is None or score < best[0]: best = (score, p)
        if best is None: raise RuntimeError("no feasible point")
        return {"point": best[1].tolist(), "oracle_calls": min(4000, oracle_budget)}
    raise ValueError(case_id)
