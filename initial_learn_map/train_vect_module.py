#library import
import numpy as np
import numpy.matlib
import math
from itertools import combinations

#file directives import
from mai import modular_residue
from inference import inference_core

def to_uint(arr):
    """Converte array para inteiro sem sinal (uint8)."""
    return np.round(arr).astype(np.uint8)

# ============================================================
# Funções auxiliares (MATLAB -> Python)
# ============================================================

def nchoosek(v, k):
    """Combinações de k elementos de v (similar ao nchoosek do MATLAB)."""
    v = np.asarray(v).flatten()
    if k == 0:
        return np.array([[]])
    comb = list(combinations(v, k))
    return np.array(comb)

def accumarray(subs, vals, size=None, func=np.mean):
    """Agrupa valores por índices (similar ao accumarray do MATLAB)."""
    subs = np.asarray(subs).flatten().astype(int)
    vals = np.asarray(vals).flatten()
    if size is None:
        size = subs.max() + 1
    result = np.zeros(size)
    for i in range(size):
        mask = (subs == i)
        if np.any(mask):
            result[i] = func(vals[mask])
    return result

def double(x):
    return x.astype(np.float64)

def real(x):
    return np.real(x)

# ============================================================
# Função principal de treinamento
# ============================================================

def train_vect_module(xh=None, yh=None):
    # --- Valores padrão (XOR) ---
    if xh is None or yh is None:
        xh = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        yh = np.array([[0], [0], [0], [1]])

    S = xh.shape[0]
    n_entradas = xh.shape[1]
    model = {}

    # ------------------------------------------------------------
    # MÉTODO DE REFINAMENTO (para n_entradas <= 2)
    # ------------------------------------------------------------
    if False:   # desativa o método de refinamento
        if n_entradas == 1:
            M = 2
            l_bus = 0
        else:
            M = 5
            l_bus = int(np.log2(M - 1))

        G = n_entradas + 1
        X = [xh]          # X{1}
        l = 1
        Dy = yh.copy()
        W = []
        L = []
        I = np.zeros((G, 0))

        while l <= len(X) and len(X[l-1]) > 0:
            N = X[l-1].shape[1]

            if G <= N:
                Gs = nchoosek(np.arange(1, N+1), G)
            else:
                Gs = np.arange(1, N+1).reshape(1, -1)

            if l <= l_bus:
                factor = 2 ** (l_bus - l - 1)
                X.append(np.mod(np.floor(xh / factor), 2))
            else:
                X.append(np.array([]))

            for g in range(Gs.shape[0]):
                cols = Gs[g, :] - 1
                xs = X[l-1][:, cols]
                xs = xs * (2.0 ** np.arange(Gs.shape[1]))
                if xs.shape[1] == 1:
                    xs = xs.reshape(-1, 1)

                n = max(3, Gs.shape[1])     
                ones = np.ones((xs.shape[0], 1))
                rep = np.tile(xs, (1, n - 1))
                V_raw = np.cumprod(np.hstack([ones, rep]), axis=1)

                # Usa a classe modular_residue do mai.py
                V = modular_residue(to_uint(V_raw), M)
                v = double(V._residue)   # <--- CORRIGIDO: ._residue

                if xs.ndim == 1:
                    xs = xs.reshape(-1, 1)
                _, ju = np.unique(xs, axis=0, return_inverse=True)
                Dyp = accumarray(ju, Dy, size=ju.max()+1, func=np.mean)
                Dyp = Dyp[ju]

                if np.any(Dyp):
                    rows_v, _ = v.shape
                    rows_Dy = Dy.shape[0]
                    if rows_Dy != rows_v:
                        if rows_Dy > rows_v:
                            Dy_adj = Dy[:rows_v, :]
                        else:
                            pad = np.zeros((rows_v - rows_Dy, Dy.shape[1]))
                            Dy_adj = np.vstack([Dy, pad])
                    else:
                        Dy_adj = Dy

                    # Resolver sistema linear (mínimos quadrados)
                    w, _, _, _ = np.linalg.lstsq(v, Dy_adj, rcond=None)

                    # Aplicar módulo SEM conversão para uint8 (mantém float)
                    w = np.mod(w, M).astype(np.float64)

                    # erro
                    error = v @ w   # w já é float

                    print(f"layer {l}, g={g}: v.shape = {v.shape}, w.shape = {w.shape}")
                    print(f"w (antes do modulo): {w.flatten()[:5]}")
                    
                    print(f"Layer {id}: v.shape={v.shape}, w.shape={w.shape}")
                    contrib = v @ double(w)
                    print(f"contrib min={contrib.min()}, max={contrib.max()}")
                    # ... ajuste de error e Dy ...

                    # Armazenar
                    W.append(w)
                    L.append(l)

                    # Atualizar I (com tamanho G)
                    nova_coluna = np.zeros((G, 1))
                    k = len(Gs[g, :])
                    nova_coluna[:k, 0] = Gs[g, :]
                    I = np.hstack([I, nova_coluna])

                    if not np.any(Dy):
                        break

            if not np.any(Dy):
                break
            l += 1

        model['W'] = W
        model['I'] = I
        model['L'] = L
        model['M'] = M
        model['l_bus'] = l_bus
        model['G'] = G
        model['inference'] = lambda x=None: inference_core(x, W, I, L, M, l_bus)
        Deltay = Dy

    # ------------------------------------------------------------
    # MÉTODO POLINOMIAL (fallback para n_entradas > 2)
    # ------------------------------------------------------------
    else:
        features_list = [np.ones((S, 1))]
        for i in range(n_entradas):
            features_list.append(xh[:, i:i+1])
        for i in range(n_entradas):
            for j in range(i+1, n_entradas):
                features_list.append((xh[:, i] * xh[:, j]).reshape(-1, 1))
        if n_entradas >= 3:
            features_list.append((xh[:, 0] * xh[:, 1] * xh[:, 2]).reshape(-1, 1))

        X_features = np.hstack(features_list)
        M = 3
        lambda_ = 0.001
        XtX = X_features.T @ X_features
        reg = lambda_ * np.eye(X_features.shape[1])
        w = np.linalg.solve(XtX + reg, X_features.T @ yh)
        w = np.mod(w, M)
        y_pred = np.mod(X_features @ w, M)
        Deltay = np.mod(yh - y_pred, M)

        model['W'] = [w]
        model['I'] = np.arange(1, X_features.shape[1]+1).reshape(1, -1)
        model['L'] = 1
        model['M'] = M
        model['G'] = n_entradas + 1
        model['l_bus'] = 0
        model['inference'] = lambda x=None: inference_poly(x, w, M, n_entradas)

    return model, Deltay


# ============================================================
# Funções de inferência (placeholders)
# ============================================================

def inference_poly(x, w, M, n_entradas):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    S = x.shape[0]
    features_list = [np.ones((S, 1))]
    for i in range(n_entradas):
        features_list.append(x[:, i:i+1])
    for i in range(n_entradas):
        for j in range(i+1, n_entradas):
            features_list.append((x[:, i] * x[:, j]).reshape(-1, 1))
    if n_entradas >= 3:
        features_list.append((x[:, 0] * x[:, 1] * x[:, 2]).reshape(-1, 1))
    X_features = np.hstack(features_list)
    y = np.mod(np.round(X_features @ w), M)
    return np.mod(y, 2)
