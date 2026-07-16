import numpy as np
import os
import warnings
import numpy.matlib
import math
import numpy as np

def double(x):
    """Converte para float64."""
    return x.astype(np.float64)

def real(x):
    """Parte real."""
    return np.real(x)

def to_uint(arr):
    """Converte para inteiro sem sinal (uint8)."""
    return np.round(arr).astype(np.uint8)

# Importa a classe modular_residue do mai.py
from mai import modular_residue

# ============================================================
# Função de inferência para o método de refinamento
# ============================================================

def inference_core(x=None, W=None, I=None, L=None, M=None, l_bar=None):
    """
    Inferência para o método de refinamento.
    Reconstrói as camadas a partir de x e aplica os pesos salvos.
    """
    if x is None or W is None or I is None or L is None or M is None or l_bar is None:
        raise ValueError("Todos os parâmetros são obrigatórios.")
    
    # Garantir que x seja 2D (n_amostras, n_features)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    S = x.shape[0]
    
    # --- 1. Reconstruir camadas (mesma lógica do treinamento) ---
    X = [x]  # camada 0 (1ª camada)
    max_l = max(L) if L else 0
    for l in range(1, max_l + 1):
        if l <= l_bar:
            factor = 2 ** (l_bar - l - 1)
            X.append(np.mod(np.floor(x / factor), 2))
        else:
            X.append(np.array([]))   # camada vazia
    
    # --- 2. Aplicar pesos incrementalmente ---
    y = np.zeros((S, 1))
    
    for idx, w in enumerate(W):
        # Índice da camada (0‑based)
        layer_idx = L[idx] - 1
        # Índices das colunas (0‑based, removendo padding zeros)
        cols = I[:, idx] - 1
        cols = cols[cols >= 0].astype(int)
        
        # Pular se a camada estiver vazia ou não tiver colunas suficientes
        X_layer = X[layer_idx]
        if X_layer.size == 0 or X_layer.shape[1] < len(cols):
            continue
        
        # Selecionar colunas e aplicar pesos binários
        xs = X_layer[:, cols]
        xs = xs * (2.0 ** np.arange(len(cols)))
        if xs.shape[1] == 1:
            xs = xs.reshape(-1, 1)
        
        # Matriz de Vandermonde
        n = max(3, len(cols))
        ones = np.ones((xs.shape[0], 1))
        rep = np.tile(xs, (1, n - 1))
        V_raw = np.cumprod(np.hstack([ones, rep]), axis=1)
        V = modular_residue(to_uint(V_raw), M)
        v = double(V._residue)
        
        # Ajustar dimensões de v para combinar com w
        if v.shape[1] > len(w):
            v = v[:, :len(w)]
        elif v.shape[1] < len(w):
            pad = np.zeros((v.shape[0], len(w) - v.shape[1]))
            v = np.hstack([v, pad])
        
        # Contribuição deste layer (mantendo no módulo para evitar crescimento)
        contrib = v @ double(w)
        y = np.mod(y + contrib, M)   # <--- aplica módulo a cada passo
    
    # --- 3. Aplicar módulo final e converter para binário ---
    y = np.mod(np.round(y), M)
    y = np.mod(y, 2)
    return y
