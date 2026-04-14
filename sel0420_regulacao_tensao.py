import numpy as np

# ============================================================
# SEL420 - Distribuição de Energia Elétrica | EESC-USP
# ============================================================

# -----------------------------------------------------------
# Matrizes Y (admitância) e Identidade
# -----------------------------------------------------------
Y = np.array([
    [ 4.09168e-6, -9.82321e-7, -1.0769e-6],
    [-9.82321e-7,  6.95203e-6, -1.05867e-6],
    [-1.0769e-6,  -1.05867e-6,  4.77215e-6]
])  # 3j # JS

I = np.identity(3)
import numpy as np

Z = np.array([
    [1.46261 + 1.272981j, 0.223787 + 0.531915j, 0.209859 + 0.552882j],
    [0.223787 + 0.531915j, 1.45317 + 1.257177j, 0.214685 + 0.545496j],
    [0.209859 + 0.552882j, 0.214685 + 0.545496j, 1.42569 + 1.29866j]
])

# -----------------------------------------------------------
# Montagem dos blocos da matriz ABCD
# -----------------------------------------------------------
a = I + 1/2 * Z @ Y

b = Z

c = Y + 1/4 * Y @ Z @ Y

d = I + 1/2 * Z @ Y

np.set_printoptions(precision=4)

print("Matriz a:\n", a)
print("Matriz b:\n", b)
print("Matriz c:\n", c)
print("Matriz d:\n", d)

# -----------------------------------------------------------
# Função auxiliar: polar → retangular
# -----------------------------------------------------------
def polar_to_rect(magnitude, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    return magnitude * np.exp(1j * angle_rad)

# -----------------------------------------------------------
# Tensões e correntes no nó m (fonte)
# -----------------------------------------------------------
Vf_abc_m = np.array([
    polar_to_rect(magnitude=8025.17, angle_deg=0),
    polar_to_rect(magnitude=8025.17, angle_deg=120),
    polar_to_rect(magnitude=8025.17, angle_deg=-120)
])

I_abc_m = np.array([
    polar_to_rect(magnitude=41.436, angle_deg=-23.07),
    polar_to_rect(magnitude=41.436, angle_deg=96.93),
    polar_to_rect(magnitude=41.436, angle_deg=-143.07)
])

# Empilhando os vetores em uma matriz 6x1
Vm_Im = np.concatenate(arrays=(Vf_abc_m, I_abc_m), axis=0).reshape(6, 1)

# Montagem da matriz 6x6 com blocos a, b, c, d
top    = np.hstack((a, b))   # 3x6
bottom = np.hstack((c, d))   # 3x6
abcd   = np.vstack((top, bottom))  # 6x6

# Multiplicação matricial para obter valores no nó n
Vf_abc_n_I_abc_n = abcd @ Vm_Im

# Separando os resultados
Vf_abc_n = Vf_abc_n_I_abc_n[:3].flatten()
I_abc_n  = Vf_abc_n_I_abc_n[3:].flatten()

print("Tensões no nó n:")
print(Vf_abc_n)

print("\nCorrentes no nó n:")
print(I_abc_n)

# -----------------------------------------------------------
# Potência complexa trifásica no nó n
# -----------------------------------------------------------

# Potência complexa por fase: S = V * I_conjugado
S_fases = Vf_abc_n * np.conj(I_abc_n)

# Soma total da potência complexa trifásica
S_total = np.sum(S_fases)

# Separando em partes ativa (P) e reativa (Q)
P_total = S_total.real
Q_total = S_total.imag

print(f"Potência por fase (S_a, S_b, S_c):\n{S_fases}")
print(f"\nPotência total: {S_total:.2f} VA")
print(f"Ativa (P): {P_total:.2f} W")
print(f"Reativa (Q): {Q_total:.2f} VAr")

# -----------------------------------------------------------
# Cálculo de tn e corrente In
# -----------------------------------------------------------
Znn = 8.2053 + 9.4666j
Znj = np.array([0.5926 + 4.6139j, 0.5926 + 4.7512j, 0.5926 + 4.6382j])

tn = -(1 / Znn) * Znj

# Exibe o resultado formatado
print(f"\ntn: {tn}")

In = tn * I_abc_n
print(f"\nIn: {In}")

# -----------------------------------------------------------
# Corrente na terra (KCL)
# It = -(Ia + Ib + Ic + In1 + In2 + ... + Inn)
# -----------------------------------------------------------
It = -(np.sum(I_abc_n) + np.sum(In))
print(f"\nIt: {It}")
