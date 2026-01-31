import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path

# 固定亂數，方便比較 1000 / 10000 次
np.random.seed(42)

# ======================
# 1. Load data
# ======================
root = Path(__file__).resolve().parent
mat = loadmat(root / "purdue.mat")

cube = None
for k in mat.keys():
    if not k.startswith("__") and isinstance(mat[k], np.ndarray) and mat[k].ndim == 3:
        cube = mat[k]
        print(f"Using array '{k}' with shape {cube.shape}")
        break

if cube is None:
    raise ValueError("No 3D hyperspectral cube found")

H, W, B = cube.shape
print(f"H={H}, W={W}, Bands={B}")

# ======================
# 2. Reshape to (Bands, Pixels)
# ======================
X = cube.reshape(H * W, B).T   # (B, N)
mean = X.mean(axis=1, keepdims=True)
Xc = X - mean

# ======================
# 3. Covariance & eigen
# ======================
cov = (Xc @ Xc.T) / Xc.shape[1]
eigenvalues, eigenvectors = np.linalg.eigh(cov)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]


# ======================
# 4. Decide k (99% energy, then *2)
# ======================
energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
r = np.searchsorted(energy, 0.99) + 1
k = min(2 * r, B)

print(f"r (99% energy) = {r}, k = {k}")
# ======================
# Show eigenvalue decay & cumulative energy (final version)
# ======================
fig, ax1 = plt.subplots(figsize=(8, 5))

x = np.arange(1, len(eigenvalues) + 1)

# --- Left axis: eigenvalue decay ---
ax1.plot(x, eigenvalues, color="black", linewidth=1.5, label="Eigenvalue")
ax1.set_xlabel("Component index")
ax1.set_ylabel("Eigenvalue", color="black")
ax1.tick_params(axis='y', labelcolor="black")

# --- Right axis: cumulative energy ---
ax2 = ax1.twinx()
ax2.plot(x, energy, color="red", linewidth=2, label="Cumulative energy")
ax2.axhline(0.99, color="red", linestyle="--", linewidth=1)
ax2.axvline(r, color="blue", linestyle="--", linewidth=1)

ax2.scatter(r, energy[r-1], color="blue", zorder=5)
ax2.set_ylabel("Cumulative energy ratio", color="red")
ax2.tick_params(axis='y', labelcolor="red")

# --- Annotations ---
ax2.text(r + 1, 0.92, f"r = {r}\n(99% energy)", color="blue")

plt.title("Eigenvalue distribution and cumulative energy (99% criterion)")
fig.tight_layout()
plt.show()


# ======================
# 5. PCA projection
# ======================
num_pca = 8
Y = eigenvectors[:, :num_pca].T @ Xc   # (8, N)

# ======================
# 6. Show PCA(1)~PCA(8)
# ======================
plt.figure(figsize=(10, 5))
for i in range(num_pca):
    plt.subplot(2, 4, i + 1)
    img = Y[i].reshape(H, W)
    plt.imshow(img, cmap="gray")
    plt.title(f"PCA({i+1})")
    plt.axis("off")

plt.tight_layout()
plt.show()

# ======================
# 7. PPI function
# ======================
def ppi(data, iterations):
    """
    data: (D, N)
    """
    counts = np.zeros(data.shape[1], dtype=int)

    for _ in range(iterations):
        r = np.random.randn(data.shape[0], 1)
        r /= (np.linalg.norm(r) + 1e-12)

        proj = (r.T @ data).ravel()
        counts[np.argmax(proj)] += 1
        counts[np.argmin(proj)] += 1

    return counts

# ======================
# 8. PPI comparison (draw on PCA(1))
# ======================
pca1_img = Y[0].reshape(H, W)

cases = [
    ("RAW", Xc, 1000),
    ("RAW", Xc, 10000),
    ("PCA", Y, 1000),
    ("PCA", Y, 10000),
]

plt.figure(figsize=(10, 10))

for i, (name, data_space, iters) in enumerate(cases, 1):
    counts = ppi(data_space, iters)
    top_idx = np.argsort(counts)[::-1][:k]

    ys, xs = np.divmod(top_idx, W)

    plt.subplot(2, 2, i)
    plt.imshow(pca1_img, cmap="gray")
    plt.scatter(xs, ys, c="red", s=15)
    plt.title(f"PPI on PCA(1)\n{name}, iter={iters}")
    plt.axis("off")

plt.tight_layout()
plt.show()
