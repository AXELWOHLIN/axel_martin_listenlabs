import numpy as np

def nearest_psd_correlation(C: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Project a symmetric matrix to the nearest correlation matrix by
    eigenvalue clipping and re-scaling the diagonal to ones.
    Higham (2002) idea; good enough for our use.  # noqa
    """
    # symmetrize
    C = 0.5 * (C + C.T)
    # eigen clip
    w, V = np.linalg.eigh(C)
    w_clipped = np.clip(w, tol, None)
    C_psd = (V * w_clipped) @ V.T
    # set diag to 1
    d = np.sqrt(np.diag(C_psd))
    C_corr = C_psd / np.outer(d, d)
    np.fill_diagonal(C_corr, 1.0)
    return C_corr
