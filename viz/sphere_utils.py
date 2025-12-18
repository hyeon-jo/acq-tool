import numpy as np


def l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    각 행 벡터를 L2 정규화하여 단위 구 좌표로 투영.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D (num_points, dim)")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


def to_spherical_coords(cartesian: np.ndarray) -> np.ndarray:
    """
    카테시안(x,y,z) -> 구면 좌표(theta, phi) [라디안]
    theta: 경도(-pi ~ pi), phi: 위도(-pi/2 ~ pi/2)
    """
    if cartesian.shape[1] != 3:
        raise ValueError("cartesian must have shape (N, 3)")
    x, y, z = cartesian[:, 0], cartesian[:, 1], cartesian[:, 2]
    theta = np.arctan2(y, x)
    r = np.linalg.norm(cartesian, axis=1)
    r = np.maximum(r, 1e-12)
    phi = np.arcsin(np.clip(z / r, -1.0, 1.0))
    return np.stack([theta, phi], axis=1)


def from_spherical_coords(theta_phi: np.ndarray) -> np.ndarray:
    """
    구면 좌표(theta, phi) -> 카테시안(x,y,z), r=1
    """
    if theta_phi.shape[1] != 2:
        raise ValueError("theta_phi must have shape (N, 2)")
    theta, phi = theta_phi[:, 0], theta_phi[:, 1]
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    return np.stack([x, y, z], axis=1)


def generate_sphere_grid(num_longitude: int = 72, num_latitude: int = 36) -> np.ndarray:
    """
    균일한 구면 격자 샘플(대략적). 반환 shape=(num_longitude*num_latitude, 3)
    - 경도: [-pi, pi)
    - 위도: (-pi/2, pi/2)
    """
    longitudes = np.linspace(-np.pi, np.pi, num=num_longitude, endpoint=False)
    latitudes = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, num=num_latitude, endpoint=True)
    theta, phi = np.meshgrid(longitudes, latitudes)
    theta_phi = np.stack([theta.flatten(), phi.flatten()], axis=1)
    return from_spherical_coords(theta_phi)


def maybe_sample_points(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """
    points에서 최대 max_points만 균일 샘플링.
    """
    n = points.shape[0]
    if n <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return points[idx]


