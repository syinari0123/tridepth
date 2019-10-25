import cv2


def calculate_canny_edges(image, denoise=True):
    """OTSU-based adaptive canny edge
    Args:
        image: [H,W,3]
    Returns:
        edge: [H,W]
    """
    # Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply denoising (bilateral filter)
    if denoise:
        gray_image = cv2.bilateralFilter(gray_image, 5, 50, 50)

    # Detect canny edge by OTSU threshold
    otsu_th, ret = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Set double threshold of canny edge
    lower = otsu_th * 0.5
    upper = otsu_th
    edge = cv2.Canny(gray_image, lower, upper) / 255.0

    # Invert edgemap
    inv_edge = (1.0 - edge) * 255.0

    return inv_edge, gray_image
