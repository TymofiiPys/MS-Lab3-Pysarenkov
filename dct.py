import numpy as np
import matplotlib.pyplot as plt

def dct_matrix(N = 8) -> np.ndarray:
    # formula: https://www.mathworks.com/help/images/discrete-cosine-transform.html
    dct_matrix = np.zeros((N, N))
    alpha = np.sqrt(2 / N) * np.ones(N)
    alpha[0] = np.sqrt(1 / N)

    for u in range(N):
        for x in range(N):
            dct_matrix[u, x] = alpha[u] * np.cos((np.pi / N) * (x + 0.5) * u)

    return dct_matrix
    
dct_matrix_8 = dct_matrix()

def apply_dct(block : np.ndarray) -> np.ndarray:

    dct_applied = dct_matrix_8 @ block @ dct_matrix_8.T

    return dct_applied

def invert_dct(block : np.ndarray) -> np.ndarray:
    dct_applied = dct_matrix_8.T @ block @ dct_matrix_8

    return dct_applied


def PSNR(I_orig : np.ndarray, I_new : np.ndarray) -> float:
    MSE = np.sum((I_orig - I_new)**2)/I_orig.size
    return 10 * np.log10(np.max(I_orig) ** 2 / MSE)

def compress_image(image, block_size=8):
    h, w = image.shape
    compressed_image = np.zeros_like(image, dtype=np.float64)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            dct_block = apply_dct(block)
            compressed_image[i:i+block_size, j:j+block_size] = dct_block

    return compressed_image

def decompress_image(compressed_image : np.ndarray, block_size=8) -> np.ndarray:
    h, w = compressed_image.shape
    decompressed_image = np.zeros_like(compressed_image, dtype=np.float64)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            dct_block = compressed_image[i:i+block_size, j:j+block_size]
            block = invert_dct(dct_block)
            decompressed_image[i:i+block_size, j:j+block_size] = block

    return np.clip(decompressed_image, 0, 255).astype(np.uint8)

def main():
    # No size conversion (by appending or removing pixels) is performed.
    # It's up to you to use images of satisfactory (multiples of 8, but best if powers of 2) size.
    img = plt.imread(
        # "cameraman.tif"
        'lena_gray.bmp'
    )
    plt.figure(figsize=(10, 10))
    plt.title("Початкове зображення")
    plt.imshow(img, cmap='gray')
    plt.show()
    print("\nCompressing...")
    c_img = compress_image(img)
    print("Done.")
    plt.figure(figsize=(10, 10))
    plt.title("Стиснене зображення")
    plt.imshow(c_img, cmap='gray')
    plt.show()
    print("\nDecompressing...")
    d_img = decompress_image(c_img)
    print("Done.")
    plt.figure(figsize=(10, 10))
    plt.title("Відновлене зображення")
    plt.imshow(d_img, cmap='gray')
    plt.show()
    print("\nPSNR:", PSNR(img, d_img))
main()
