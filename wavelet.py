import numpy as np
import matplotlib.pyplot as plt


def PSNR(I_orig : np.ndarray, I_new : np.ndarray) -> float:
    MSE = np.sum((I_orig - I_new)**2)/I_orig.size
    return 10 * np.log10(np.max(I_orig) ** 2 / MSE)

def haar_1d(signal):
    """
    Perform the 1D Haar wavelet transform on an array.
    """
    N = len(signal)
    output = np.zeros(N)
    
    half_N = N // 2
    for i in range(half_N):
        output[i] = (signal[2 * i] + signal[2 * i + 1]) / np.sqrt(2)
        output[half_N + i] = (signal[2 * i] - signal[2 * i + 1]) / np.sqrt(2)
    
    return output

def inverse_haar_1d(transformed_signal):
    """
    Perform the inverse 1D Haar wavelet transform.
    """
    N = len(transformed_signal)
    output = np.zeros(N)
    
    half_N = N // 2
    for i in range(half_N):
        output[2 * i] = (transformed_signal[i] + transformed_signal[half_N + i]) / np.sqrt(2)
        output[2 * i + 1] = (transformed_signal[i] - transformed_signal[half_N + i]) / np.sqrt(2)
    
    return output

def haar_2d(image, levels=1):
    """
    Perform the 2D Haar wavelet transform on an image.
    The transform is applied up to the specified number of levels.
    """
    transformed_image = image.copy().astype('float64')
    h, w = transformed_image.shape

    for level in range(levels):
        # Apply the 1D Haar transform to rows
        for i in range(h):
            transformed_image[i, :w] = haar_1d(transformed_image[i, :w])

        # Apply the 1D Haar transform to columns
        for j in range(w):
            transformed_image[:h, j] = haar_1d(transformed_image[:h, j])

        # Update the dimensions for the next level
        h //= 2
        w //= 2

    return transformed_image

def inverse_haar_2d(transformed_image, levels=1):
    """
    Perform the inverse 2D Haar wavelet transform on an image.
    """
    reconstructed_image = transformed_image.copy()
    h, w = reconstructed_image.shape

    # Iterate in reverse for the inverse transform
    for level in range(levels - 1, -1, -1):
        h //= (2 ** level) 
        w //= (2 ** level) 

        # Apply the inverse transform to columns
        for j in range(w):
            reconstructed_image[:h, j] = inverse_haar_1d(reconstructed_image[:h, j])

        # Apply the inverse transform to rows
        for i in range(h):
            reconstructed_image[i, :w] = inverse_haar_1d(reconstructed_image[i, :w])
        
        

    return reconstructed_image

def main():
    img = plt.imread(
        "cameraman.tif"
        # 'lena_gray.bmp'
    )

    plt.figure(figsize=(10, 10))
    plt.title("Початкове зображення")
    plt.imshow(img, cmap='gray')
    plt.show()

    levels = 1
 
    print("\nCompressing...")
    c_img = haar_2d(img, levels=levels)
    print("Done.")
    plt.figure(figsize=(10, 10))
    plt.title("Стиснене зображення")
    plt.imshow(c_img, cmap='gray')
    plt.show()

    print("\nDecompressing...")
    d_img = inverse_haar_2d(c_img, levels=levels)
    print("Done.")
    plt.figure(figsize=(10, 10))
    plt.title("Відновлене зображення")
    plt.imshow(d_img, cmap='gray')
    plt.show()

    print("\nPSNR:", PSNR(img, d_img))

main()