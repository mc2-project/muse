import sympy
import numpy as np


def get_indices(X_shape, HF, WF, stride, pad):
    """
        Returns index matrices in order to transform our input image into a
        matrix.

        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.

        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d.
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1

    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(HF), WF)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----

    # Slide 1 vector.
    slide1 = np.tile(np.arange(WF), HF)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d


def im2col(X, HF, WF, stride, pad):
    """
        Transforms our input image into a matrix.

        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.

        Returns:
        -cols: output matrix.
    """
    # Padding
    X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols


def evaluate_conv_layer(kernel, X, stride, padding):
    """
        Performs a forward convolution.

        Parameters:
        - X : Last conv layer of shape (m, n_C_prev, n_H_prev, n_W_prev).
        Returns:
        - out: previous layer convolved.
    """
    (m, n_C_prev, n_H_prev, n_W_prev) = X.shape

    (n_F, _, f, f) = kernel.shape

    n_C = n_F
    n_H = int((n_H_prev + 2 * padding - f) / stride) + 1
    n_W = int((n_W_prev + 2 * padding - f) / stride) + 1

    X_col = im2col(X, f, f, stride, padding)
    w_col = kernel.reshape((n_F, -1))
    # Perform matrix multiplication.
    out = w_col @ X_col
    # Reshape back matrix to image.
    out = np.array(np.hsplit(out, m)).reshape((m, n_C, n_H, n_W))
    return out


def random_kernel(size):
    num_channels = 1
    num_filters = 1
    return (np.random.randn(num_filters, num_channels, size, size)
            * np.sqrt(1. / size))


def random_input(size):
    batch_size = 1
    num_channels = 1
    inp = np.random.rand(batch_size, num_channels, size, size)
    return inp


def output_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1;


if __name__ == '__main__':
    stride = 2
    padding = 1
    kernel_size = 3
    input_size = 4
    output_size = output_size(input_size, kernel_size, stride, padding)
    num_classes = 10

    eqn_size = min(num_classes, kernel_size * kernel_size);

    # let's focus on 3x3 kernels
    kernel = random_kernel(kernel_size)
    inp = random_input(input_size)
    output = evaluate_conv_layer(kernel, inp, stride, padding)
    output = output.reshape(output_size**2)
    coeffs = im2col(inp, kernel_size, kernel_size, stride, padding)

    (_, inds) = sympy.Matrix(coeffs).rref()
    eqn_coeffs = []
    eqn_outputs = []
    for col in inds:
        eqn_coeffs.append(coeffs[:, col])
        eqn_outputs.append(output[col])
    eqn_coeffs = np.array(eqn_coeffs)
    eqn_outputs = np.array(eqn_outputs)

    print(output.shape)
    print(eqn_outputs.shape)
    print(eqn_coeffs.shape)
    result = np.linalg.solve(eqn_coeffs, eqn_outputs)
    print("Recovered result:")
    print(result)
    print("Kernel:")
    print(kernel)
