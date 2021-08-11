import numpy as np
import sys


def evaluate_linear_layer(layer, state):
    return np.dot(layer, state)


def evaluate_relu(state):
    return (state > 0) * state


def evaluate_malleated_relu(state, shift=20):
    state = state + shift
    state = (state > 0) * state
    state = state - shift
    return state


def evaluate_masked_relu(state, mask_start, mask_stop, shift=20):
    mask = np.full(state.shape, -shift)
    mask[mask_start:mask_stop] = shift
    state = state + mask
    state = (state > 0) * state
    mask = (mask > 0) * mask
    state = state - mask
    return state


def evaluate_network_upto(linear_layers, state, stop_at):
    for (i, layer) in enumerate(linear_layers[:stop_at]):
        state = evaluate_linear_layer(layer, state)
        # only perform ReLU if we're not the last layer
        if i != len(layer) - 1:
            state = evaluate_relu(state)
    return state


def evaluate_network_after_malleation(linear_layers, state, start_at,
                                      shift=20):
    for (i, layer) in enumerate(linear_layers[start_at:]):
        state = evaluate_linear_layer(layer, state)
        # only perform ReLU if we're not the last layer
        if i != len(layer) - 1:
            state = evaluate_malleated_relu(state, shift)
    return state


def unit_vector(dim, i):
    vec = np.zeros(dim)
    vec[i] = 1.0
    return vec


def extract_network(linear_layers):
    starting_dim = linear_layers[0].shape[1]
    num_classes = linear_layers[-1].shape[0]
    initial_state = np.zeros((starting_dim, 1))

    extracted_layers = []
    num_queries = 0
    # We iterate in reverse.
    for (i, layer) in list(enumerate(linear_layers))[::-1]:
        (num_rows, num_cols) = layer.shape
        extracted_layer = np.zeros(layer.shape)
        # If we haven't extracted the last layer yet:
        if len(extracted_layers) == 0:
            # this is the simple case
            for col in range(0, num_cols):
                state = initial_state
                num_queries += 1
                last_state = evaluate_network_upto(linear_layers, state, i)
                # At this point, the `last_state` should be all-zero vector.
                # To extract the column number `col`, we set the col-th column
                # of `last_state` to be 1.
                last_state = last_state + unit_vector(last_state.shape, col)

                result = evaluate_linear_layer(layer, last_state)

                # update extracted_layer with results
                for row in range(0, num_rows):
                    extracted_layer[row, col] = result[row, 0]
        else:
            # we are now recovering intermediate layers
            next_matrix = np.identity(linear_layers[i+1].shape[1])
            for _layer in linear_layers[i + 1:]:
                next_matrix = np.dot(_layer, next_matrix)
            assert(next_matrix.shape[0] == num_classes)

            for col in range(0, num_cols):
                for row in range(0, num_rows, num_classes):
                    state = initial_state
                    num_queries += 1
                    state = evaluate_network_upto(linear_layers, state, i)
                    # At this point, the `last_state` should be all-zero
                    # vector.
                    # 
                    # To extract elements of the column `col`, we set the col-th
                    # column of `last_state` to be 1.
                    state = state + unit_vector(state.shape, col)

                    state = evaluate_linear_layer(layer, state)

                    # At this point, we have all the rows in column i.
                    # However, because eventually we'll only obtain information
                    # about `num_classes` rows at a time, we mask out the rest.
                    start = (num_rows - num_classes
                             if row + num_classes > num_rows
                             else row)

                    end = min(row + num_classes, num_rows)
                    state = evaluate_masked_relu(state, start, end)

                    # evaluate the rest of the network
                    result = evaluate_network_after_malleation(linear_layers, state, i + 1)
                    sub_matrix = next_matrix[:, start:end]
                    result = np.linalg.solve(sub_matrix, result)
                    extracted_layer[start:end, col] = result.reshape((num_classes,))
        extracted_layers.append(extracted_layer)
    extracted_layers.reverse()
    print(num_queries)
    return extracted_layers


if __name__ == '__main__':
    sizes = list(map(int, sys.argv[1].split("-")))
    dimensions = [tuple([x]) for x in sizes]
    layers = []
    for (row, col) in zip(sizes[1:], sizes):
        layers.append(np.random.rand(row, col))

    extracted_layers = extract_network(layers)

    for (layer, extracted_layer) in zip(layers, extracted_layers):
        assert(np.allclose(layer, extracted_layer))
