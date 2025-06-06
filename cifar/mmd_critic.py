### FROM : https://github.com/maxidl/MMD-critic/blob/main/mmd_critic.py

import torch

def select_prototypes(K, num_prototypes):
    sample_indices = torch.arange(0, K.shape[0])
    num_samples = sample_indices.shape[0]

    colsum = 2 * K.sum(0) / num_samples
    is_selected = torch.zeros_like(sample_indices)
    selected = sample_indices[is_selected > 0]

    for i in range(num_prototypes):
        candidate_indices = sample_indices[is_selected == 0]
        s1 = colsum[candidate_indices]

        if selected.shape[0] == 0:
            s1 -= K.diagonal()[candidate_indices].abs()
        else:
            temp = K[selected, :][:, candidate_indices]
            s2 = temp.sum(0) * 2 + K.diagonal()[candidate_indices]
            s2 /= selected.shape[0] + 1
            s1 -= s2

        best_sample_index = candidate_indices[s1.argmax()]
        is_selected[best_sample_index] = i + 1
        selected = sample_indices[is_selected > 0]

    selected_in_order = selected[is_selected[is_selected > 0].argsort()]
    return selected_in_order

