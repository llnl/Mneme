// Adapted from HeCBench; see NOTICE and licenses/.
#pragma once

// Each thread rotates one concentric ring in place.
__global__ void rotate_matrix_parallel(float *matrix, const int n) {
  int layer = blockIdx.x * blockDim.x + threadIdx.x;
  if (layer < n / 2) {
    int first = layer;
    int last = n - 1 - layer;
    for (int i = first; i < last; ++i) {
      int offset = i - first;

      float top = matrix[first * n + i]; // save top
      // left -> top
      matrix[first * n + i] = matrix[(last - offset) * n + first];

      // bottom -> left
      matrix[(last - offset) * n + first] = matrix[last * n + (last - offset)];

      // right -> bottom
      matrix[last * n + (last - offset)] = matrix[i * n + last];

      // top -> right
      matrix[i * n + last] = top; // right <- saved top
    }
  }
}
