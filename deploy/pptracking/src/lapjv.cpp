//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The code is based on:
// https://github.com/gatagat/lap/blob/master/lap/lapjv.cpp
// Ths copyright of gatagat/lap is as follows:
// MIT License

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/lapjv.h"

namespace PaddleDetection {

/** Column-reduction and reduction transfer for a dense cost matrix.
 */
int _ccrrt_dense(
    const int n, float *cost[], int *free_rows, int *x, int *y, float *v) {
  int n_free_rows;
  bool *unique;

  for (int i = 0; i < n; i++) {
    x[i] = -1;
    v[i] = LARGE;
    y[i] = 0;
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      const float c = cost[i][j];
      if (c < v[j]) {
        v[j] = c;
        y[j] = i;
      }
    }
  }
  NEW(unique, bool, n);
  memset(unique, TRUE, n);
  {
    int j = n;
    do {
      j--;
      const int i = y[j];
      if (x[i] < 0) {
        x[i] = j;
      } else {
        unique[i] = FALSE;
        y[j] = -1;
      }
    } while (j > 0);
  }
  n_free_rows = 0;
  for (int i = 0; i < n; i++) {
    if (x[i] < 0) {
      free_rows[n_free_rows++] = i;
    } else if (unique[i]) {
      const int j = x[i];
      float min = LARGE;
      for (int j2 = 0; j2 < n; j2++) {
        if (j2 == static_cast<int>(j)) {
          continue;
        }
        const float c = cost[i][j2] - v[j2];
        if (c < min) {
          min = c;
        }
      }
      v[j] -= min;
    }
  }
  FREE(unique);
  return n_free_rows;
}

/** Augmenting row reduction for a dense cost matrix.
 */
int _carr_dense(const int n,
                float *cost[],
                const int n_free_rows,
                int *free_rows,
                int *x,
                int *y,
                float *v) {
  int current = 0;
  int new_free_rows = 0;
  int rr_cnt = 0;
  while (current < n_free_rows) {
    int i0;
    int j1, j2;
    float v1, v2, v1_new;
    bool v1_lowers;

    rr_cnt++;
    const int free_i = free_rows[current++];
    j1 = 0;
    v1 = cost[free_i][0] - v[0];
    j2 = -1;
    v2 = LARGE;
    for (int j = 1; j < n; j++) {
      const float c = cost[free_i][j] - v[j];
      if (c < v2) {
        if (c >= v1) {
          v2 = c;
          j2 = j;
        } else {
          v2 = v1;
          v1 = c;
          j2 = j1;
          j1 = j;
        }
      }
    }
    i0 = y[j1];
    v1_new = v[j1] - (v2 - v1);
    v1_lowers = v1_new < v[j1];
    if (rr_cnt < current * n) {
      if (v1_lowers) {
        v[j1] = v1_new;
      } else if (i0 >= 0 && j2 >= 0) {
        j1 = j2;
        i0 = y[j2];
      }
      if (i0 >= 0) {
        if (v1_lowers) {
          free_rows[--current] = i0;
        } else {
          free_rows[new_free_rows++] = i0;
        }
      }
    } else {
      if (i0 >= 0) {
        free_rows[new_free_rows++] = i0;
      }
    }
    x[free_i] = j1;
    y[j1] = free_i;
  }
  return new_free_rows;
}

/** Find columns with minimum d[j] and put them on the SCAN list.
 */
int _find_dense(const int n, int lo, float *d, int *cols, int *y) {
  int hi = lo + 1;
  float mind = d[cols[lo]];
  for (int k = hi; k < n; k++) {
    int j = cols[k];
    if (d[j] <= mind) {
      if (d[j] < mind) {
        hi = lo;
        mind = d[j];
      }
      cols[k] = cols[hi];
      cols[hi++] = j;
    }
  }
  return hi;
}

// Scan all columns in TODO starting from arbitrary column in SCAN
// and try to decrease d of the TODO columns using the SCAN column.
int _scan_dense(const int n,
                float *cost[],
                int *plo,
                int *phi,
                float *d,
                int *cols,
                int *pred,
                int *y,
                float *v) {
  int lo = *plo;
  int hi = *phi;
  float h, cred_ij;

  while (lo != hi) {
    int j = cols[lo++];
    const int i = y[j];
    const float mind = d[j];
    h = cost[i][j] - v[j] - mind;
    // For all columns in TODO
    for (int k = hi; k < n; k++) {
      j = cols[k];
      cred_ij = cost[i][j] - v[j] - h;
      if (cred_ij < d[j]) {
        d[j] = cred_ij;
        pred[j] = i;
        if (cred_ij == mind) {
          if (y[j] < 0) {
            return j;
          }
          cols[k] = cols[hi];
          cols[hi++] = j;
        }
      }
    }
  }
  *plo = lo;
  *phi = hi;
  return -1;
}

/** Single iteration of modified Dijkstra shortest path algorithm as explained
 * in the JV paper.
 *
 * This is a dense matrix version.
 *
 * \return The closest free column index.
 */
int find_path_dense(const int n,
                    float *cost[],
                    const int start_i,
                    int *y,
                    float *v,
                    int *pred) {
  int lo = 0, hi = 0;
  int final_j = -1;
  int n_ready = 0;
  int *cols;
  float *d;

  NEW(cols, int, n);
  NEW(d, float, n);

  for (int i = 0; i < n; i++) {
    cols[i] = i;
    pred[i] = start_i;
    d[i] = cost[start_i][i] - v[i];
  }
  while (final_j == -1) {
    // No columns left on the SCAN list.
    if (lo == hi) {
      n_ready = lo;
      hi = _find_dense(n, lo, d, cols, y);
      for (int k = lo; k < hi; k++) {
        const int j = cols[k];
        if (y[j] < 0) {
          final_j = j;
        }
      }
    }
    if (final_j == -1) {
      final_j = _scan_dense(n, cost, &lo, &hi, d, cols, pred, y, v);
    }
  }

  {
    const float mind = d[cols[lo]];
    for (int k = 0; k < n_ready; k++) {
      const int j = cols[k];
      v[j] += d[j] - mind;
    }
  }

  FREE(cols);
  FREE(d);

  return final_j;
}

/** Augment for a dense cost matrix.
 */
int _ca_dense(const int n,
              float *cost[],
              const int n_free_rows,
              int *free_rows,
              int *x,
              int *y,
              float *v) {
  int *pred;

  NEW(pred, int, n);

  for (int *pfree_i = free_rows; pfree_i < free_rows + n_free_rows; pfree_i++) {
    int i = -1, j;
    int k = 0;

    j = find_path_dense(n, cost, *pfree_i, y, v, pred);
    while (i != *pfree_i) {
      i = pred[j];
      y[j] = i;
      SWAP_INDICES(j, x[i]);
      k++;
    }
  }
  FREE(pred);
  return 0;
}

/** Solve dense sparse LAP.
 */
int lapjv_internal(const cv::Mat &cost,
                   const bool extend_cost,
                   const float cost_limit,
                   int *x,
                   int *y) {
  int n_rows = cost.rows;
  int n_cols = cost.cols;
  int n;
  if (n_rows == n_cols) {
    n = n_rows;
  } else if (!extend_cost) {
    throw std::invalid_argument(
        "Square cost array expected. If cost is intentionally non-square, pass "
        "extend_cost=True.");
  }

  // Get extend cost
  if (extend_cost || cost_limit < LARGE) {
    n = n_rows + n_cols;
  }
  cv::Mat cost_expand(n, n, CV_32F);
  float expand_value;
  if (cost_limit < LARGE) {
    expand_value = cost_limit / 2;
  } else {
    double max_v;
    minMaxLoc(cost, nullptr, &max_v);
    expand_value = static_cast<float>(max_v) + 1.;
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      cost_expand.at<float>(i, j) = expand_value;
      if (i >= n_rows && j >= n_cols) {
        cost_expand.at<float>(i, j) = 0;
      } else if (i < n_rows && j < n_cols) {
        cost_expand.at<float>(i, j) = cost.at<float>(i, j);
      }
    }
  }

  // Convert Mat to pointer array
  float **cost_ptr;
  NEW(cost_ptr, float *, n);
  for (int i = 0; i < n; ++i) {
    NEW(cost_ptr[i], float, n);
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      cost_ptr[i][j] = cost_expand.at<float>(i, j);
    }
  }

  int ret;
  int *free_rows;
  float *v;
  int *x_c;
  int *y_c;

  NEW(free_rows, int, n);
  NEW(v, float, n);
  NEW(x_c, int, n);
  NEW(y_c, int, n);

  ret = _ccrrt_dense(n, cost_ptr, free_rows, x_c, y_c, v);
  int i = 0;
  while (ret > 0 && i < 2) {
    ret = _carr_dense(n, cost_ptr, ret, free_rows, x_c, y_c, v);
    i++;
  }
  if (ret > 0) {
    ret = _ca_dense(n, cost_ptr, ret, free_rows, x_c, y_c, v);
  }
  FREE(v);
  FREE(free_rows);
  for (int i = 0; i < n; ++i) {
    FREE(cost_ptr[i]);
  }
  FREE(cost_ptr);
  if (ret != 0) {
    if (ret == -1) {
      throw "Out of memory.";
    }
    throw "Unknown error (lapjv_internal)";
  }
  // Get output of x, y, opt
  for (int i = 0; i < n; ++i) {
    if (i < n_rows) {
      x[i] = x_c[i];
      if (x[i] >= n_cols) {
        x[i] = -1;
      }
    }
    if (i < n_cols) {
      y[i] = y_c[i];
      if (y[i] >= n_rows) {
        y[i] = -1;
      }
    }
  }

  FREE(x_c);
  FREE(y_c);
  return ret;
}

}  // namespace PaddleDetection
