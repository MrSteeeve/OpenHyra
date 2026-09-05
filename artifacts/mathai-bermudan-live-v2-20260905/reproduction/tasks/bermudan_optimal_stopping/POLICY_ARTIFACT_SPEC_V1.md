# OpenHyra Bermudan Policy Artifact Wire Format v1

Status: candidate-facing normative specification for
`openhyra-policy-spec.v1`.

This document describes the only frozen MLP policy format accepted by the
trusted v1 loader. Candidate training code may produce the data files below,
but candidate code is never imported by the evaluator. If this document and
the trusted loader disagree, the loader fails closed; the inconsistency must be
fixed before running candidates.

## 1. Submission and per-instance output are separate

The candidate submission contains `manifest.json` (and may contain candidate
training source outside the artifact output directory). For every evaluated
instance, the trusted caller passes that manifest separately and gives the
training process a fresh output directory.

For `T = n_exercise_times`, the per-instance output directory must contain
exactly:

```text
normalization.json
step_000.npy
step_001.npy
...
step_{T-2:03d}.npy
```

There are exactly `T - 1` step files, one for each non-terminal exercise time.
The first is `step_000.npy`; numbering is contiguous with no gaps. `T` must be
an integer in `[2, 1000]`.

No other directory entry is accepted. In particular:

- `training_meta.json`: **NOT ACCEPTED in v1**.
- `manifest.json`: **NOT ACCEPTED inside the per-instance output directory**;
  it is supplied separately.
- checkpoints, logs, subdirectories, temporary files, NPZ archives, and hidden
  files: **NOT ACCEPTED**.

All accepted files and the artifact root must be real filesystem objects.
Symbolic links, multiply-linked files, special files, missing files, unexpected
files, and files that change while being read are rejected.

## 2. `manifest.json`

`manifest.json` is UTF-8 JSON and must contain exactly the six top-level fields
shown below. All fields are required; unknown fields and duplicate JSON keys
are rejected.

<!-- BEGIN CANONICAL MANIFEST EXAMPLE -->
```json
{
  "schema": "openhyra-policy-spec.v1",
  "runner_type": "mlp",
  "inference_config": {
    "input_dim": "n_assets",
    "layers": [64, 64],
    "activation": "tanh",
    "output_dim": 1,
    "output_clip": [-1000000.0, 1000000.0]
  },
  "output_semantics": "discounted_continuation_value_t0",
  "normalization": "per_step",
  "weight_pattern": "step_{:03d}.npy"
}
```
<!-- END CANONICAL MANIFEST EXAMPLE -->

The exact field contract is:

| Field | Required v1 value |
|---|---|
| `schema` | Exact string `openhyra-policy-spec.v1` |
| `runner_type` | Exact string `mlp`; no other runner is registered in this wire format |
| `output_semantics` | Exact string `discounted_continuation_value_t0` |
| `normalization` | Exact string `per_step` |
| `weight_pattern` | Exact string `step_{:03d}.npy` |
| `inference_config` | Exact object described below |

`inference_config` must contain exactly these five fields:

- `input_dim`: either the exact string `"n_assets"`, or an integer in
  `[1, 4096]`. Boolean values are not integers. With `"n_assets"`, the trusted
  caller must provide the instance's actual asset count. With an integer, it
  must equal the trusted input dimension.
- `layers`: a JSON array of zero to eight hidden-layer widths. Every width is
  an integer in `[1, 4096]`. `[]` means one affine input-to-output layer.
- `activation`: exact string `"relu"` or `"tanh"`. It is applied after each
  hidden affine layer, never after the scalar output layer.
- `output_dim`: integer `1` exactly.
- `output_clip`: the exact two-number array
  `[-1000000.0, 1000000.0]`. The clip is protocol-owned, not a tunable
  hyperparameter.

All JSON numbers must be finite. JSON `NaN`, `Infinity`, `-Infinity`, booleans
where numbers are required, missing fields, and additional fields are rejected.
The manifest size limit is 65,536 bytes.

## 3. Step numbering and dimensions

Let:

- `d` be the resolved `input_dim`;
- `layers = [h1, ..., hL]` be the hidden widths; and
- `widths = [d, h1, ..., hL, 1]`.

Step index `i` corresponds exactly to trusted runner call
`continuation(time_index=i, states=...)` and therefore to non-terminal exercise
time `i`. `step_i` and `normalization.steps[i]` always form a pair. There is no
file for terminal time `T - 1` because terminal payoff is evaluator-owned.

For each layer `l = 0, ..., L`:

- `W_l` has shape `(widths[l + 1], widths[l])`, i.e. output-by-input;
- `b_l` has shape `(widths[l + 1],)`.

The exact parameter count in every step file is:

```text
P = sum_l(widths[l + 1] * widths[l] + widths[l + 1])
```

Every step uses the same architecture and therefore the same `P`. `P` must not
exceed 131,040 parameters.

## 4. `step_XXX.npy`: canonical flat weights

Each step file contains one and only one NPY array with all of these properties:

- NPY format version 1.0 or 2.0;
- shape exactly `(P,)` (one-dimensional);
- native-endian NumPy `float64` exactly;
- C-contiguous / C order (`fortran_order` is false);
- every value finite (no NaN or positive/negative infinity);
- no object dtype and no pickle;
- no bytes after the NPY array;
- file size at most 1,048,576 bytes.

NPZ files, arrays with an extra dimension, `float32`, non-native-endian `f8`,
Fortran-order headers, object arrays, and appended payloads are rejected.

### 4.1 Mandatory flattening order

For every layer, append the C-order flattened weight matrix and then that
layer's bias. Process layers from input to output:

```text
flat = concat(
    W_0.ravel(order="C"), b_0,
    W_1.ravel(order="C"), b_1,
    ...,
    W_L.ravel(order="C"), b_L,
)
```

This ordering is normative. For a framework that stores dense weights as
`(input_width, output_width)`, transpose to the required
`(output_width, input_width)` shape before C-order flattening.

Example for `d=2`, `layers=[2]`, and scalar output:

```text
W_0 shape (2,2): 4 values in row-major order
b_0 shape (2,):  2 values
W_1 shape (1,2): 2 values in row-major order
b_1 shape (1,):  1 value
P = 9
```

## 5. `normalization.json`

`normalization.json` is UTF-8 JSON with this exact schema:

```json
{
  "steps": [
    {
      "mean": [0.0, 0.0],
      "scale": [1.0, 1.0]
    },
    {
      "mean": [0.0, 0.0],
      "scale": [1.0, 1.0]
    }
  ]
}
```

Normative rules:

- The root object contains exactly `steps`.
- `steps` has exactly `T - 1` entries, ordered by step index.
- Every entry contains exactly `mean` and `scale`.
- Each `mean` and `scale` is a JSON array of exactly `d` finite numbers.
- Every scale value is strictly greater than `1e-10`.
- Duplicate keys, unknown fields, non-finite values, and booleans as numbers
  are rejected.
- File size is at most 262,144 bytes.

For step `i`, the runner computes each input component in float64 as:

```text
x_normalized[j] = (states[..., j] - mean_i[j]) / scale_i[j]
```

The trusted runner rejects non-finite states and non-finite normalized values.

## 6. Trusted inference semantics

The input shape is `(..., d)`, and the output shape is the leading `...` shape.
For each sample and each layer, the trusted runner performs the affine map
`W_l @ x + b_l` in fixed scalar float64 reduction order. It applies the
manifest activation only to hidden layers. The scalar output layer is linear,
then clipped to the closed interval `[-1000000.0, 1000000.0]`.

The returned scalar is the continuation value at that exercise step already
discounted to time zero. The runner does not apply an additional discount, does
not calculate payoff, and does not make the exercise/continue decision.
Candidate training targets and exported parameters must use these same units.

## 7. Candidate export recipe

The following is the intended export shape. `layer_parameters_by_step[i]` must
list `(W, b)` pairs from input to output, already using output-by-input `W`.

```python
import json
from pathlib import Path

import numpy as np


def export_policy(output_dir, layer_parameters_by_step, normalizations):
    output_dir = Path(output_dir)
    # The trusted caller provides a fresh, empty directory.

    for step_index, layer_parameters in enumerate(layer_parameters_by_step):
        parts = []
        for weights, bias in layer_parameters:
            weights = np.asarray(weights, dtype=np.float64)
            bias = np.asarray(bias, dtype=np.float64)
            parts.append(weights.ravel(order="C"))
            parts.append(bias.reshape(-1))
        flat = np.ascontiguousarray(np.concatenate(parts), dtype=np.float64)
        if flat.ndim != 1 or not flat.dtype.isnative:
            raise ValueError("weights must be canonical native float64")
        if not np.all(np.isfinite(flat)):
            raise ValueError("weights must be finite")
        np.save(
            output_dir / f"step_{step_index:03d}.npy",
            flat,
            allow_pickle=False,
        )

    payload = {
        "steps": [
            {
                "mean": np.asarray(mean, dtype=np.float64).tolist(),
                "scale": np.asarray(scale, dtype=np.float64).tolist(),
            }
            for mean, scale in normalizations
        ]
    }
    (output_dir / "normalization.json").write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
```

Before returning successfully, candidate code should verify that:

1. it emitted exactly `T - 1` contiguous step files;
2. each flattened array length is the derived `P`;
3. normalization has exactly `T - 1` entries of dimension `d`;
4. every weight, mean, and scale is finite and every scale is `> 1e-10`; and
5. the output directory contains no other entry.

`manifest.json` should be generated with `allow_nan=False` and retained on the
candidate submission surface, not copied into the per-instance output.

## 8. Size and version boundaries

The trusted v1 limits are:

| Object | Limit |
|---|---:|
| `manifest.json` | 65,536 bytes |
| `normalization.json` | 262,144 bytes |
| each `step_XXX.npy` | 1,048,576 bytes |
| manifest plus complete artifact bundle | 8,388,608 bytes |
| exercise times `T` | 2 to 1000 |
| resolved input dimension `d` | 1 to 4096 |
| hidden layers | 0 to 8 |
| each hidden width | 1 to 4096 |
| parameters per step | at most 131,040 |

Adding a runner type, field, file, activation, metadata file, array layout, or
new output meaning is not a compatible v1 extension. It requires an explicit
version decision and synchronized trusted-loader, specification, and contract
test changes.
