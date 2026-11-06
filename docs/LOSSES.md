# Age objectives, numerical policy and inference bounds

## Raw regression training

With `loss.type=l1` and `model.output_dim=1`, `AgeEstimationModel.forward` returns a
raw scalar per sample. Training minimizes `mean(abs(raw_age - target_age))` in
float32. There is no clamp in the training graph: predictions below zero or above
100 can still receive a useful gradient towards an in-range target.

The modern inference policy remains `clamp_0_100` in the checkpoint's predictor
contract. `L1AgeLoss.predict_age` applies that bound in float32. CLI, evaluation and
demo use the decoder rather than presenting model logits as final ages. Thus raw
training loss and the reported bounded prediction MAE can differ, especially early
in training; this distinction is intentional and does not indicate a metric bug.

The historical ResNet50 profile remains raw at inference as well. It has its own
decoder and is not silently subjected to the modern clamp policy.

## L1 plus central variance: the repository variant

The default config now names the objective `l1_mean_variance` and the Python class
is `L1MeanVarianceLoss`. The older config spelling `mean_variance` and Python import
`MeanVarianceLoss` remain compatible aliases of this same variant.

For logits `z`, inclusive integer ages `a_j`, and targets `y_i`:

```text
p_ij = softmax(z_i)_j
mu_i = sum_j(p_ij * a_j)
v_i  = sum_j(p_ij * (a_j - mu_i)^2)

L_mean = mean_i(abs(mu_i - y_i))
L_var  = mean_i(v_i)
L      = lambda_mean * L_mean + lambda_variance * L_var
```

Targets can be fractional ages inside the configured support. The factory checks
that the number of logits matches the support. Prediction is the expected age
`mu_i`; it is not the largest-probability class and is not separately rounded in
evaluation. Variance encourages concentration, including concentration on an
incorrect age; it is not a guarantee against confident wrong predictions.

This is **not an exact implementation of Pan et al., CVPR 2018**: the published
joint objective uses a squared mean-error term and a softmax classification loss
alongside its variance term. This project uses an absolute mean-error term and
does not include that classification term. The reference informs the method;
its results are not this repository's benchmark. See the
[original paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Pan_Mean-Variance_Loss_for_CVPR_2018_paper.pdf).

No controlled UTKFace ablation establishes an advantage over L1/MSE here. Such a
comparison requires a fixed split, training budget, model-selection policy and
multiple runs as described in P4.1. A separate paper-faithful mode is not added in
P1.4; it would need its own formula, tests, label policy and comparison protocol.

## Numerical implementation

Softmax explicitly requests float32; expected age, central variance, L1 arithmetic
and modern age decoding also use float32. Casting is differentiable back to the
model output. The numerical policy is the same for float16/bfloat16 inputs and
under autocast. Even float64 input logits are intentionally evaluated using the
declared float32 objective; float64 is used independently as a testing oracle.

Central variance avoids subtracting `E[a]^2` from the nearly equal `E[a^2]`, which
can lose small spreads through cancellation. It is a sum of nonnegative weighted
squares and does not require clamping away a negative computed variance. The
historical `ages_sq` loss buffer is retained only for state_dict compatibility;
the calculation no longer uses it. The mathematical L1+variance objective is
unchanged in real arithmetic, but floating-point results can differ from old runs.

Float32 accumulation cannot repair logits that have already overflowed inside the
model. Tests use finite logits, including saturated distributions; they check
finite gradients, not an assertion that every saturated softmax gradient is nonzero.
CUDA autocast/GradScaler tests are included and skip when no GPU is available.
CPU low-precision checks are not reported as CUDA verification.

## Configuration and experiment identity

```bash
train-age --print-config loss.type=l1_mean_variance
# Compatible old spelling:
train-age --print-config loss.type=mean_variance
# Raw regression training, modern bounded age predictions:
train-age --print-config loss.type=l1 model.output_dim=1
```

New Trainer checkpoints and W&B metadata record `training_objective` with a
canonical name, revision `2` and compute dtype `float32`. Distribution metadata
also identifies the absolute expected-age term and central variance. This separates
the corrected implementation from older runs even when an old config alias is used.

Generic `create_checkpoint` calls and old-weight repackaging do not invent that
training metadata. The field is optional; a missing value remains unknown. The
inference schema remains version 1 because age decoding and the saved preprocessing
contract remain compatible. Existing artifacts preserve their stored alias.

## Export and legacy compatibility

The modern model now exports raw regression logits. Consumers must apply the
sidecar's predictor policy to obtain final ages. Applying `clamp_0_100` to the
raw output produces the same final ages as the former model's in-forward clamp.
Older exports that already contain the clamp remain compatible with applying the
same idempotent decoder. The common PyTorch pipeline applies the policy itself.
See [checkpoint contracts](CHECKPOINTS.md).

Tests force negative and above-range heads on each supported CNN, verify nonzero
correctly directed bias gradients and a decreasing raw L1 loss, and separately
verify bounded ages. Distribution tests cover uniform, concentrated and two-point
distributions, shifted support, large-offset cancellation stress, float64 value/
gradient comparisons and finite low-precision inputs. Checkpoint/export tests
verify that a negative raw regression logit stays raw in TorchScript and becomes
zero only after the modern decoder; the legacy decoder remains raw.
