# M11 AdamC report (running log)

Pod 2i3qig1anzudm4 (secure RTX 4090, $0.89/h, started 2026-11-09 20:05 UTC), queue `experiments/m11_adamc.txt`,
seed 2022, fixed LR drop after epoch 4, commit 103b1e5. Test AUC / LogLoss, best epoch.

| Job | Valid AUC | Test AUC | Test LogLoss | Best epoch | Notes |
|---|---|---|---|---|---|
| lrdrop4_ref | 0.814606 | 0.814925 | 0.437186 | 6 of 8 | reproduces earlier `lrdrop4` exactly |
| adamw_0.01 | 0.814607 | 0.815015 | 0.437096 | 6 of 8 | +0.00009 AUC vs ref: noise |
| adamc_0.01 | 0.814607 | 0.815015 | 0.437097 | 6 of 8 | same as adamw_0.01 to 1e-6; grad_norm after drop 0.056->0.065->0.074->0.083 (same rise); mlp_weight_norm 161.4->163.4 vs 160.9->161.3 for adamw |
| adamw_0.05 | 0.814572 | 0.814996 | 0.437050 | 6 of 8 | +0.00007 AUC vs ref: noise; grad_norm after drop 0.055->0.064->0.072->0.082 (same rise as ref); mlp_weight_norm 129.6 (ep 4) -> 123.9 (ep 8), shrinking after drop vs ref 170.6 -> 173.2 |
| adamc_0.05 | 0.814573 | 0.814997 | 0.437049 | 6 of 8 | same as adamw_0.05 to 1e-6; grad_norm after drop 0.055->0.064->0.072->0.082 (same rise); mlp_weight_norm 130.2 -> 132.2 (ep 5-8) vs 128.1 -> 123.9 for adamw |
| adamw_0.2 | 0.814543 | 0.814931 | 0.437140 | 7 of 9 | +0.00001 AUC vs ref: noise; ran 9 epochs; grad_norm after drop 0.056->0.066->0.076->0.087->0.100 (steeper rise than ref, 0.084 at ep 8); mlp_weight_norm 84.5 (ep 4) -> 65.0 (ep 9), shrinking |
| adamc_0.2 | 0.814542 | 0.814937 | 0.437141 | 7 of 9 | +0.000006 AUC vs adamw_0.2; grad_norm after drop 0.056->0.065->0.075->0.086->0.098 (ep 9: 0.0976 vs 0.1000 for adamw, -2%); mlp_weight_norm 85.2 -> 88.0 (ep 5-9) vs 79.8 -> 65.0 for adamw |

## Observations so far
- Reference grad_norm (mean, pre-clip) rises after the LR drop: 0.054 (ep 4) -> 0.057, 0.066, 0.075, 0.084 (ep 8).
  Clip (10) only fires in epoch 1 (grad_norm_max 93).
- AdamW 0.01 and AdamC 0.01 are identical until the drop (lr == lr_max), as expected.
- AdamW 0.01 tracks the reference within 0.00005 val AUC at every epoch; MLP weight norm ~6% lower.
- AdamW 0.2 / AdamC 0.2 ran 9 epochs (best epoch 7), the others 8 (best 6). At 0.2 grad_norm rises faster after the drop
  than in the reference (0.100 vs 0.084 at ep 8-9 level), i.e. stronger decay makes the late rise worse, not better.

## Comparison (test AUC, single seed 2022, threshold 0.0003)

| lambda | lrdrop4_ref | adamw | adamc | adamc - adamw | grad_norm ep 8 (adamw / adamc) | mlp_weight_norm ep 8 (adamw / adamc) | passes 0.0003? |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.814925 | 0.815015 | 0.815015 | -0.000000 | 0.0828 / 0.0828 | 161.3 / 163.4 | no |
| 0.05 | 0.814925 | 0.814996 | 0.814997 | +0.000001 | 0.0817 / 0.0816 | 123.9 / 132.2 | no |
| 0.2  | 0.814925 | 0.814931 | 0.814937 | +0.000006 | 0.0870 / 0.0857 | 68.2 / 87.3 | no |

Test LogLoss: ref 0.437186; adamw 0.437096 / 0.437050 / 0.437140; adamc 0.437097 / 0.437049 / 0.437141 (all within 1e-4 of ref).

## Verdict

Negative result. AdamC does not beat AdamW at any lambda: the largest gap is +0.000006 test AUC, 50x below the 0.0003 threshold,
and no variant differs from the lrdrop4 reference by more than 0.0001 (all noise). The post-drop gradient-norm rise
(0.054 -> ~0.084 over four epochs, +57% in the reference) is essentially unchanged by AdamC: at most 2% lower at lambda 0.2.
AdamC does change the weight norm as designed (it stops the MLP weights from shrinking after the LR drop), but that has no effect
on validation or test metrics here. Plain AdamW decay also does not help (0.01 and 0.05 are +0.00007..+0.00009, noise; 0.2 is flat
and ends with a faster grad-norm rise). Likely reasons: the phenomenon is small in this short 8-9 epoch step-schedule run,
and the gradient norm is dominated by the embedding/cross parts, which AdamC does not touch. Not worth more seeds or a PROGRESS follow-up;
the `weight_decay` / `adamc` options and grad-norm logging stay as optional switches (default off, no-op).
