# nnQC MCP server demo

This page walks through the nnQC [MCP](https://modelcontextprotocol.io) server:
how to start it, and what real JSON-RPC exchanges with its three tools look
like. The server is `nnqc/mcp_server.py`; it needs a GPU (the conditioning
backbone is CUDA-only).

## Starting the server

```bash
python -m nnqc.mcp_server
```

The server speaks JSON-RPC over stdio. stdout is reserved for the protocol,
so nnqc's progress prints are redirected to stderr; set
`NNQC_MCP_STRICT_STDOUT=1` to turn any stray non-JSON stdout write into an
error (useful in tests). The model is loaded lazily on the first `check_mask`
call, so start-up is instant.

For a desktop client (Claude Desktop style), register it as:

```json
{
  "mcpServers": {
    "nnqc": {
      "command": "python",
      "args": ["-m", "nnqc.mcp_server"]
    }
  }
}
```

## Tool 1: `list_tasks`

Request:

```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
 "params": {"name": "list_tasks", "arguments": {}}}
```

Response (formatted; the `content[0].text` payload is this JSON):

```json
{
  "tasks": [
    {"task": "cardiac", "modality": "mri", "num_classes": 4,
     "classes": ["background", "right ventricle", "myocardium", "left ventricle"],
     "weights_ready": true, "model_dir": "trained_weights/cardiac"},
    {"task": "liver", "modality": "ct", "num_classes": 1,
     "classes": ["background", "liver"],
     "weights_ready": true, "model_dir": "trained_weights/liver"},
    {"task": "prostate", "modality": "mri", "num_classes": 3,
     "classes": ["background", "peripheral zone", "transition zone"],
     "weights_ready": true, "model_dir": "trained_weights/prostate"}
  ]
}
```

`weights_ready: true` means the checkpoints are present on disk; `false` means
the task's weights still need downloading (`nnqc download <task>`).

## Tool 2: `check_mask`

Request:

```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/call",
 "params": {"name": "check_mask",
            "arguments": {"task": "prostate",
                          "image": "data/case_01_img.nii.gz",
                          "mask": "data/case_01_seg.nii.gz"}}}
```

Response (example):

```json
{
  "task": "prostate",
  "qc_score": 0.8712,
  "metric": "dice",
  "higher_is_better": true,
  "per_class": {"peripheral zone": 0.8541, "transition zone": 0.9033},
  "worst_slices": [
    {"slice_position": 0.03, "score": 0.612},
    {"slice_position": 0.97, "score": 0.655},
    {"slice_position": 0.91, "score": 0.802}
  ],
  "worst_slices_note": "slice_position is normalised 0 = apex to 1 = base. Low scores clustered near 0 or 1 are the known apex/base weakness (see explain_qc_score), not necessarily a fault in your mask.",
  "interpretation": "High agreement - the mask matches what the model reconstructs."
}
```

Optional arguments: `metric` (`dice`, `iou`, `hd95`, `assd`) and `num_steps`
(DDIM sampling steps, default 5). Binary tasks return a `per_class_note`
instead of a breakdown, because the overall score already is the single
foreground class score.

## Tool 3: `explain_qc_score`

Request:

```json
{"jsonrpc": "2.0", "id": 3, "method": "tools/call",
 "params": {"name": "explain_qc_score", "arguments": {}}}
```

Response (abbreviated):

```json
{
  "what_it_is": "nnQC reconstructs the segmentation it believes is correct from the scan plus your candidate mask, then reports the agreement between your mask and that reconstruction. Ground truth is never used, so it works on unlabelled data.",
  "measured_reliability": {
    "liver": "Spearman 0.89 vs true Dice, MAE 0.088 (15 volumes, 75 candidates)",
    "cardiac": "Spearman 0.71 (early checkpoint, 20 volumes)",
    "prostate": "Spearman 0.77 (5 volumes only - treat with caution)"
  },
  "known_limits": [
    "Apex and base slices score lower than mid-organ ones ...",
    "A class absent from the ground truth but present in the mask (or vice versa) produces a degenerate per-class score ...",
    "The score is an estimate of agreement, not a certificate ..."
  ],
  "how_to_act": "Rank candidates by qc_score and review the lowest. worst_slices gives the slice positions (0 = apex, 1 = base) most worth opening first."
}
```

The intent is that an agent reads this before treating a borderline score as a
verdict.

## What the scores look like in practice

The calibration plots in `results/` show the QC score against the true Dice
(versus ground truth) for synthetically corrupted masks, for example
`results/calibration_liver_final.png`, `results/calibration_prostate_final.png`
and `results/calibration_cardiac_final.png`. A well-calibrated panel hugs the
diagonal: the QC score tracks the true segmentation quality, which is what lets
`check_mask` rank candidate masks without ground truth. The measured caveats
(apex/base weakness, synthetic vs real corruptions) are documented in
`results/FINDINGS.md` and summarised by `explain_qc_score`.

## Notes

- Weight download: `nnqc download <task>` fetches checkpoints from Zenodo into
  `trained_weights/<task>/`; `check_mask` can also auto-download on first use.
- Errors are returned as structured payloads (`{"error": ..., "hint": ...}`)
  with `isError` set, so clients see the reason rather than a bare failure.
