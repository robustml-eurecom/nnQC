"""MCP server exposing nnQC segmentation quality control to AI agents.

Lets any MCP-capable agent (Claude, Codex, ...) score a segmentation mask
against its scan and get back a quality estimate, per-class breakdown and a
per-slice profile that localises where a mask goes wrong - all without ground
truth.

Run it::

    python -m nnqc.mcp_server                 # stdio transport

**It needs a GPU.** ``nnqc/xa.py`` calls ``.cuda()`` unconditionally, so there
is no CPU path; on Jean Zay run it inside a GPU allocation
(``slurm/mcp_server.sh``). The model is loaded lazily on the first ``check``
call and cached, so start-up is instant and the ~1 GB of weights is only paid
for if a tool is actually used.

Measured footprint for the 2-D pipeline: **2.6 GB peak for 16 slices**, so a
single volume fits comfortably alongside other work on one card.

Tools
-----
``list_tasks``       which trained organ models are available, and their classes
``check_mask``       QC a scan + candidate mask pair
``explain_qc_score`` how to read a score, including the failure modes measured
                     in results/FINDINGS.md
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

NNQC_ROOT = Path(os.environ.get("NNQC_ROOT", Path(__file__).resolve().parent.parent))
CONFIG_ROOT = NNQC_ROOT / "configs" / "jz"

# Human-readable class names per task, so an agent gets "peripheral zone"
# rather than "class 1".
CLASS_NAMES = {
    "prostate": ["background", "peripheral zone", "transition zone"],
    "cardiac": ["background", "right ventricle", "myocardium", "left ventricle"],
    "liver": ["background", "liver"],
}

_MODEL_CACHE: dict[str, object] = {}
server = Server("nnqc")


def _available_tasks() -> list[dict]:
    """Tasks that have both a config pair and trained diffusion weights."""
    out = []
    if not CONFIG_ROOT.is_dir():
        return out
    for d in sorted(CONFIG_ROOT.iterdir()):
        cfg, env = d / "config.json", d / "env.json"
        if not (cfg.is_file() and env.is_file()):
            continue
        try:
            e = json.loads(env.read_text())
        except Exception:
            continue
        md = Path(e.get("model_dir", ""))
        ready = (md / "autoencoder.pt").is_file() and (md / "diffusion_unet.pt").is_file()
        out.append({
            "task": d.name,
            "modality": e.get("modality"),
            "num_classes": e.get("num_classes"),
            "classes": CLASS_NAMES.get(d.name),
            "weights_ready": ready,
            "model_dir": str(md),
        })
    return out


def _run_check(task: str, image: str, mask: str, metric: str, num_steps: int) -> dict:
    """Blocking nnQC inference. Called in a thread so the event loop stays free.

    CRITICAL: on the stdio transport, **stdout is the JSON-RPC channel**. nnqc
    prints progress with bare ``print()`` ("[nnqc] check | device=cuda ..."),
    which lands in that stream and corrupts the protocol - the client fails with
    ``Invalid JSON: expected ident``. Everything nnqc emits is therefore
    redirected to stderr, where it stays visible as diagnostics without being
    parsed as a message.
    """
    import nnqc

    cfg_p = CONFIG_ROOT / task / "config.json"
    env_p = CONFIG_ROOT / task / "env.json"
    if not cfg_p.is_file():
        raise ValueError(f"unknown task {task!r}; available: "
                         f"{[t['task'] for t in _available_tasks()]}")
    for p, what in ((image, "image"), (mask, "mask")):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{what} not found: {p}")

    with contextlib.redirect_stdout(sys.stderr):
        res = nnqc.check(image, mask, config=str(cfg_p), env=str(env_p),
                         metric=metric, num_steps=num_steps, return_volume=False)

    names = CLASS_NAMES.get(task, [])
    per_class = {
        (names[k] if k < len(names) else f"class {k}"): round(v, 4)
        for k, v in (res.qc_score_per_class or {}).items()
    }
    # An empty dict reads as "data missing" to an agent. Binary tasks genuinely
    # have no per-class breakdown - there is one foreground class and its score
    # IS qc_score - so say that rather than returning {}.
    per_class_note = None
    if not per_class:
        fg = [n for n in names[1:]] or ["foreground"]
        per_class_note = (
            f"Binary task: a single foreground class ({fg[0]}), so qc_score is "
            "already the per-class score. Not missing data."
        )

    # Per-slice profile, summarised: an agent wants "which slices look wrong",
    # not 200 raw numbers.
    worst = []
    if res.slice_scores is not None and res.slice_ratios is not None:
        import numpy as np
        s, r = np.asarray(res.slice_scores), np.asarray(res.slice_ratios)
        ok = ~np.isnan(s)
        if ok.any():
            order = np.argsort(s[ok])[:5]
            worst = [{"slice_position": round(float(r[ok][i]), 3),
                      "score": round(float(s[ok][i]), 4)} for i in order]

    return {
        "task": task,
        "qc_score": round(float(res.qc_score), 4),
        "metric": res.metric_name,
        "higher_is_better": bool(res.higher_is_better),
        "per_class": per_class,
        **({"per_class_note": per_class_note} if per_class_note else {}),
        "worst_slices": worst,
        "worst_slices_note": (
            "slice_position is normalised 0 = apex to 1 = base. Low scores "
            "clustered near 0 or 1 are the known apex/base weakness (see "
            "explain_qc_score), not necessarily a fault in your mask."
        ),
        "interpretation": _interpret(float(res.qc_score), res.higher_is_better),
    }


def _interpret(score: float, higher_is_better: bool) -> str:
    if not higher_is_better:
        return ("Lower is better for this metric; compare against a reference "
                "distribution for the task rather than an absolute threshold.")
    if score >= 0.85:
        return "High agreement - the mask matches what the model reconstructs."
    if score >= 0.70:
        return "Moderate agreement - plausible, but worth a look at worst_slices."
    if score >= 0.50:
        return "Low agreement - likely a real segmentation error."
    return "Very low agreement - the mask disagrees strongly with the anatomy."


def _tools() -> list[types.Tool]:
    tasks = [t["task"] for t in _available_tasks()]
    return [
        types.Tool(
            name="list_tasks",
            description=(
                "List the organ models available for quality control, with their "
                "modality, class names, and whether trained weights are present."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="check_mask",
            description=(
                "Score a candidate segmentation mask against its scan WITHOUT ground "
                "truth. Returns an overall quality score, a per-class breakdown, and "
                "the worst slices so you can localise the error. Use this to triage "
                "segmentations, flag likely failures for review, or compare two "
                "candidate masks for the same scan."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "enum": tasks or ["prostate", "cardiac", "liver"],
                             "description": "Which organ model to use."},
                    "image": {"type": "string", "description": "Path to the scan (NIfTI)."},
                    "mask": {"type": "string",
                             "description": "Path to the candidate segmentation (NIfTI)."},
                    "metric": {"type": "string", "default": "dice",
                               "description": "dice | iou | hd95 | assd."},
                    "num_steps": {"type": "integer", "default": 5,
                                  "description": "DDIM sampling steps; 5 is the tuned default."},
                },
                "required": ["task", "image", "mask"],
            },
        ),
        types.Tool(
            name="explain_qc_score",
            description=(
                "Explain what an nnQC score means and the measured limits of its "
                "reliability. Read this before acting on a borderline score."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


async def _dispatch(name: str, arguments: dict) -> list[types.TextContent]:
    def ok(payload):
        return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

    if name == "list_tasks":
        return ok({"tasks": _available_tasks()})

    if name == "explain_qc_score":
        return ok({
            "what_it_is": (
                "nnQC reconstructs the segmentation it believes is correct from the "
                "scan plus your candidate mask, then reports the agreement between "
                "your mask and that reconstruction. Ground truth is never used, so "
                "it works on unlabelled data."
            ),
            "measured_reliability": {
                "liver": "Spearman 0.89 vs true Dice, MAE 0.088 (15 volumes, 75 candidates)",
                "cardiac": "Spearman 0.71 (early checkpoint, 20 volumes)",
                "prostate": "Spearman 0.77 (5 volumes only - treat with caution)",
            },
            "known_limits": [
                "Apex and base slices score lower than mid-organ ones, partly because "
                "Dice punishes a fixed boundary error harder on a small structure. "
                "Measured apex/mid ratio: liver 0.82, cardiac 0.76, prostate 0.66.",
                "A class absent from the ground truth but present in the mask (or vice "
                "versa) produces a degenerate per-class score; read per_class alongside "
                "the overall number.",
                "The score is an estimate of agreement, not a certificate. Use it to "
                "rank and triage, not as a pass/fail gate on its own.",
            ],
            "how_to_act": (
                "Rank candidates by qc_score and review the lowest. worst_slices gives "
                "the slice positions (0 = apex, 1 = base) most worth opening first."
            ),
        })

    if name == "check_mask":
        try:
            result = await asyncio.to_thread(
                _run_check,
                arguments["task"], arguments["image"], arguments["mask"],
                arguments.get("metric", "dice"), int(arguments.get("num_steps", 5)),
            )
            return ok(result)
        except Exception as exc:  # surface the reason, not a bare failure
            return ok({"error": f"{type(exc).__name__}: {exc}",
                       "hint": "Check that the task has trained weights (list_tasks) "
                               "and that a CUDA device is visible - nnQC has no CPU path."})

    return ok({"error": f"unknown tool {name!r}"})


# mcp 2.0 registers handlers explicitly rather than via decorators. A handler is
# `(ctx, params) -> result`, and `params_type` is the *params* model the incoming
# request is validated against - not the request model.
async def _handle_list_tools(_ctx, _params) -> types.ListToolsResult:
    return types.ListToolsResult(tools=_tools())


async def _handle_call_tool(_ctx, params) -> types.CallToolResult:
    content = await _dispatch(params.name, dict(params.arguments or {}))
    is_error = False
    try:
        is_error = "error" in json.loads(content[0].text)
    except Exception:
        pass
    return types.CallToolResult(content=content, is_error=is_error)


server.add_request_handler("tools/list", types.PaginatedRequestParams, _handle_list_tools)
server.add_request_handler("tools/call", types.CallToolRequestParams, _handle_call_tool)


async def _main() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


class _StdoutGuard:
    """Fail loudly in tests if anything writes to the protocol stream.

    Only active when NNQC_MCP_STRICT_STDOUT=1; used by the test harness to catch
    a new chatty dependency before it reaches an agent.
    """

    def __init__(self, real):
        self._real = real

    def write(self, data):
        if data.strip() and not data.lstrip().startswith(("{", "[")):
            raise RuntimeError(
                f"non-JSON written to the MCP stdout transport: {data[:80]!r}. "
                "Redirect library output to stderr."
            )
        return self._real.write(data)

    def __getattr__(self, name):
        return getattr(self._real, name)


def main() -> None:
    if os.environ.get("NNQC_MCP_STRICT_STDOUT") == "1":
        sys.stdout = _StdoutGuard(sys.stdout)
    asyncio.run(_main())


if __name__ == "__main__":
    main()
