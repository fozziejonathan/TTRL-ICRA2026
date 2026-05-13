"""
add_t1_paddle_to_k1.py
======================

Add the T1 ping-pong paddle / 3D-printed right-hand attachment onto Booster K1
as a child Xform named `paddle_adapter` under K1's existing `right_hand_link`.

What this does (and does NOT do):
  * Does     create   <K1_RIGHT_HAND_LINK>/paddle_adapter (Xform)
  * Does     copy     <T1_RIGHT_HAND_LINK>/{marker_ball, visuals, collisions}
                      under that adapter, preserving each prim's attributes,
                      references, payloads, and applied API schemas
                      (UsdPhysics.CollisionAPI etc.).
  * Does NOT copy or replace `right_hand_link` itself, so K1's joint tree,
    articulation, and DoF count are untouched.
  * Does NOT add new rigid bodies or joints. The paddle collisions become
    children of K1's existing right-hand RigidBody, which is the desired
    behavior in PhysX/IsaacSim.

Why a script instead of the Isaac Sim UI:
  When the source paddle prims live inside a referenced/instanced subtree,
  the UI fails with:
      "Cannot create prim spec at path ... authoring to an instance proxy
       is not allowed."
  This script avoids that by:
    (a) trying Sdf.CopySpec directly from T1's root layer first, and
    (b) falling back to a flattened in-memory layer for any prim that
        wasn't authored in T1's root layer (instance proxies / references).
  It also temporarily clears `instanceable=true` on K1 ancestors of
  right_hand_link in the OUTPUT layer only (the original K1 file is not
  modified).

How to run (you need `pxr`, which stock system Python usually lacks; use the
Isaac Sim Python instead):

    # Option A - conda env that has isaacsim (pxr is bundled with it):
    conda activate Isaaclab_Hover
    python tools\add_t1_paddle_to_k1.py ^
        --t1  C:\path\to\T1_with_paddle.usd ^
        --k1  C:\path\to\K1.usd ^
        --out C:\path\to\K1_22dof_pingpong.usd ^
        --t1-link /T1/right_hand_link ^
        --k1-link /K1/right_hand_link

    # Option B - Isaac Sim launcher (headless):
    isaacsim --no-window --exec tools\add_t1_paddle_to_k1.py

    # Option C - Isaac Lab launcher:
    isaaclab.bat -p tools\add_t1_paddle_to_k1.py

You can also just edit the CONFIG block below and run with no CLI args.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

# =========================== CONFIG (defaults) =============================
# Edit these to your real paths, or pass --t1 / --k1 / --out on the command line.

T1_USD_PATH        = r"C:\Users\kylel\TTRL-ICRA2026\legged_lab\assets\booster\T1_TT\T1_TT.usd"
K1_USD_PATH        = r"C:\Users\kylel\TTRL-ICRA2026\legged_lab\assets\booster\K1_TT\source\K1_22dof\K1_22dof.usd"
OUTPUT_USD_PATH    = r"C:\Users\kylel\TTRL-ICRA2026\legged_lab\assets\booster\K1_TT\K1_TT.usd"

# Absolute prim paths inside each stage. Adjust to match your actual hierarchy.
T1_RIGHT_HAND_LINK = "/T1_TT/right_hand_link"
K1_RIGHT_HAND_LINK = "/K1/right_hand_link"

# Children of right_hand_link to copy. Skip the link itself.
CHILDREN_TO_COPY   = ("marker_ball", "visuals", "collisions")

ADAPTER_NAME       = "paddle_adapter"

# Optional: extra absolute T1 prim paths to also copy into the output (sibling
# of paddle_adapter, under /AdapterAssets), in case your materials live outside
# right_hand_link (e.g. "/T1/Looks") and the visuals reference them by path.
# Leave as () if not needed.
EXTRA_SRC_PATHS    = ()  # e.g. ("/T1/Looks",)
EXTRA_DST_PARENT   = "/AdapterAssets"  # only used if EXTRA_SRC_PATHS is non-empty
# ===========================================================================


def _bootstrap_pxr_from_isaacsim() -> bool:
    """
    When Isaac Sim is installed as a pip package (isaacsim 4.x), pxr lives
    inside extscache/omni.usd.libs-*/  and its native DLLs are in a bin/
    subfolder that is NOT on PATH by default.  This function:
      1. Imports omni.kit_app  – bootstraps carb.dll and adds the omni
         package directory to os.add_dll_directory().
      2. Locates omni.usd.libs-* inside the isaacsim extscache.
      3. Adds its root (for the pxr Python package) and bin/ (for DLLs) to
         sys.path / os.add_dll_directory() respectively.
    Returns True if pxr becomes importable, False otherwise.
    """
    import glob
    try:
        import omni.kit_app  # noqa: F401  – side-effect: loads carb.dll
    except Exception:
        return False
    try:
        import isaacsim as _isaacsim
        isaacsim_path = os.path.dirname(os.path.abspath(_isaacsim.__file__))
    except Exception:
        return False
    usd_libs = glob.glob(
        os.path.join(isaacsim_path, "extscache", "omni.usd.libs-*")
    )
    if not usd_libs:
        return False
    usd_lib_path = usd_libs[0]
    usd_lib_bin  = os.path.join(usd_lib_path, "bin")
    if usd_lib_path not in sys.path:
        sys.path.insert(0, usd_lib_path)
    if hasattr(os, "add_dll_directory"):
        for d in (usd_lib_path, usd_lib_bin):
            if os.path.isdir(d):
                os.add_dll_directory(d)
    try:
        from pxr import Sdf as _Sdf  # noqa: F401
        return True
    except ImportError:
        return False


try:
    from pxr import Sdf, Usd, UsdGeom
except ImportError as _first_exc:
    if not _bootstrap_pxr_from_isaacsim():
        sys.stderr.write(
            "ERROR: 'pxr' (USD Python bindings) is not importable in this Python.\n"
            "Run the script with Isaac Sim's Python. Examples:\n"
            "    isaacsim --no-window --exec tools\\add_t1_paddle_to_k1.py\n"
            "    isaaclab.bat -p tools\\add_t1_paddle_to_k1.py\n"
            f"Underlying error: {_first_exc}\n"
        )
        sys.exit(1)
    from pxr import Sdf, Usd, UsdGeom


# --------------------------- tiny logging shims ----------------------------
def info(msg: str) -> None:  print(f"[paddle->K1]       {msg}")
def warn(msg: str) -> None:  print(f"[paddle->K1][WARN] {msg}")
def err(msg: str)  -> None:  print(f"[paddle->K1][ERROR]{msg}", file=sys.stderr)


# --------------------------- USD helpers -----------------------------------
def open_stage_or_die(path: str, label: str) -> Usd.Stage:
    if not os.path.isfile(path):
        err(f"{label} USD not found: {path}")
        sys.exit(2)
    stage = Usd.Stage.Open(path)
    if stage is None:
        err(f"failed to open {label} USD: {path}")
        sys.exit(3)
    return stage


def list_root_children(stage: Usd.Stage) -> str:
    return ", ".join(p.GetName() for p in stage.GetPseudoRoot().GetChildren()) or "(none)"


def build_output_with_k1_sublayer(k1_path: str, out_path: str) -> Usd.Stage:
    """
    Create the OUTPUT USD as a new root layer that sublayers K1's USD.
    K1's references/payloads/meshes stay external (no flattening of K1), and
    we author new opinions for paddle_adapter only in the output root layer.
    """
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path):
        info(f"output exists, replacing: {out_path}")
        existing = Sdf.Layer.FindOrOpen(out_path)
        if existing is not None:
            existing.Clear()
            out_layer = existing
        else:
            os.remove(out_path)
            out_layer = Sdf.Layer.CreateNew(out_path)
    else:
        out_layer = Sdf.Layer.CreateNew(out_path)

    rel_to_k1 = os.path.relpath(k1_path, out_dir).replace(os.sep, "/")
    out_layer.subLayerPaths.append(rel_to_k1)
    out_layer.Save()
    info(f"output sublayers K1 via: {rel_to_k1}")

    stage = Usd.Stage.Open(out_path)
    if stage is None:
        err(f"failed to open output stage: {out_path}")
        sys.exit(5)
    stage.SetEditTarget(stage.GetRootLayer())

    # Inherit K1's upAxis and defaultPrim so the output stage matches K1 exactly.
    # Without this, CreateNew defaults to Y-up which rotates Z-up robots 90 degrees.
    k1_stage = Usd.Stage.Open(k1_path)
    k1_up    = UsdGeom.GetStageUpAxis(k1_stage)
    k1_def   = k1_stage.GetDefaultPrim()
    UsdGeom.SetStageUpAxis(stage, k1_up)
    info(f"upAxis set to: {k1_up}")
    if k1_def and k1_def.IsValid():
        stage.SetDefaultPrim(stage.GetPrimAtPath(k1_def.GetPath()))
        info(f"defaultPrim set to: {k1_def.GetPath()}")
    stage.GetRootLayer().Save()

    return stage


def clear_instanceable_above(stage: Usd.Stage, prim_path: str) -> None:
    """
    Walk up from `prim_path`, clearing instanceable=true wherever set on the
    composed stage. Opinions are authored in the current edit target (output
    root layer), so the original K1 file is untouched. This lets us add
    `paddle_adapter` as a real child of right_hand_link even if K1 marks an
    ancestor instanceable.
    """
    p = stage.GetPrimAtPath(prim_path)
    while p and p.IsValid() and p.GetPath() != Sdf.Path.absoluteRootPath:
        if p.IsInstanceable():
            info(f"clearing instanceable=true (in output only) on {p.GetPath()}")
            p.SetInstanceable(False)
        p = p.GetParent()


def _try_copy(src_layer: Sdf.Layer, src: Sdf.Path,
              dst_layer: Sdf.Layer, dst: Sdf.Path) -> bool:
    try:
        return bool(Sdf.CopySpec(src_layer, src, dst_layer, dst))
    except Exception as e:  # noqa: BLE001 - we want any USD error noted, not fatal
        warn(f"Sdf.CopySpec raised on {src} -> {dst}: {e}")
        return False


def copy_specs(
    t1_path: str,
    src_parent: str,
    dst_layer: Sdf.Layer,
    dst_parent: str,
    names: Iterable[str],
) -> tuple[list[str], list[str]]:
    """
    For each `name`, copy `<src_parent>/<name>` from T1 into
    `<dst_parent>/<name>` in `dst_layer`.

    Strategy:
      1. Try Sdf.CopySpec from T1's ROOT layer (works if children are authored
         directly in T1's main USD file).
      2. For anything that failed or had no spec in T1's root layer, flatten
         T1 to an in-memory Sdf.Layer (which resolves instance proxies and
         references into concrete specs) and retry.

    Returns (copied_names, still_failed_names).
    """
    t1_stage = open_stage_or_die(t1_path, "T1")
    t1_root = t1_stage.GetRootLayer()

    src_parent_path = Sdf.Path(src_parent)
    dst_parent_path = Sdf.Path(dst_parent)

    if dst_layer.GetPrimAtPath(dst_parent_path) is None:
        err(f"destination parent has no spec in output layer: {dst_parent_path}")
        return [], list(names)

    copied: list[str] = []
    needs_flat: list[str] = []

    # ---- attempt 1: directly from T1 root layer ----
    for name in names:
        src = src_parent_path.AppendChild(name)
        dst = dst_parent_path.AppendChild(name)
        if t1_root.GetPrimAtPath(src) is None:
            info(f"[direct] no spec in T1 root layer for {src}; will flatten")
            needs_flat.append(name)
            continue
        if _try_copy(t1_root, src, dst_layer, dst):
            info(f"[direct] copied {src} -> {dst}")
            copied.append(name)
        else:
            warn(f"[direct] CopySpec failed for {src}; will flatten")
            needs_flat.append(name)

    if not needs_flat:
        return copied, []

    # ---- attempt 2: flatten T1 (resolves instance proxies + references) ----
    info(f"flattening T1 to copy: {needs_flat}")
    t1_flat = t1_stage.Flatten(addSourceFileComment=False)
    if t1_flat is None:
        err("failed to flatten T1 stage")
        return copied, needs_flat

    still_failed: list[str] = []
    for name in needs_flat:
        src = src_parent_path.AppendChild(name)
        dst = dst_parent_path.AppendChild(name)
        if t1_flat.GetPrimAtPath(src) is None:
            warn(f"[flat] not present even after flattening: {src}")
            still_failed.append(name)
            continue
        if _try_copy(t1_flat, src, dst_layer, dst):
            info(f"[flat] copied {src} -> {dst}")
            copied.append(name)
        else:
            warn(f"[flat] CopySpec failed: {src} -> {dst}")
            still_failed.append(name)

    # When Stage.Flatten() resolves instances it writes prototype content as
    # top-level "Flattened_Prototype_N" prims in the flat layer, and the copied
    # instance specs contain internal references pointing to those paths.
    # We must copy those prototype prims into the output layer so the references
    # resolve; otherwise USD reports "Unresolved reference prim path".
    pseudo_root = t1_flat.pseudoRoot
    if pseudo_root:
        for proto_spec in pseudo_root.nameChildren:
            if proto_spec.name.startswith("Flattened_Prototype"):
                proto_path = Sdf.Path(f"/{proto_spec.name}")
                if _try_copy(t1_flat, proto_path, dst_layer, proto_path):
                    info(f"[flat] copied prototype {proto_path} into output layer")
                else:
                    warn(f"[flat] failed to copy prototype {proto_path}")

    return copied, still_failed


# --------------------------- main flow -------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t1",  default=T1_USD_PATH,        help="T1 source USD")
    ap.add_argument("--k1",  default=K1_USD_PATH,        help="K1 destination USD")
    ap.add_argument("--out", default=OUTPUT_USD_PATH,    help="Output USD path (will be (over)written)")
    ap.add_argument("--t1-link", default=T1_RIGHT_HAND_LINK, help="T1 right_hand_link prim path")
    ap.add_argument("--k1-link", default=K1_RIGHT_HAND_LINK, help="K1 right_hand_link prim path")
    ap.add_argument("--adapter-name", default=ADAPTER_NAME, help="Name of the adapter Xform child")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    info(f"T1 USD     : {args.t1}")
    info(f"K1 USD     : {args.k1}")
    info(f"OUT USD    : {args.out}")
    info(f"T1 link    : {args.t1_link}")
    info(f"K1 link    : {args.k1_link}")
    info(f"adapter    : {args.adapter_name}")

    # ---- validate sources -------------------------------------------------
    k1_probe = open_stage_or_die(args.k1, "K1")
    k1_rh = k1_probe.GetPrimAtPath(args.k1_link)
    if not k1_rh or not k1_rh.IsValid():
        err(f"K1 right_hand_link not found at: {args.k1_link}")
        info(f"K1 root prims: {list_root_children(k1_probe)}")
        return 6
    info(f"K1 right_hand_link present (is_instance_proxy={k1_rh.IsInstanceProxy()})")

    t1_probe = open_stage_or_die(args.t1, "T1")
    t1_rh = t1_probe.GetPrimAtPath(args.t1_link)
    if not t1_rh or not t1_rh.IsValid():
        err(f"T1 right_hand_link not found at: {args.t1_link}")
        info(f"T1 root prims: {list_root_children(t1_probe)}")
        return 7
    info(f"T1 right_hand_link present (is_instance_proxy={t1_rh.IsInstanceProxy()})")
    info("T1 children of right_hand_link: " +
         ", ".join(c.GetName() for c in t1_rh.GetChildren()))

    del k1_probe, t1_probe

    # ---- build output stage that sublayers K1 -----------------------------
    out_stage = build_output_with_k1_sublayer(args.k1, args.out)

    # ---- if K1 hierarchy is instanceable, undo it in the OUTPUT only ------
    clear_instanceable_above(out_stage, args.k1_link)

    # ---- ensure paddle_adapter Xform exists -------------------------------
    adapter_path = f"{args.k1_link}/{args.adapter_name}"
    existing_adapter = out_stage.GetPrimAtPath(adapter_path)
    if existing_adapter and existing_adapter.IsValid():
        info(f"adapter already exists, reusing: {adapter_path}")
    else:
        UsdGeom.Xform.Define(out_stage, adapter_path)
        info(f"created adapter Xform: {adapter_path}")

    # ---- copy children specs from T1 into output --------------------------
    out_root = out_stage.GetRootLayer()
    copied, failed = copy_specs(
        t1_path=args.t1,
        src_parent=args.t1_link,
        dst_layer=out_root,
        dst_parent=adapter_path,
        names=CHILDREN_TO_COPY,
    )

    # ---- optional: copy extra absolute prim paths (e.g. /T1/Looks) --------
    extras_copied: list[str] = []
    extras_failed: list[str] = []
    if EXTRA_SRC_PATHS:
        if out_stage.GetPrimAtPath(EXTRA_DST_PARENT) is None:
            UsdGeom.Xform.Define(out_stage, EXTRA_DST_PARENT)
            info(f"created extras parent Xform: {EXTRA_DST_PARENT}")
        for src_abs in EXTRA_SRC_PATHS:
            name = Sdf.Path(src_abs).name
            ec, ef = copy_specs(
                t1_path=args.t1,
                src_parent=str(Sdf.Path(src_abs).GetParentPath()),
                dst_layer=out_root,
                dst_parent=EXTRA_DST_PARENT,
                names=(name,),
            )
            extras_copied += [f"{EXTRA_DST_PARENT}/{n}" for n in ec]
            extras_failed += [f"{EXTRA_DST_PARENT}/{n}" for n in ef]

    # ---- post-copy verification ------------------------------------------
    bad = []
    for name in copied:
        p = out_stage.GetPrimAtPath(f"{adapter_path}/{name}")
        if not p or not p.IsValid():
            bad.append(name)
    if bad:
        warn(f"post-copy: these adapter children did not appear in the stage: {bad}")

    # ---- save & summarize -------------------------------------------------
    out_root.Save()
    info("---- SUMMARY ----")
    info(f"output      : {args.out}")
    info(f"adapter     : {adapter_path}")
    info(f"copied      : {copied}")
    if failed:
        warn(f"NOT copied  : {failed}")
    if EXTRA_SRC_PATHS:
        info(f"extras OK   : {extras_copied}")
        if extras_failed:
            warn(f"extras FAIL : {extras_failed}")
    info("done.")
    return 0 if not failed else 9


if __name__ == "__main__":
    sys.exit(main())
