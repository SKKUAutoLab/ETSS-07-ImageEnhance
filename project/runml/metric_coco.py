#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Measures COCO metrics for a given model and dataset."""

import argparse
import json
import logging
from datetime import datetime

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import mon

mon.disable_print()


# ----- COCO -----
def measure_metric(input_json: mon.Path, target_json: mon.Path):
    assert  input_json and  mon.Path(input_json).is_json_file()
    assert target_json and mon.Path(target_json).is_json_file()

    coco_gt   = COCO(str(target_json))
    coco_dt   = coco_gt.loadRes(str(input_json))
    imgIds    = sorted(coco_gt.getImgIds())

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = imgIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results = {
        "AP"    : coco_eval.stats[0],
        "AP50"  : coco_eval.stats[1],
        "AP75"  : coco_eval.stats[2],
        "APs"   : coco_eval.stats[3],
        "APm"   : coco_eval.stats[4],
        "APl"   : coco_eval.stats[5],
        "AR@1"  : coco_eval.stats[6],
        "AR@10" : coco_eval.stats[7],
        "AR@100": coco_eval.stats[8],
        "ARs"   : coco_eval.stats[9],
        "ARm"   : coco_eval.stats[10],
        "ARl"   : coco_eval.stats[11],
    }
    return results


def convert_label_to_coco(
    input_dir    : mon.Path,
    label_dir    : mon.Path,
    input_json   : mon.Path,
    remap_classes: mon.Path,
    bbox_format  : str,
) -> mon.Path:
    assert input_dir and input_dir.is_dir()
    assert label_dir and label_dir.is_dir()

    # Parse files
    if not (input_json and input_json.is_json_file(exist=False)):
        input_json = label_dir.parent / f"{label_dir.stem}.json"
    input_json.parent.mkdir(parents=True, exist_ok=True)

    if remap_classes and remap_classes.is_file():
        remap_classes = mon.load_config(config=remap_classes)["remap_classes"]
    else:
        remap_classes = None

    if bbox_format != "coco":
        code = mon.ShapeCode.from_value(value=f"{bbox_format}_to_coco")
    else:
        code = None
    
    # COCO JSON Format
    annotations = []
    image_files = list(input_dir.rglob("*"))
    image_files = sorted([f for f in image_files if f.is_image_file()])
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
                sequence    = enumerate(image_files),
                total       = len(image_files),
                description = f"[bright_yellow] Converting"
        ):
            # Append image
            h, w, c  = mon.read_image_shape(image_file)
            image_id = i

            # Append annotations
            label_file = label_dir / f"{image_file.stem}.txt"
            if not label_file.is_txt_file():
                continue
            
            with open(label_file, "r") as f:
                lines = f.readlines()
            lines   = [l.strip().split(" ") for l in lines]
            lines   = [l for l in lines if len(l) >= 5]
            if len(lines) == 0:
                continue
            classes = [int(l[0]) for l in lines]
            bboxes  = np.array([list(map(float, l[1:5])) for l in lines])
            scores  = [float(l[5]) for l in lines]
            if code:
                bboxes = mon.convert_bbox(bbox=bboxes, code=code, height=h, width=w)
            assert len(lines) == len(classes)
            assert len(lines) == len(bboxes)

            for c, b, s in zip(classes, bboxes, scores):
                if remap_classes:
                    if c in remap_classes:
                        c = int(remap_classes[c])
                    else:
                        continue
                annotations.append({
                    "image_id"   : image_id,
                    "category_id": c,
                    "bbox"       : [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
                    "score"      : s,
                })

    # Write to JSON file
    with open(str(input_json), "w") as f:
        json.dump(annotations, f, indent=None)

    return input_json


# ----- Main -----
def main(
    input_dir    : mon.Path,
    label_dir    : mon.Path,
    input_json   : mon.Path,
    target_json  : mon.Path,
    result_file  : mon.Path,
    remap_classes: mon.Path,
    arch         : str,
    model        : str,
    data         : str,
    device       : int | list[int] | str,
    bbox_format  : str,
    save_txt     : bool,
    exist_ok     : bool,
    verbose      : bool,
):
    if not verbose:
        logger = logging.getLogger()
        logger.disabled = True
    mon.console.rule(f"[bold red] {model}")
    mon.console.log(f"[bold green]Model: {model}")
    mon.console.log(f"[bold red]Data : {data}")

    input_dir     = mon.Path(input_dir)     if input_dir     else None
    label_dir     = mon.Path(label_dir)     if label_dir     else None
    input_json    = mon.Path(input_json)    if input_json    else None
    target_json   = mon.Path(target_json)   if target_json   else None
    result_file   = mon.Path(result_file)   if result_file   else None
    remap_classes = mon.Path(remap_classes) if remap_classes else None

    assert target_json and mon.Path(target_json).is_json_file()

    if not exist_ok and input_json and input_json.is_json_file():
        mon.delete_files(input_json)
    if input_json and input_json.is_json_file():
        results = measure_metric(input_json=input_json, target_json=target_json)
    elif input_dir and input_dir.is_dir() and label_dir and label_dir.is_dir():
        input_json = convert_label_to_coco(
            input_dir     = input_dir,
            label_dir     = label_dir,
            input_json    = input_json,
            remap_classes = remap_classes,
            bbox_format   = bbox_format,
        )
        results = measure_metric(input_json=input_json, target_json=target_json)
    else:
        raise RuntimeError(
            f"Either ``input_json`` or [``input_dir`` and ``label_dir``] must be provided, got:\n"
            f"input_json: {input_json}\n"
            f"input_dir : {input_dir}\n"
            f"label_dir : {label_dir}"
        )

    # Show results
    message = ""
    # Headers
    for m, v in results.items():
        if v:
            message += f"{f'{m}':<10}\t"
    message += "\n"
    # Values
    for i, (m, v) in enumerate(results.items()):
        if v:
            if i == len(results) - 1:
                message += f"{v:.10f}\n"
            else:
                message += f"{v:.10f}\t"
    print(f"{message}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="metric_iqa")
    parser.add_argument("--input-dir",     type=str, help="Input image directory.")
    parser.add_argument("--label-dir",     type=str, help="Input label directory.")
    parser.add_argument("--input-json",    type=str, help="Input JSON file.")
    parser.add_argument("--target-json",   type=str, help="Ground-truth JSON file.")
    parser.add_argument("--result-file",   type=str, help="Result file.")
    parser.add_argument("--remap-classes", type=str, help="Classes re-map definition file.")
    parser.add_argument("--arch",          type=str, help="Model's architecture.")
    parser.add_argument("--model",         type=str, help="Model's fullname.")
    parser.add_argument("--data",          type=str, help="Source data name.")
    parser.add_argument("--device",        type=str, help="Running devices.")
    parser.add_argument("--bbox-format",   choices=["coco", "voc", "yolo"], default="coco")
    parser.add_argument("--save-txt",      action="store_true")
    parser.add_argument("--exist-ok",      action="store_true")
    parser.add_argument("--verbose",       action="store_true")
    args = parser.parse_args()
    main(**vars(args))
