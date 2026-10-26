#!/usr/bin/env python3
"""
Thin CLI wrapper for eval_utils_sep.run_eval_from_eval_yaml

Writes one pickle per dataset:
  <model_id>__<dataset_id>__<ckpt_tag>.pkl
"""

import argparse
from eval_utils_sep import run_eval_from_eval_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    outs = run_eval_from_eval_yaml(args.eval_config, device=args.device)
    for p in outs:
        print(f"[ok] wrote {p}")


if __name__ == "__main__":
    main()
