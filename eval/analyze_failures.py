# -*- coding: utf-8 -*-
"""
失败画像分析：读取 rollout.py 落盘的 eval/results/<prefix>_results.csv，
统计失败 case 按 spec 维度（gain / ibias_max / phm / ugbw）的失守分布。

口径与 autockt/envs/ngspice_vanilla_opamp.py 的 reward() 完全一致：
  rel = (final - goal) / (final + goal)；ibias_max 取反；
  rel < 0 即该维度失守（未达标）。reward = sum(rel<0 的 rel)，>-0.02 即达标(done)。
CSV 中 goal_specs / final_specs 按字典序排列：gain, ibias_max, phm, ugbw
（ibias_max 为上限约束，越界方向与其他三个相反）。
"""
import csv
import sys
import os

DIMS = ["gain", "ibias_max", "phm", "ugbw"]


def main(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    fails = [r for r in rows if r["success"] == "0"]
    succs = [r for r in rows if r["success"] == "1"]
    print("=" * 60)
    print("文件: %s" % csv_path)
    print("总 case: %d | 成功: %d | 失败: %d | 成功率: %.1f%%" %
          (total, len(succs), len(fails), 100.0 * len(succs) / total))
    print("=" * 60)

    if not fails:
        print("无失败 case。")
        return

    # 1) 失守维度计数（单 case 可多维度同时失守）
    dim_count = {d: 0 for d in DIMS}
    # 2) 失守 case 的归一化缺口总和（越负越差）与均值
    dim_gap_sum = {d: 0.0 for d in DIMS}
    # 3) 失守维度数分布（1维/2维/...）
    ndim_hist = {}
    # 4) 逐 case 明细
    details = []

    for r in fails:
        goal = [float(x) for x in r["goal_specs"].split(";")]
        fin = [float(x) for x in r["final_specs"].split(";")]
        violated = []
        for i, d in enumerate(DIMS):
            rel = (fin[i] - goal[i]) / (fin[i] + goal[i])
            if d == "ibias_max":
                rel = -rel
            if rel < 0:
                dim_count[d] += 1
                dim_gap_sum[d] += rel
                violated.append((d, rel))
        ndim_hist[len(violated)] = ndim_hist.get(len(violated), 0) + 1
        details.append((int(r["case_id"]), violated, int(r["steps"]), float(r["reward_total"])))

    nfail = len(fails)
    print("\n【失守维度分布】（占失败 case 比例，单 case 可多维度失守）")
    for d in DIMS:
        print("  %-9s 失守 %3d/%3d  (%5.1f%%)   平均归一化缺口: %.4f" %
              (d, dim_count[d], nfail, 100.0 * dim_count[d] / nfail,
               dim_gap_sum[d] / dim_count[d] if dim_count[d] else 0.0))

    print("\n【同时失守维度数分布】")
    for k in sorted(ndim_hist):
        print("  %d 维失守: %d case (%5.1f%%)" % (k, ndim_hist[k], 100.0 * ndim_hist[k] / nfail))

    print("\n【失败 case 明细】(case_id | steps | reward_total | 失守维度:缺口)")
    for cid, violated, steps, rt in details:
        vs = " ".join("%s:%.3f" % (d, g) for d, g in violated)
        print("  %4d | steps=%2d | reward=%7.2f | %s" % (cid, steps, rt, vs))

    # 5) 成功 case 平均达标步数
    if succs:
        steps_succ = [int(r["steps"]) for r in succs]
        print("\n【成功 case 达标步数】mean=%.1f min=%d max=%d" %
              (sum(steps_succ) / len(steps_succ), min(steps_succ), max(steps_succ)))
    # 6) 失败 case 步数分布（判断是否跑满 traj_len = 震荡游走 or 提前停）
    steps_fail = [int(r["steps"]) for r in fails]
    full = sum(1 for s in steps_fail if s >= 60)
    print("【失败 case 步数】mean=%.1f | 跑满60步: %d/%d (%.0f%%)" %
          (sum(steps_fail) / nfail, full, nfail, 100.0 * full / nfail))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python analyze_failures.py <results_csv>")
        sys.exit(1)
    main(os.path.abspath(sys.argv[1]))
