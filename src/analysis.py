import math
import json
import re
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "exp"


def _spearman(x, y):
    n = len(x)

    def _rank(a):
        order = sorted(range(n), key=lambda i: a[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and a[order[j]] == a[order[i]]:
                j += 1
            avg = (i + j - 1) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg
            i = j
        return ranks

    rx, ry = _rank(x), _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(
        sum((rx[i] - mx) ** 2 for i in range(n))
        * sum((ry[i] - my) ** 2 for i in range(n))
    )
    return num / (den + 1e-10)


def _ndcg_at_k(true_scores, pred_scores, k=10):
    n = len(true_scores)
    k = min(k, n)
    order = sorted(range(n), key=lambda i: pred_scores[i], reverse=True)
    dcg = sum(
        (2.0 ** true_scores[order[i]] - 1.0) / math.log2(i + 2)
        for i in range(k)
    )
    ideal = sorted(range(n), key=lambda i: true_scores[i], reverse=True)
    idcg = sum(
        (2.0 ** true_scores[ideal[i]] - 1.0) / math.log2(i + 2)
        for i in range(k)
    )
    return dcg / idcg if idcg > 0.0 else 0.0


def _aggregate_model(args, model_name: str):
    import csv

    version = args.version
    genre = args.genre

    data_dir = Path(getattr(args, "data_dir", None) or
                    Path(__file__).resolve().parent.parent / "data")

    model_dir = Path(__file__).resolve().parent.parent / "reports" / "exp" / model_name
    matched = list(model_dir.glob(f"{genre}_piaa_results*.json"))
    if not matched:
        matched = list(model_dir.glob(f"{genre}_results*.json"))
    if not matched:
        matched = list(model_dir.glob(f"{genre}_giaa_results*.json"))
    if not matched:
        print(
            f"Error: {model_name} results not found in {model_dir} "
            f"(pattern: {genre}_piaa_results*.json / {genre}_giaa_results*.json)",
            file=sys.stderr,
        )
        sys.exit(1)
    model_json = matched[0]

    with open(model_json) as f:
        llm_data = json.load(f)

    per_user_pred = {}
    per_user_pred_stem = {}
    fallback_pred = {}
    fallback_pred_stem = {}
    n_user_preds = 0
    for entry in llm_data["per_sample"]:
        sf = entry["sample_file"]
        stem = Path(sf).stem
        ratings = entry.get("ratings")
        if ratings:
            for r in ratings:
                uid = str(r["user_id"])
                p = float(r["pred_score"])
                per_user_pred[(uid, sf)] = p
                per_user_pred_stem[(uid, stem)] = p
                n_user_preds += 1
        dist = entry.get("pred_dist")
        if dist:
            e = sum(i * p for i, p in enumerate(dist))
            fallback_pred[sf] = e
            fallback_pred_stem[stem] = e

    piaa_mode = n_user_preds > 0
    if piaa_mode:
        print(
            f"Loaded {n_user_preds} per-user {model_name} predictions "
            f"over {len(llm_data['per_sample'])} samples  (genre='{genre}')"
        )
    else:
        if not fallback_pred:
            print(
                f"Error: {model_json.name} has neither per-user ratings nor pred_dist",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"[WARN] No per-user ratings in {model_json.name}; "
            f"falling back to zero-shot (pred_dist expected value). "
            f"Note: this is NOT a fair PIAA comparison — "
            f"generate {genre}_piaa_results.json for per-user predictions.",
            file=sys.stderr,
        )
        print(
            f"Loaded {len(fallback_pred)} {model_name} pred_dist predictions "
            f"(zero-shot fallback)  (genre='{genre}')"
        )

    def _lookup_pred(uid, sf):
        if piaa_mode:
            if (uid, sf) in per_user_pred:
                return per_user_pred[(uid, sf)]
            stem = Path(sf).stem
            return per_user_pred_stem.get((uid, stem))
        if sf in fallback_pred:
            return fallback_pred[sf]
        return fallback_pred_stem.get(Path(sf).stem)

    ratings_path = data_dir / "maked" / "ratings.csv"
    if not ratings_path.exists():
        print(f"Error: ratings.csv not found: {ratings_path}", file=sys.stderr)
        sys.exit(1)

    gt_scores = {}
    with open(ratings_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["genre"] != genre:
                continue
            try:
                gt_scores[(row["user_id"], row["sample_file"])] = float(row["Aesthetic"])
            except (ValueError, KeyError):
                pass

    print(f"Loaded {len(gt_scores)} ground-truth ratings  (genre='{genre}')")

    split_dir = data_dir / "split"
    fold_dirs = sorted(split_dir.glob(f"{version}_fold*"))
    if not fold_dirs:
        print(
            f"Error: No fold directories for version '{version}' in {split_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.folds is not None:
        fold_set = set(args.folds)
        fold_dirs = [
            d for d in fold_dirs
            if int(d.name.split("fold")[-1]) in fold_set
        ]
        if not fold_dirs:
            print(f"Error: No matching fold directories for folds {args.folds}", file=sys.stderr)
            sys.exit(1)

    all_user_mae   = {}
    all_user_ndcg  = {}
    all_user_srocc = {}
    all_user_ccc   = {}

    skipped_missing_pred = 0

    for fold_dir in fold_dirs:
        test_file = fold_dir / genre / "test_PIAA.txt"
        if not test_file.exists():
            print(f"  Warning: {test_file} not found, skipping")
            continue

        user_test: dict = {}
        with open(test_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    user_test.setdefault(parts[0], []).append(parts[1])

        n_pairs = sum(len(v) for v in user_test.values())
        print(f"  [{fold_dir.name}] {len(user_test)} users, {n_pairs} test pairs")

        for uid, sample_files in user_test.items():
            preds, gts = [], []
            for sf in sample_files:
                p = _lookup_pred(uid, sf)
                if p is None:
                    skipped_missing_pred += 1
                    continue
                key = (uid, sf)
                if key not in gt_scores:
                    continue
                preds.append(p)
                gts.append(gt_scores[key])

            if len(preds) < 2:
                continue

            n = len(preds)
            srocc = _spearman(preds, gts)
            ndcg  = _ndcg_at_k(gts, preds, k=10)
            mae   = sum(abs(preds[i] / 6.0 - gts[i] / 6.0) for i in range(n)) / n

            mu_p  = sum(preds) / n
            mu_t  = sum(gts)   / n
            cov   = sum((preds[i] - mu_p) * (gts[i] - mu_t) for i in range(n)) / n
            var_p = sum((preds[i] - mu_p) ** 2 for i in range(n)) / n
            var_t = sum((gts[i]   - mu_t) ** 2 for i in range(n)) / n
            ccc   = float(2 * cov / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8))

            all_user_mae.setdefault(uid,   []).append(mae)
            all_user_ndcg.setdefault(uid,  []).append(ndcg)
            all_user_srocc.setdefault(uid, []).append(srocc)
            all_user_ccc.setdefault(uid,   []).append(ccc)

    if not all_user_mae:
        print("Error: No user metrics computed.", file=sys.stderr)
        sys.exit(1)

    user_avg_mae   = [sum(v) / len(v) for v in all_user_mae.values()]
    user_avg_ndcg  = [sum(v) / len(v) for v in all_user_ndcg.values()]
    user_avg_srocc = [sum(v) / len(v) for v in all_user_srocc.values()]
    user_avg_ccc   = [sum(v) / len(v) for v in all_user_ccc.values()]

    n_users = len(user_avg_mae)

    avg_mae   = sum(user_avg_mae)   / n_users
    avg_ndcg  = sum(user_avg_ndcg)  / n_users
    avg_srocc = sum(user_avg_srocc) / n_users
    avg_ccc   = sum(user_avg_ccc)   / n_users

    std_mae   = math.sqrt(sum((x - avg_mae)   ** 2 for x in user_avg_mae)   / n_users)
    std_ndcg  = math.sqrt(sum((x - avg_ndcg)  ** 2 for x in user_avg_ndcg)  / n_users)
    std_srocc = math.sqrt(sum((x - avg_srocc) ** 2 for x in user_avg_srocc) / n_users)
    std_ccc   = math.sqrt(sum((x - avg_ccc)   ** 2 for x in user_avg_ccc)   / n_users)

    mode_tag = "PIAA" if piaa_mode else "Zero-Shot"
    print(f"\n=== {model_name.capitalize()} {mode_tag} Results ({version}, {genre}) ===")
    print(f"  Source:          {model_json.name}")
    print(f"  Folds:           {len(fold_dirs)}")
    print(f"  Total users:     {n_users}")
    if skipped_missing_pred:
        print(f"  Pairs skipped (no LLM prediction): {skipped_missing_pred}")
    print(f"  Average MAE:     {avg_mae:.6f} (std: {std_mae:.6f})")
    print(f"  Average NDCG@10: {avg_ndcg:.6f} (std: {std_ndcg:.6f})")
    print(f"  Average SROCC:   {avg_srocc:.6f} (std: {std_srocc:.6f})")
    print(f"  Average CCC:     {avg_ccc:.6f} (std: {std_ccc:.6f})")


def _aggregate_model_giaa(args, model_name: str):
    import csv

    version = args.version
    genre = args.genre
    data_dir = Path(getattr(args, "data_dir", None) or
                    Path(__file__).resolve().parent.parent / "data")

    model_dir = Path(__file__).resolve().parent.parent / "reports" / "exp" / model_name
    matched = list(model_dir.glob(f"{genre}_giaa_results*.json"))
    if not matched:
        matched = list(model_dir.glob(f"{genre}_results*.json"))
    if not matched:
        print(
            f"Error: {model_name} GIAA results not found in {model_dir} "
            f"(pattern: {genre}_giaa_results*.json)",
            file=sys.stderr,
        )
        sys.exit(1)
    model_json = matched[0]

    with open(model_json) as f:
        llm_data = json.load(f)

    pred_score = {}
    pred_score_stem = {}
    pred_dist = {}
    pred_dist_stem = {}
    for entry in llm_data["per_sample"]:
        dist = entry["pred_dist"]
        e = sum(i * p for i, p in enumerate(dist))
        sf = entry["sample_file"]
        pred_score[sf] = e
        pred_score_stem[Path(sf).stem] = e
        pred_dist[sf] = dist
        pred_dist_stem[Path(sf).stem] = dist

    print(f"Loaded {len(pred_score)} {model_name} predictions  (genre='{genre}')")

    NUM_BINS = 7
    ratings_path = data_dir / "maked" / "ratings.csv"
    img_sum: dict = {}
    img_cnt: dict = {}
    img_hist: dict = {}
    with open(ratings_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["genre"] != genre:
                continue
            try:
                sf = row["sample_file"]
                score = int(float(row["Aesthetic"]))
                img_sum[sf] = img_sum.get(sf, 0.0) + float(row["Aesthetic"])
                img_cnt[sf] = img_cnt.get(sf, 0) + 1
                hist = img_hist.setdefault(sf, [0] * NUM_BINS)
                if 0 <= score < NUM_BINS:
                    hist[score] += 1
            except (ValueError, KeyError):
                pass
    img_mean_gt = {sf: img_sum[sf] / img_cnt[sf] for sf in img_sum}
    img_hist_norm = {sf: [c / img_cnt[sf] for c in hist] for sf, hist in img_hist.items()}
    print(f"Loaded mean GT for {len(img_mean_gt)} images  (genre='{genre}')")

    split_dir = data_dir / "split"
    fold_dirs = sorted(split_dir.glob(f"{version}_fold*"))
    if not fold_dirs:
        print(f"Error: No fold directories for version '{version}' in {split_dir}", file=sys.stderr)
        sys.exit(1)
    if args.folds is not None:
        fold_set = set(args.folds)
        fold_dirs = [d for d in fold_dirs if int(d.name.split("fold")[-1]) in fold_set]

    def _emd(p, q):
        cp, cq = 0.0, 0.0
        acc = 0.0
        for a, b in zip(p, q):
            cp += a;  cq += b
            acc += (cp - cq) ** 2
        return acc ** 0.5

    fold_srocc, fold_mae, fold_ccc, fold_emd = [], [], [], []

    for fold_dir in fold_dirs:
        test_file = fold_dir / genre / "test_images_GIAA.txt"
        if not test_file.exists():
            print(f"  Warning: {test_file} not found, skipping")
            continue

        with open(test_file) as f:
            test_images = [l.strip() for l in f if l.strip()]

        preds, gts, dists, gt_hists = [], [], [], []
        for sf in test_images:
            stem = Path(sf).stem
            p = pred_score.get(sf) or pred_score_stem.get(stem)
            d = pred_dist.get(sf) or pred_dist_stem.get(stem)
            if p is None:
                continue
            gt_key = sf if sf in img_mean_gt else next(
                (k for k in img_mean_gt if Path(k).stem == stem), None)
            if gt_key is None:
                continue
            preds.append(p)
            gts.append(img_mean_gt[gt_key])
            if d is not None and gt_key in img_hist_norm:
                dists.append(d)
                gt_hists.append(img_hist_norm[gt_key])

        if len(preds) < 2:
            print(f"  [{fold_dir.name}] Too few matched images ({len(preds)}), skipping")
            continue

        n = len(preds)
        srocc = _spearman(preds, gts)
        mae = sum(abs(preds[i] / 6.0 - gts[i] / 6.0) for i in range(n)) / n
        mu_p = sum(preds) / n;  mu_t = sum(gts) / n
        cov = sum((preds[i] - mu_p) * (gts[i] - mu_t) for i in range(n)) / n
        var_p = sum((preds[i] - mu_p) ** 2 for i in range(n)) / n
        var_t = sum((gts[i] - mu_t) ** 2 for i in range(n)) / n
        ccc = float(2 * cov / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8))
        emd = sum(_emd(dists[i], gt_hists[i]) for i in range(len(dists))) / len(dists) if dists else float("nan")

        fold_srocc.append(srocc);  fold_mae.append(mae)
        fold_ccc.append(ccc);      fold_emd.append(emd)
        print(f"  [{fold_dir.name}] n={n}  EMD={emd:.4f}  SROCC={srocc:.4f}  MAE={mae:.6f}  CCC={ccc:.4f}")

    if not fold_srocc:
        print("Error: No fold metrics computed.", file=sys.stderr)
        sys.exit(1)

    def _stats(vals):
        avg = sum(vals) / len(vals)
        std = math.sqrt(sum((x - avg) ** 2 for x in vals) / len(vals))
        return avg, std

    avg_emd,   std_emd   = _stats(fold_emd)
    avg_srocc, std_srocc = _stats(fold_srocc)
    avg_mae,   std_mae   = _stats(fold_mae)
    avg_ccc,   std_ccc   = _stats(fold_ccc)

    print(f"\n=== {model_name.capitalize()} GIAA Results ({version}, {genre}) ===")
    print(f"  Folds:           {len(fold_srocc)}")
    print(f"  Average EMD:     {avg_emd:.6f} (std: {std_emd:.6f})")
    print(f"  Average SROCC:   {avg_srocc:.6f} (std: {std_srocc:.6f})")
    print(f"  Average MAE:     {avg_mae:.6f} (std: {std_mae:.6f})")
    print(f"  Average CCC:     {avg_ccc:.6f} (std: {std_ccc:.6f})")


def _aggregate_giaa(args):
    version = args.version
    genre = args.genre
    pattern = args.pattern
    method = args.method
    reports_dir = Path(args.reports_dir)

    fold_dirs = sorted(reports_dir.glob(f"{version}_fold*"))
    if not fold_dirs:
        print(f"Error: No fold directories for version '{version}' in {reports_dir}", file=sys.stderr)
        sys.exit(1)
    if args.folds is not None:
        fold_set = set(args.folds)
        fold_dirs = [d for d in fold_dirs if int(d.name.split("fold")[-1]) in fold_set]

    m2 = re.match(r'^(\w+)2(\w+)$', genre)
    if m2:
        metric_key = m2.group(1)
    else:
        metric_key = genre

    fold_emd, fold_srocc, fold_mae, fold_ccc = [], [], [], []
    cd_emd: dict = {}
    cd_srocc: dict = {}
    cd_mae: dict = {}
    cd_ccc: dict = {}
    cd_source_head: dict = {}

    for fold_dir in fold_dirs:
        genre_dir = fold_dir / genre
        if not genre_dir.is_dir():
            print(f"Error: Genre directory not found: {genre_dir}", file=sys.stderr)
            sys.exit(1)

        if method and pattern:
            glob_pattern = f"*{method}*{pattern}*.json"
        elif method:
            glob_pattern = f"*{method}*.json"
        elif pattern:
            glob_pattern = f"*{pattern}*.json"
        else:
            glob_pattern = "*.json"

        matched_jsons = [p for p in genre_dir.glob(glob_pattern)
                         if json.loads(p.read_text()).get("mode") == "GIAA"]
        if len(matched_jsons) == 0:
            print(f"Error: No GIAA JSON matching '{glob_pattern}' in {genre_dir}", file=sys.stderr)
            sys.exit(1)
        if len(matched_jsons) > 1:
            print(f"Error: Multiple GIAA JSONs found in {genre_dir}: {[f.name for f in matched_jsons]}", file=sys.stderr)
            sys.exit(1)

        data = json.loads(matched_jsons[0].read_text())
        m = data.get("average_metrics", {}).get(metric_key, {})
        if not m:
            print(f"  Warning: No average_metrics for genre '{metric_key}' in {matched_jsons[0].name}, skipping")
            continue

        fold_emd.append(m["emd"]);  fold_srocc.append(m["srocc"])
        fold_mae.append(m["mae"]);  fold_ccc.append(m["ccc"])
        print(f"  Loaded: {matched_jsons[0].relative_to(reports_dir)}  "
              f"EMD={m['emd']:.4f}  SROCC={m['srocc']:.4f}  CCC={m['ccc']:.4f}")

        cross_domain = data.get("cross_domain_metrics", {})
        for target_genre, cd_data in cross_domain.items():
            avg = cd_data.get("average", {})
            if not avg:
                continue
            cd_emd.setdefault(target_genre, []).append(avg["emd"])
            cd_srocc.setdefault(target_genre, []).append(avg["srocc"])
            cd_mae.setdefault(target_genre, []).append(avg["mae"])
            cd_ccc.setdefault(target_genre, []).append(avg["ccc"])
            if "source_head" in cd_data:
                cd_source_head[target_genre] = cd_data["source_head"]

    if not fold_emd:
        print("Error: No fold metrics found.", file=sys.stderr)
        sys.exit(1)

    def _stats(vals):
        avg = sum(vals) / len(vals)
        std = math.sqrt(sum((x - avg) ** 2 for x in vals) / len(vals))
        return avg, std

    avg_emd,   std_emd   = _stats(fold_emd)
    avg_srocc, std_srocc = _stats(fold_srocc)
    avg_mae,   std_mae   = _stats(fold_mae)
    avg_ccc,   std_ccc   = _stats(fold_ccc)

    print(f"\n=== Aggregated GIAA Results ({version}, {genre}, pattern='{pattern}') ===")
    print(f"  Folds:           {len(fold_emd)}")
    print(f"  Average EMD:     {avg_emd:.6f} (std: {std_emd:.6f})")
    print(f"  Average SROCC:   {avg_srocc:.6f} (std: {std_srocc:.6f})")
    print(f"  Average MAE:     {avg_mae:.6f} (std: {std_mae:.6f})")
    print(f"  Average CCC:     {avg_ccc:.6f} (std: {std_ccc:.6f})")

    if cd_emd:
        print(f"\n  --- Cross-Domain (GIAA) ---")
        for target_genre in sorted(cd_emd.keys()):
            if not cd_emd[target_genre]:
                continue
            cavg_emd,   cstd_emd   = _stats(cd_emd[target_genre])
            cavg_srocc, cstd_srocc = _stats(cd_srocc[target_genre])
            cavg_mae,   cstd_mae   = _stats(cd_mae[target_genre])
            cavg_ccc,   cstd_ccc   = _stats(cd_ccc[target_genre])
            src = cd_source_head.get(target_genre, metric_key)
            print(f"  [{src} -> {target_genre}]")
            print(f"    Folds:           {len(cd_emd[target_genre])}")
            print(f"    Average EMD:     {cavg_emd:.6f} (std: {cstd_emd:.6f})")
            print(f"    Average SROCC:   {cavg_srocc:.6f} (std: {cstd_srocc:.6f})")
            print(f"    Average MAE:     {cavg_mae:.6f} (std: {cstd_mae:.6f})")
            print(f"    Average CCC:     {cavg_ccc:.6f} (std: {cstd_ccc:.6f})")


def aggregate(args):
    version = args.version
    genre = args.genre
    pattern = args.pattern

    giaa_mode = getattr(args, "giaa_mode", False)

    if pattern in ("claude", "gemini", "gpt"):
        if giaa_mode:
            _aggregate_model_giaa(args, pattern)
        else:
            _aggregate_model(args, pattern)
        return

    if giaa_mode:
        _aggregate_giaa(args)
        return
    method = args.method
    min_id = args.min_id
    max_id = args.max_id
    ids = set(args.ids) if args.ids is not None else None
    reports_dir = Path(args.reports_dir)

    fold_dirs = sorted(reports_dir.glob(f"{version}_fold*"))
    if not fold_dirs:
        print(
            f"Error: No fold directories found for version '{version}' in {reports_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.folds is not None:
        fold_set = set(args.folds)
        fold_dirs = [
            d for d in fold_dirs
            if int(d.name.split("fold")[-1]) in fold_set
        ]
        if not fold_dirs:
            print(
                f"Error: No matching fold directories for folds {args.folds}",
                file=sys.stderr,
            )
            sys.exit(1)

    m2 = re.match(r'^(\w+)2(\w+)$', genre)
    if m2:
        sub_genres = [m2.group(1)]
    else:
        sub_genres = genre.split("-")

    all_user_mae  = {sg: {} for sg in sub_genres}
    all_user_ndcg = {sg: {} for sg in sub_genres}
    all_user_srocc = {sg: {} for sg in sub_genres}
    all_user_ccc = {sg: {} for sg in sub_genres}

    cd_user_mae  = {}
    cd_user_srocc = {}
    cd_user_ndcg = {}
    cd_user_ccc = {}

    for fold_dir in fold_dirs:
        genre_dir = fold_dir / genre
        if not genre_dir.is_dir():
            print(f"Error: Genre directory not found: {genre_dir}", file=sys.stderr)
            sys.exit(1)

        if method and pattern:
            glob_pattern = f"*{method}*{pattern}*.json"
        elif method:
            glob_pattern = f"*{method}*.json"
        elif pattern:
            glob_pattern = f"*{pattern}*.json"
        else:
            glob_pattern = "*.json"
        matched_jsons = list(genre_dir.glob(glob_pattern))
        if min_id is not None or max_id is not None or ids is not None:
            def _extract_id(p):
                m = re.search(r'-(\d+)[_.]', p.name)
                return int(m.group(1)) if m else -1
            matched_jsons = [
                p for p in matched_jsons
                if (min_id is None or _extract_id(p) >= min_id)
                and (max_id is None or _extract_id(p) <= max_id)
                and (ids is None or _extract_id(p) in ids)
            ]
        if len(matched_jsons) == 0:
            print(f"Error: No JSON matching '{glob_pattern}' found in {genre_dir}", file=sys.stderr)
            sys.exit(1)
        if len(matched_jsons) > 1:
            print(
                f"Error: Multiple JSONs matching '{glob_pattern}' found in {genre_dir}: {[f.name for f in matched_jsons]}",
                file=sys.stderr,
            )
            sys.exit(1)

        json_path = matched_jsons[0]
        with open(json_path) as f:
            data = json.load(f)

        per_user = data.get("per_user_metrics", {})
        for user_id, metrics in per_user.items():
            for sg in sub_genres:
                genre_metrics = metrics.get(sg, {})
                mae  = genre_metrics.get("mae")
                ndcg = genre_metrics.get("ndcg@10")
                srocc = genre_metrics.get("srocc")
                ccc = genre_metrics.get("ccc")
                if mae is not None:
                    all_user_mae[sg].setdefault(user_id, []).append(mae)
                if ndcg is not None:
                    all_user_ndcg[sg].setdefault(user_id, []).append(ndcg)
                if srocc is not None:
                    all_user_srocc[sg].setdefault(user_id, []).append(srocc)
                if ccc is not None:
                    all_user_ccc[sg].setdefault(user_id, []).append(ccc)

        cross_domain = data.get("cross_domain_metrics", {})
        for target_genre, cd_data in cross_domain.items():
            if target_genre not in cd_user_mae:
                cd_user_mae[target_genre]  = {}
                cd_user_srocc[target_genre] = {}
                cd_user_ndcg[target_genre] = {}
                cd_user_ccc[target_genre] = {}
            per_user_cd = cd_data.get("per_user", {})
            for user_id, cd_metrics in per_user_cd.items():
                mae  = cd_metrics.get("mae")
                ndcg = cd_metrics.get("ndcg@10")
                srocc = cd_metrics.get("srocc")
                ccc = cd_metrics.get("ccc")
                if mae is not None:
                    cd_user_mae[target_genre].setdefault(user_id, []).append(mae)
                if ndcg is not None:
                    cd_user_ndcg[target_genre].setdefault(user_id, []).append(ndcg)
                if srocc is not None:
                    cd_user_srocc[target_genre].setdefault(user_id, []).append(srocc)
                if ccc is not None:
                    cd_user_ccc[target_genre].setdefault(user_id, []).append(ccc)

        print(
            f"  Loaded: {json_path.relative_to(reports_dir)} ({len(per_user)} users)"
        )

    if not any(all_user_mae[sg] for sg in sub_genres):
        print("Error: No user metrics found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Aggregated Results ({version}, {genre}, pattern='{pattern}') ===")
    print(f"  Folds:         {len(fold_dirs)}")

    for sg in sub_genres:
        if not all_user_mae[sg]:
            continue

        user_avg_mae  = [sum(vals) / len(vals) for vals in all_user_mae[sg].values()]
        user_avg_ndcg = [sum(vals) / len(vals) for vals in all_user_ndcg[sg].values()]
        user_avg_srocc = [sum(vals) / len(vals) for vals in all_user_srocc[sg].values()]
        user_avg_ccc  = [sum(vals) / len(vals) for vals in all_user_ccc[sg].values()]

        avg_mae   = sum(user_avg_mae)   / len(user_avg_mae)
        avg_ndcg  = sum(user_avg_ndcg)  / len(user_avg_ndcg)
        avg_srocc = sum(user_avg_srocc) / len(user_avg_srocc)
        avg_ccc   = sum(user_avg_ccc)   / len(user_avg_ccc) if user_avg_ccc else None

        std_mae  = math.sqrt(sum((x - avg_mae)   ** 2 for x in user_avg_mae)   / len(user_avg_mae))
        std_ndcg = math.sqrt(sum((x - avg_ndcg)  ** 2 for x in user_avg_ndcg)  / len(user_avg_ndcg))
        std_srocc = math.sqrt(sum((x - avg_srocc) ** 2 for x in user_avg_srocc) / len(user_avg_srocc))
        std_ccc  = math.sqrt(
            sum((x - avg_ccc) ** 2 for x in user_avg_ccc) / len(user_avg_ccc)
        ) if user_avg_ccc else None

        print(f"  [{sg}]")
        print(f"    Total users:     {len(all_user_mae[sg])}")
        print(f"    Average MAE:     {avg_mae:.6f} (std: {std_mae:.6f})")
        print(f"    Average NDCG@10: {avg_ndcg:.6f} (std: {std_ndcg:.6f})")
        print(f"    Average SROCC:   {avg_srocc:.6f} (std: {std_srocc:.6f})")
        if avg_ccc is not None:
            print(f"    Average CCC:     {avg_ccc:.6f} (std: {std_ccc:.6f})")

    if cd_user_mae:
        print(f"\n  --- Cross-Domain (head average) ---")
        for target_genre in sorted(cd_user_mae.keys()):
            if not cd_user_mae[target_genre]:
                continue

            user_avg_mae  = [sum(vals) / len(vals) for vals in cd_user_mae[target_genre].values()]
            user_avg_ndcg = [sum(vals) / len(vals) for vals in cd_user_ndcg[target_genre].values()]
            user_avg_srocc = [sum(vals) / len(vals) for vals in cd_user_srocc[target_genre].values()]
            user_avg_ccc  = [sum(vals) / len(vals) for vals in cd_user_ccc[target_genre].values()]

            avg_mae   = sum(user_avg_mae)   / len(user_avg_mae)
            avg_ndcg  = sum(user_avg_ndcg)  / len(user_avg_ndcg)
            avg_srocc = sum(user_avg_srocc) / len(user_avg_srocc)
            avg_ccc   = sum(user_avg_ccc)   / len(user_avg_ccc) if user_avg_ccc else None

            std_mae  = math.sqrt(sum((x - avg_mae)   ** 2 for x in user_avg_mae)   / len(user_avg_mae))
            std_ndcg = math.sqrt(sum((x - avg_ndcg)  ** 2 for x in user_avg_ndcg)  / len(user_avg_ndcg))
            std_srocc = math.sqrt(sum((x - avg_srocc) ** 2 for x in user_avg_srocc) / len(user_avg_srocc))
            std_ccc  = math.sqrt(
                sum((x - avg_ccc) ** 2 for x in user_avg_ccc) / len(user_avg_ccc)
            ) if user_avg_ccc else None

            print(f"  [{genre} -> {target_genre}]")
            print(f"    Total users:     {len(cd_user_mae[target_genre])}")
            print(f"    Average MAE:     {avg_mae:.6f} (std: {std_mae:.6f})")
            print(f"    Average NDCG@10: {avg_ndcg:.6f} (std: {std_ndcg:.6f})")
            print(f"    Average SROCC:   {avg_srocc:.6f} (std: {std_srocc:.6f})")
            if avg_ccc is not None:
                print(f"    Average CCC:     {avg_ccc:.6f} (std: {std_ccc:.6f})")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Analysis utilities for XPass project',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    agg_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate results across folds",
    )
    agg_parser.add_argument(
        "--version", type=str, required=True, help="Dataset version (e.g., v3)"
    )
    agg_parser.add_argument(
        "--genre", type=str, required=True, help="Genre (e.g., art, scenery)"
    )
    agg_parser.add_argument(
        "--pattern",
        type=str,
        default="",
        help="Glob pattern to match JSON files. e.g., pretrain, finetune",
    )
    agg_parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method name to filter JSON files (e.g., ICI). Used when multiple methods match the pattern.",
    )
    agg_parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Specific fold indices to aggregate (e.g., --folds 0 2 4). If omitted, all folds are used.",
    )
    agg_parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific run IDs to include (e.g., --ids 61 65 70). Only files whose ID matches one of these values are aggregated.",
    )
    agg_parser.add_argument(
        "--min-id",
        type=int,
        default=None,
        dest="min_id",
        help="Minimum run ID to include (e.g., 61 filters to files with ID >= 61, like 'name-61_pretrain.json')",
    )
    agg_parser.add_argument(
        "--max-id",
        type=int,
        default=None,
        dest="max_id",
        help="Maximum run ID to include (e.g., 80 filters to files with ID <= 80, like 'name-80_pretrain.json')",
    )
    agg_parser.add_argument(
        "--reports_dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Path to reports/exp directory",
    )
    agg_parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        dest="data_dir",
        help="Path to data directory containing split/ and maked/ (used with --pattern claude). "
             "Default: <project_root>/data",
    )
    agg_parser.add_argument(
        "--giaa_mode",
        action="store_true",
        default=False,
        dest="giaa_mode",
        help="Aggregate GIAA results (EMD/SROCC/MSE/MAE/CCC). "
             "For NN: reads average_metrics from GIAA JSONs. "
             "For LLM (--pattern claude/gemini/gpt): evaluates image-level predictions against test_images_GIAA.txt.",
    )

    args = parser.parse_args()

    if args.command == 'aggregate':
        aggregate(args)
    else:
        parser.print_help()
