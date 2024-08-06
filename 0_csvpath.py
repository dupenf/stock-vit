import os

# base_dir = os.path.join(os.path.dirname(__file__), "../datasets")


def csv_paths(dataset_path=""):
    qfq_weekly = os.path.join(dataset_path, "weekly_qfq")
    qfq_daily = os.path.join(dataset_path, "daily_qfq")
    hfq_weekly = os.path.join(dataset_path, "weekly_hfq")
    hfq_daily = os.path.join(dataset_path, "daily_hfq")

    features_weekly = os.path.join(dataset_path, "weekly_features")
    features_daily = os.path.join(dataset_path, "daily_features")

    templates_weekly = os.path.join(dataset_path, "weekly_templates")
    templates_daily = os.path.join(dataset_path, "daily_templates")

    results_weekly = os.path.join(dataset_path, "weekly_results")
    results_daily = os.path.join(dataset_path, "daily_results")

    if not os.path.exists(qfq_daily):
        os.makedirs(qfq_daily)
    if not os.path.exists(qfq_weekly):
        os.makedirs(qfq_weekly)

    if not os.path.exists(hfq_weekly):
        os.makedirs(hfq_weekly)
    if not os.path.exists(hfq_daily):
        os.makedirs(hfq_daily)

    if not os.path.exists(features_weekly):
        os.makedirs(features_weekly)
    if not os.path.exists(features_daily):
        os.makedirs(features_daily)

    if not os.path.exists(templates_weekly):
        os.makedirs(templates_weekly)
    if not os.path.exists(templates_daily):
        os.makedirs(templates_daily)

    if not os.path.exists(results_weekly):
        os.makedirs(results_weekly)
    if not os.path.exists(results_daily):
        os.makedirs(results_daily)

    week = {
        "qfq": qfq_weekly,
        "hfq": hfq_weekly,
        "features": features_weekly,
        "templates": templates_weekly,
        "results": results_weekly
    }
    daily = {
        "qfq": qfq_daily,
        "hfq": hfq_daily,
        "features": features_daily,
        "templates": templates_daily,
        "results": results_daily
    }
    all_codes_path = os.path.join(dataset_path, "0all.csv")
    return all_codes_path, week, daily
