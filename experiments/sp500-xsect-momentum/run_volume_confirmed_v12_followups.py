from common import (
    build_volume_confirmed_iteration_v12,
    build_volume_confirmed_iteration_v15,
    build_volume_confirmed_iteration_v16,
    build_volume_confirmed_iteration_v17,
)

BUILDERS = {
    "v12": build_volume_confirmed_iteration_v12,
    "v15": build_volume_confirmed_iteration_v15,
    "v16": build_volume_confirmed_iteration_v16,
    "v17": build_volume_confirmed_iteration_v17,
}


def main():
    for key, builder in BUILDERS.items():
        study_name = f"sp500_xsect_momentum_{key}_volume_confirmed"
        study = builder(study_name)
        metrics = study.metrics_dict()
        print(
            f"{key}: sharpe={metrics['sharpe']:.3f}, "
            f"ann_return={metrics['ann_return']:.3f}, "
            f"max_drawdown={metrics['max_drawdown']:.3f}"
        )


if __name__ == "__main__":
    main()
