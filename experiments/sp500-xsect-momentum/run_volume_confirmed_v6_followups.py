from common import (
    build_volume_confirmed_iteration_v6,
    build_volume_confirmed_iteration_v11,
    build_volume_confirmed_iteration_v12,
    build_volume_confirmed_iteration_v13,
    build_volume_confirmed_iteration_v14,
)

BUILDERS = {
    "v6": build_volume_confirmed_iteration_v6,
    "v11": build_volume_confirmed_iteration_v11,
    "v12": build_volume_confirmed_iteration_v12,
    "v13": build_volume_confirmed_iteration_v13,
    "v14": build_volume_confirmed_iteration_v14,
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
