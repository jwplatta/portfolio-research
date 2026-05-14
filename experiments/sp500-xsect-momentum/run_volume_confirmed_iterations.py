from common import (
    build_volume_confirmed_iteration_v2,
    build_volume_confirmed_iteration_v3,
    build_volume_confirmed_iteration_v4,
    build_volume_confirmed_iteration_v5,
    build_volume_confirmed_iteration_v6,
    build_volume_confirmed_iteration_v7,
    build_volume_confirmed_iteration_v8,
    build_volume_confirmed_iteration_v9,
    build_volume_confirmed_iteration_v10,
    build_volume_confirmed_momentum_study,
)

BUILDERS = {
    "v1": build_volume_confirmed_momentum_study,
    "v2": build_volume_confirmed_iteration_v2,
    "v3": build_volume_confirmed_iteration_v3,
    "v4": build_volume_confirmed_iteration_v4,
    "v5": build_volume_confirmed_iteration_v5,
    "v6": build_volume_confirmed_iteration_v6,
    "v7": build_volume_confirmed_iteration_v7,
    "v8": build_volume_confirmed_iteration_v8,
    "v9": build_volume_confirmed_iteration_v9,
    "v10": build_volume_confirmed_iteration_v10,
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
