from common import (
    build_cross_sectional_momentum_study,
    build_industry_relative_momentum_study,
    build_residual_momentum_study,
    build_sector_rotation_study,
    build_volatility_adjusted_momentum_study,
    build_volume_confirmed_momentum_study,
)

BUILDERS = {
    "cross_sectional": build_cross_sectional_momentum_study,
    "sector_rotation": build_sector_rotation_study,
    "residual": build_residual_momentum_study,
    "industry_relative": build_industry_relative_momentum_study,
    "volatility_adjusted": build_volatility_adjusted_momentum_study,
    "volume_confirmed": build_volume_confirmed_momentum_study,
}


def main():
    for key, builder in BUILDERS.items():
        study_name = f"sp500_xsect_momentum_v1_{key}"
        study = builder(study_name)
        metrics = study.metrics_dict()
        print(
            f"{key}: sharpe={metrics['sharpe']:.3f}, "
            f"ann_return={metrics['ann_return']:.3f}, "
            f"max_drawdown={metrics['max_drawdown']:.3f}"
        )


if __name__ == "__main__":
    main()
