from pathlib import Path
from importlib import import_module


_CONFIG_MODULES = {
    "3DMatch": "threedmatch_config",
    "3DLoMatch": "threedlomatch_config",
    "Scannetpp_iphone": "scannetpp_iphone_config",
    "Scannetpp_faro": "scannetpp_faro_config",
    "TIERS": "tiers_config",
    "TIERS_hetero": "tiers_hetero_config",
    "KITTI": "kitti_config",
    "WOD": "wod_config",
    "MIT": "mit_config",
    "KAIST": "kaist_config",
    "KAIST_hetero": "kaist_hetero_config",
    "ETH": "eth_config",
    "Oxford": "oxford_config",
    "ModelNet40": "modelnet40_config",
}


def make_cfg(dataset_name, root_dir=None):
    """
    Generalized function to return the appropriate configuration based on dataset name.
    """
    if root_dir is None:
        root_dir = Path("../datasets")
    elif not isinstance(root_dir, Path):
        root_dir = Path(root_dir)

    module_name = _CONFIG_MODULES.get(dataset_name)
    if module_name is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    try:
        module = import_module(f".{module_name}", package=__name__)
    except ModuleNotFoundError as exc:
        if exc.name == f"{__name__}.{module_name}":
            raise ModuleNotFoundError(
                f"Configuration module for dataset '{dataset_name}' is missing: "
                f"config/{module_name}.py"
            ) from exc
        raise

    return module.make_cfg(root_dir)
