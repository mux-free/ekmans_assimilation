@dataclass
class SimConfig:
    total_run_time: int
    ug_truth: float
    K_truth: float
    ug_error: float
    K_error: float
    ensemble_size: int

    data_assimilation: dict
    observations: dict

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
