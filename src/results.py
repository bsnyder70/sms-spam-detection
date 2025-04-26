from typing import TypedDict, Any


class BestResult(TypedDict):
    param: str
    f1: float
    precision: float
    recall: float
    val_loss: float
    train_loss: float
    epoch: int
    config: dict[str, Any]
    sweep_param: str
    sweep_value: Any
    num_params: int
    loss_label: str

    @staticmethod
    def init() -> "BestResult":
        return BestResult(
            param="",
            f1=0.0,
            precision=0.0,
            recall=0.0,
            val_loss=float("inf"),
            train_loss=float("inf"),
            epoch=0,
            config={},
            sweep_param="",
            sweep_value=None,
            num_params=0,
            loss_label="",
        )
