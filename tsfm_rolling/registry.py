from tsfm_rolling.adapters.chronos2 import Chronos2Adapter
from tsfm_rolling.adapters.hart36m import HART36MAdapter
from tsfm_rolling.adapters.patchtst_fm_r1 import PatchTSTFMR1Adapter
from tsfm_rolling.adapters.timesfm25 import TimesFM25Adapter
from tsfm_rolling.adapters.timesfm30 import TimesFM30Adapter
from tsfm_rolling.adapters.timestral import TimestralAdapter
from tsfm_rolling.adapters.tirex import TiRexAdapter
from tsfm_rolling.adapters.tirex2 import TiRex2Adapter
from tsfm_rolling.adapters.toto2 import Toto2Adapter


def build_model(cfg: dict):
    mcfg = cfg["model"]
    fcfg = cfg["forecast"]
    name = mcfg["name"].lower()
    path = mcfg["checkpoint_path"]
    device = mcfg.get("device", "cuda")
    context_limit = int(fcfg.get("context_limit", 8192))
    batch_size = int(mcfg.get("batch_size", 1))

    if name == "chronos2":
        return Chronos2Adapter(path, device, context_limit)

    if name == "hart_36m":
        return HART36MAdapter(
            checkpoint_path=path,
            repo_path=mcfg["hart_repo_path"],
            kernel_type=mcfg.get("hart_kernel_type", "cuda"),
            context_limit=context_limit,
        )

    if name == "timesfm_2_5":
        return TimesFM25Adapter(
            checkpoint_path=path,
            context_limit=context_limit,
            max_horizon=int(fcfg["horizon"]),
            batch_size=batch_size,
            device=device,
            normalize_inputs=mcfg.get("timesfm_normalize_inputs", True),
            infer_is_positive=mcfg.get("timesfm_infer_is_positive", False),
        )

    if name == "timesfm_3_0":
        return TimesFM30Adapter(path, batch_size, device, context_limit)

    if name in {"toto_2_0_22m", "toto_2_0_313m"}:
        return Toto2Adapter(path, device, context_limit)

    if name in {
        "timestral_5m",
        "timestral_28m",
        "timestral_123m",
        "timestral_209m",
    }:
        return TimestralAdapter(path, device, context_limit)

    if name == "tirex":
        return TiRexAdapter(
            checkpoint_path=path,
            device=device,
            backend=mcfg.get("tirex_backend", "torch"),
            compile_model=mcfg.get("tirex_compile", False),
            context_limit=context_limit,
        )

    if name == "tirex2":
        return TiRex2Adapter(path, device, context_limit)

    if name == "patchtst_fm_r1":
        return PatchTSTFMR1Adapter(path, device, context_limit)

    raise ValueError(f"Unsupported model name: {name}")
