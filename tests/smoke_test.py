"""Fast structural checks that do not require datasets or checkpoints."""

from dshcbn.models import CONCEPT_SPECS, build_model


def main():
    assert len(CONCEPT_SPECS) == 8
    cnn = build_model("residual_cnn")
    unet = build_model("unet", encoder_norm="group", encoder_final_channels=192)
    assert sum(p.numel() for p in cnn.parameters()) > 0
    assert sum(p.numel() for p in unet.parameters()) > 0
    print("[PASS] Both DS-HCBN encoder variants construct successfully.")


if __name__ == "__main__":
    main()
