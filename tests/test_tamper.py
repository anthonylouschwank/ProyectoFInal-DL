import random

import numpy as np
import torch

from src.build_dataset import generate
from src.model import Detector, letterbox
from src.normalize import fecha
from src.tamper import format_like, parse_amount
from src.validate import nit_is_valid


def test_generation_is_deterministic():
    a = generate(123, 0.25)
    b = generate(123, 0.25)
    assert a[4][2] == b[4][2]  # misma alteración
    assert np.array_equal(np.asarray(a[3]), np.asarray(b[3]))  # misma foto


def test_tampering_is_plausible_fraud():
    for seed in range(60):
        _, inv, _, photo, (fake, mask, info) = generate(seed, 0.3)
        assert fake.size == photo.size == mask.size
        assert mask.getbbox() is not None  # siempre hay zona alterada
        assert info["nuevo"] != info["original"]
        if info["campo"] in ("total", "iva"):  # un fraude infla el monto, no lo baja
            assert parse_amount(info["nuevo"]) > parse_amount(info["original"])
            assert info["monto_en_riesgo"] > 0
        if info["campo"] == "fecha":  # solo día o mes, y una fecha que existe
            assert fecha(info["nuevo"]) is not None
            assert fecha(info["nuevo"])[:4] == fecha(info["original"])[:4]
        if info["campo"] == "nit_receptor" and info["metodo"] == "reescribir":
            assert nit_is_valid(info["nuevo"])  # el defraudador usa un NIT válido


def test_format_like():
    assert format_like("Q 9.24", 109.24) == "Q 109.24"
    assert format_like("Q.69.46", 1069.5) == "Q.1,069.50"
    assert format_like("395.43", 1507.5) == "1507.50"


def test_letterbox_keeps_aspect():
    from PIL import Image

    img = Image.new("RGB", (1200, 1600), "white")
    lb, s, ox, oy = letterbox(img, (768, 576))
    assert lb.size == (576, 768) and abs(s - 0.48) < 1e-9 and ox == 0 and oy == 0


def test_detector_heads_shapes():
    x = torch.randn(2, 3, 256, 192)
    for head in ("gap", "mapa"):
        logit, mapa = Detector(head=head, pretrained=False)(x)
        assert logit.shape == (2,)
        assert (mapa is None) == (head == "gap")
        if mapa is not None:
            assert mapa.shape == (2, 16, 12)


def test_random_module_unused_state():
    # generate() usa su propio random.Random: no depende del estado global
    random.seed(0)
    a = generate(7, 0.25)[4][2]
    random.seed(999)
    assert generate(7, 0.25)[4][2] == a
