"""Prepara las imágenes de fondo de Lente a partir de la fotografía original (CC BY 2.0).

Origen: «Crazy Mountains, Montana», Janusz Sobolewski, Wikimedia Commons, CC BY 2.0.
Ver public/img/CREDITS.md. Cambios aplicados: redimensionado, recorte de proporción para móvil y un
ajuste suave de color (tono ligeramente hacia el violeta, más saturación y sombras algo aclaradas).

    python scripts/prepare_background.py <original.jpg>
"""

import colorsys
from pathlib import Path
import sys

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "public" / "img"


def grade(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    rgb = np.power(rgb, 0.78)                                   # aclara sombras y medios tonos
    hsv = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).convert("HSV"), dtype=np.float32)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = (hue > 135) & (hue < 185)                            # azules (PIL: 0-255; 170 ≈ 240°)
    hue = np.where(blue, hue + 5, hue)                          # ~ +7° hacia el violeta
    sat = np.clip(sat * np.where(blue, 0.92, 1.0), 0, 255)
    rgb = np.asarray(Image.fromarray(np.stack([hue, sat, val], axis=-1).astype(np.uint8), "HSV")
                     .convert("RGB"), dtype=np.float32) / 255.0
    # Resplandor suave de atardecer: banda rosada-anaranjada centrada en la base de las nubes (~34 %).
    height, width = rgb.shape[:2]
    rows = (np.arange(height, dtype=np.float32) / height)[:, None]
    cols = (np.arange(width, dtype=np.float32) / width)[None, :]
    glow = np.exp(-(((rows - 0.34) / 0.10) ** 2)) * (0.55 + 0.45 * np.exp(-(((cols - 0.62) / 0.35) ** 2)))
    tint = np.array([0.93, 0.55, 0.62], dtype=np.float32)       # rosa melocotón
    screened = 1 - (1 - rgb) * (1 - tint * glow[..., None] * 0.34)   # mezcla "screen"
    return Image.fromarray((np.clip(screened, 0, 1) * 255).astype(np.uint8), "RGB")


def main():
    source = Image.open(sys.argv[1]).convert("RGB")
    OUT.mkdir(parents=True, exist_ok=True)
    wide = grade(source.resize((2560, round(2560 * source.height / source.width)), Image.LANCZOS))
    wide.save(OUT / "fondo-2560.jpg", quality=80, optimize=True, progressive=True)
    wide.resize((1600, round(1600 * wide.height / wide.width)), Image.LANCZOS).save(
        OUT / "fondo-1600.jpg", quality=80, optimize=True, progressive=True)
    # Móvil (vertical): recorte centrado en la cordillera.
    width, height = wide.size
    crop_width = round(height * 9 / 16 * 1.1)
    left = round(width * 0.40 - crop_width / 2)
    portrait = wide.crop((left, 0, left + crop_width, height))
    portrait.resize((900, round(900 * portrait.height / portrait.width)), Image.LANCZOS).save(
        OUT / "fondo-movil.jpg", quality=80, optimize=True, progressive=True)
    for path in sorted(OUT.glob("fondo-*.jpg")):
        print(f"{path.name}: {Image.open(path).size}, {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
