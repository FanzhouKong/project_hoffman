"""Icons and the in-page logo from packaging/mosa-logo.png (the full MOSA logo, square, transparent background).

usage: python packaging/make_icons.py      (needs Pillow; rerun after replacing the logo, then commit the outputs)

Writes:
  packaging/mosa.ico                  Windows program icon (16-256 px)
  packaging/mosa.icns                 macOS app icon (16-1024 px)
  peak3d/web/static/logo-256.png      the control window's icon (Tk)
  peak3d/web/static/logo.js           P3.LOGO: the mark as a data URI, for the app, the browser tab and report.html
                                      (inline, so a report shows it offline)
Small sizes (48 px and below) use the mark alone (the mosasaur and the peaks): the MOSA lettering cannot be read
there; larger sizes use the whole logo."""
import base64
import io
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / "peak3d" / "web" / "static"
SRC = HERE / "mosa-logo.png"
LETTERING_TOP = 0.735          # the lettering starts below this fraction of the visible height (the peaks' baseline)


def square(im: Image.Image, pad: float = 0.04) -> Image.Image:
    """the visible part, centred on a transparent square with a small margin"""
    box = im.getchannel("A").point(lambda v: 255 if v > 10 else 0).getbbox()
    im = im.crop(box)
    side = int(max(im.size) * (1 + 2 * pad))
    out = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    out.paste(im, ((side - im.size[0]) // 2, (side - im.size[1]) // 2), im)
    return out


def main() -> None:
    full = Image.open(SRC).convert("RGBA")
    box = full.getchannel("A").point(lambda v: 255 if v > 10 else 0).getbbox()
    cut = box[1] + int((box[3] - box[1]) * LETTERING_TOP)
    mark = square(full.crop((0, 0, full.size[0], cut)))
    whole = square(full)
    size = lambda im, n: im.resize((n, n), Image.LANCZOS)
    small = (16, 24, 32, 48)
    frames = {n: size(mark if n in small else whole, n) for n in (16, 24, 32, 48, 64, 128, 256, 512, 1024)}
    frames[256].save(HERE / "mosa.ico", sizes=[(n, n) for n in (16, 24, 32, 48, 64, 128, 256)],
                     append_images=[frames[n] for n in (16, 24, 32, 48, 64, 128)])
    frames[1024].save(HERE / "mosa.icns", append_images=[frames[n] for n in (16, 32, 64, 128, 256, 512)])
    size(whole, 256).save(STATIC / "logo-256.png", optimize=True)
    buf = io.BytesIO()
    size(mark, 96).save(buf, "PNG", optimize=True)
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    (STATIC / "logo.js").write_text(
        "// the MOSA mark (packaging/make_icons.py writes this file from packaging/mosa-logo.png)\n"
        f'(function (P3) {{ P3.LOGO = "{uri}"; }})(window.P3 = window.P3 || {{}});\n', encoding="utf-8")
    print(f"mark {mark.size} (cut at y={cut}), whole {whole.size}; logo.js {len(uri) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
