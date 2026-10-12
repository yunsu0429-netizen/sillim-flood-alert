r"""web/ 화면을 site/로 복사하고 정적 모드(static-adapter.js)를 켠다. 데이터는 src.site.export static site/data.

실행: .venv\Scripts\python -m src.site.build_site
"""
import shutil

from src.config import ROOT

WEB, SITE = ROOT / "web", ROOT / "site"


def build():
    SITE.mkdir(exist_ok=True)
    for f in ("app.js", "style.css", "static-adapter.js"):
        shutil.copy2(WEB / f, SITE / f)
    html = (WEB / "index.html").read_text(encoding="utf-8")
    tag = '<script src="app.js'
    assert tag in html
    html = html.replace(tag, '<script src="static-adapter.js"></script>\n' + tag, 1)
    (SITE / "index.html").write_text(html, encoding="utf-8")
    print("site/ 준비 완료")


if __name__ == "__main__":
    build()
