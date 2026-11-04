"""i18n.py — the 中文 / English switch of the whole UI.

How it works (Gradio 4.19):
  * The server keeps every component's label and button text in English. Those are the names the positional HTTP API, the Studio Loop's
    overrides, the tests and the drivers use, so switching the language never changes the API.
  * The browser swaps what is shown. `page_js()` carries one dictionary English -> 中文: the control names of `backend/ui_zh.ZH` plus every
    pair registered with `T(en, zh)` where a hint, placeholder or button is written. It replaces each text node, placeholder and title that
    equals an English key, and a MutationObserver re-applies it whenever Gradio re-renders a component. 'English' puts the originals back.
  * Messages and guide blocks that the server writes as HTML carry both languages: `L(en, zh)` = <span class="ig-en">…</span><span
    class="ig-zh">…</span>, and CSS shows the chosen one. A message already on screen follows the switch, and an HTTP client still finds the
    English text in it.
  * The choice is kept in the browser (localStorage 'imagegen_lang'); 中文 is the default.
Choice lists that hold technical names (samplers, files, the user's own setting names) are not translated.
"""
from __future__ import annotations

import json
import re

_CJK = re.compile(r"[一-鿿]")
_PAIRS: dict[str, str] = {}

LANG_KEY = "imagegen_lang"


def T(en: str, zh: str) -> str:
    """Register the Chinese of a UI string (a hint, placeholder, button or title) and return the English, which is what the server keeps."""
    if isinstance(en, str) and en.strip() and isinstance(zh, str) and zh.strip():
        _PAIRS[en.strip()] = zh.strip()
    return en


def L(en: str, zh: str, tag: str = "span") -> str:
    """HTML that shows `en` or `zh`, whichever language is chosen (both are in the page; the stylesheet hides one).
    `tag="div"` for block content (paragraphs, lists, tables)."""
    return f'<{tag} class="ig-en">{en}</{tag}><{tag} class="ig-zh">{zh}</{tag}>'


def both(en: str, zh: str) -> str:
    """Plain text that cannot carry HTML (a toast, a text box, a progress line): both languages, Chinese first."""
    return f"{zh} / {en}" if zh and en else (zh or en)


def has_zh(s) -> bool:
    return isinstance(s, str) and bool(_CJK.search(s))


# a leading / trailing emoji or symbol of the English (🎨, ✨, ▶ …) is kept on the Chinese when the Chinese has none of its own
_LEAD = re.compile(r"^([^\w\s]+\s*)")
_TRAIL = re.compile(r"(\s*[^\w\s]+)$")


def _keep_symbols(en: str, zh: str) -> str:
    m = _LEAD.match(en)
    if m and not m.group(1).isascii() and not zh.startswith(m.group(1).strip()):
        zh = m.group(1) + zh
    m = _TRAIL.search(en)
    if m and not m.group(1).isascii() and not zh.endswith(m.group(1).strip()):
        zh = zh + m.group(1)
    return zh


def table() -> dict[str, str]:
    """English -> 中文 for everything the browser swaps: the control names, then the registered hints / buttons (a registered pair wins)."""
    from backend.ui_zh import ZH
    from backend.ui_messages import exact
    out: dict[str, str] = {}
    for en, zh in list(ZH.items()) + list(exact().items()) + list(_PAIRS.items()):
        if isinstance(en, str) and en.strip() and isinstance(zh, str) and zh.strip() and not has_zh(en):
            out[en.strip()] = _keep_symbols(en.strip(), zh.strip())
    return out


def missing(blocks) -> list[str]:
    """Visible static texts of a built UI that the switch cannot show in Chinese: labels, hints, placeholders, button texts, tab and
    accordion titles with no entry (texts that already hold Chinese count as done). A test keeps this empty."""
    import gradio as gr
    tab = table()
    miss = []

    def need(s):
        if isinstance(s, str) and s.strip() and not has_zh(s) and re.search(r"[A-Za-z]{2,}", s) and s.strip() not in tab:
            miss.append(s.strip())

    for b in list(getattr(blocks, "blocks", {}).values()):
        if "ig-notr" in (getattr(b, "elem_classes", None) or []):
            continue                                   # tag chips: they show the tag they insert
        if isinstance(b, gr.Button):
            need(b.value)
        elif isinstance(b, (gr.Tab, gr.Accordion)):
            need(b.label)
        elif isinstance(b, gr.components.Component):
            if getattr(b, "show_label", True) is not False:
                need(getattr(b, "label", None))
            info = getattr(b, "info", None)
            if not callable(info):
                need(info)
            need(getattr(b, "placeholder", None))
    return sorted(set(miss))


def text_nodes(html: str) -> list[str]:
    """The text pieces of an HTML block as the page's script sees them (each one is swapped as a whole): text in textarea / input /
    select / code / pre / script / style / svg and in .ig-en / .ig-notr is left out, and so are .hg-code spans (a tag in the Help,
    which stays what you type)."""
    from html.parser import HTMLParser
    skip_tags = {"textarea", "input", "select", "code", "pre", "script", "style", "svg"}
    skip_cls = {"ig-en", "ig-notr", "hg-code"}
    void = {"br", "img", "hr", "input", "meta", "link", "wbr", "col", "source"}

    class _P(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.stack, self.out = [], []

        def handle_starttag(self, tag, attrs):
            if tag not in void:
                cls = set((dict(attrs).get("class") or "").split())
                self.stack.append((tag, tag in skip_tags or bool(cls & skip_cls)))

        def handle_endtag(self, tag):
            if tag in void or not any(t == tag for t, _ in self.stack):
                return
            while self.stack and self.stack.pop()[0] != tag:     # an unclosed <p> / <li> is closed with its parent
                pass

        def handle_data(self, data):
            t = re.sub(r"\s+", " ", data).strip()
            if t and not any(s for _, s in self.stack):
                self.out.append(t)
    p = _P()
    p.feed(html or "")
    return p.out


def english_left(texts, prose: bool = True) -> list[str]:
    """The texts (text_nodes, labels) that the switch would leave English in 中文 mode: no entry, no template (templates apply in
    HTML / Markdown blocks only: prose=False for other places), no Chinese of their own."""
    from backend import ui_messages as M
    tab, ex = table(), M.exact()
    out = []
    for t in texts:
        n = M._norm(t)
        if not n or has_zh(n) or not re.search(r"[A-Za-z]{2}", n) or n in tab or n in ex:
            continue
        if prose and M.translate(n) is not None:
            continue
        out.append(n)
    return out


CSS = """
html:not([data-ig-lang="en"]) .ig-en { display: none !important; }
html[data-ig-lang="en"] .ig-zh { display: none !important; }
#ig-lang-btn { margin-left: 12px; padding: 4px 12px; border: 1px solid #45475a; border-radius: 6px; background: #313244;
               color: #cdd6f4; font-size: 13px; cursor: pointer; }
#ig-lang-btn:hover { border-color: #89b4fa; }
"""


def toggle_button_html() -> str:
    """The switch in the page header (wired by `page_js`, no server event: the API is not touched)."""
    return ('<button id="ig-lang-btn" type="button" title="中文 / English">'
            '<span class="ig-zh">English</span><span class="ig-en">中文</span></button>')


_JS = r"""
  const IG_ZH = Object.assign(Object.create(null), __TABLE__);     // no prototype: a text 'toString' / 'valueOf' is not a key
  const IG_PAT = __PATTERNS__.map((p) => [new RegExp(p[0]), p[1]]);    // messages with values in them (backend/ui_messages.TEMPLATES)
  const IG_KEY = __KEY__;
  const igLang = () => (localStorage.getItem(IG_KEY) === 'en' ? 'en' : 'zh');
  const igSkip = 'textarea, input, select, code, pre, script, style, .ig-en, .ig-zh, .ig-notr, ul.options, [data-testid="textbox"]';
  const igOrig = new WeakMap(), igShown = new WeakMap();
  const igText = (node) => {
    const p = node.parentElement;
    if (!p || p.closest(igSkip)) return;
    const cur = node.data;
    if (igLang() === 'zh') {
      if (igShown.get(node) === cur) return;
      const t = cur.trim();
      let z = t && IG_ZH[t];
      if (!z && t && /[A-Za-z]{2}/.test(t) && p.closest('.prose')) {
        const n = t.replace(/\s+/g, ' ');
        z = IG_ZH[n];
        if (!z) for (const [r, s] of IG_PAT) { if (r.test(n)) { z = n.replace(r, s); break; } }
      }
      if (!z) return;
      const s = cur.replace(t, () => z);                 // a function: a '$' in the Chinese stays literal
      igOrig.set(node, cur); igShown.set(node, s); node.data = s;
    } else if (igShown.get(node) === cur && igOrig.has(node)) {
      const o = igOrig.get(node); igShown.delete(node); node.data = o;
    }
  };
  const igAttr = (el) => {
    for (const a of ['placeholder', 'title']) {
      const cur = el.getAttribute && el.getAttribute(a);
      if (!cur) continue;
      el.__igO = el.__igO || {}; el.__igS = el.__igS || {};
      if (igLang() === 'zh') {
        if (el.__igS[a] === cur) continue;
        const z = IG_ZH[cur.trim()];
        if (z) { el.__igO[a] = cur; el.__igS[a] = z; el.setAttribute(a, z); }
      } else if (el.__igS[a] === cur && el.__igO[a] !== undefined) {
        delete el.__igS[a]; el.setAttribute(a, el.__igO[a]);
      }
    }
  };
  const igWalk = (root) => {
    if (!root) return;
    if (root.nodeType === 3) { igText(root); return; }
    if (root.nodeType !== 1) return;
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = w.nextNode(); n; n = w.nextNode()) igText(n);
    igAttr(root);
    root.querySelectorAll('[placeholder], [title]').forEach(igAttr);
  };
  const igApply = () => {
    document.documentElement.setAttribute('data-ig-lang', igLang());
    document.documentElement.setAttribute('lang', igLang() === 'zh' ? 'zh-CN' : 'en');
    igWalk(document.body);
  };
  igApply();
  new MutationObserver((muts) => {
    for (const m of muts) {
      if (m.type === 'characterData') igText(m.target);
      else if (m.type === 'attributes') igAttr(m.target);
      else m.addedNodes.forEach(igWalk);
    }
  }).observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ['placeholder', 'title'] });
  document.addEventListener('click', (e) => {
    if (!e.target.closest || !e.target.closest('#ig-lang-btn')) return;
    localStorage.setItem(IG_KEY, igLang() === 'zh' ? 'en' : 'zh');
    igApply();
  });
  window.igLang = igLang;
"""


def page_js(base_js: str | None = None) -> str:
    """The page-load function: `base_js` (another `() => {...}`, run first) + the switch with the current table."""
    from backend.ui_messages import patterns
    body = (_JS.replace("__TABLE__", json.dumps(table(), ensure_ascii=False))
               .replace("__PATTERNS__", json.dumps(patterns(), ensure_ascii=False))
               .replace("__KEY__", json.dumps(LANG_KEY)))
    first = ""
    if base_js and base_js.strip():
        first = f"  try {{ ({base_js.strip()})(); }} catch (e) {{ console.error(e); }}\n"
    return "() => {\n" + first + body + "}\n"


# the value a request's js= hands to the server (Generate in 一键出图 says which language its own messages should use)
LANG_ARG_JS = f"() => (localStorage.getItem({json.dumps(LANG_KEY)}) === 'en' ? 'en' : 'zh')"
