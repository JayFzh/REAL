/* REAL project page: tabs, lightbox, nav state, BibTeX copy, and the speedup explorer. */
(function () {
  'use strict';

  /* ---------- math ---------- */
  function renderMath() {
    if (typeof renderMathInElement !== 'function') return;
    renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\(', right: '\\)', display: false }
      ],
      throwOnError: false
    });
  }

  /* ---------- tabs ---------- */
  function initTabs() {
    document.querySelectorAll('[data-tabs]').forEach(function (root) {
      var tabs = Array.prototype.slice.call(root.querySelectorAll('[data-tab]'));
      var panels = root.querySelectorAll('[data-panel]');
      function select(name, focus) {
        tabs.forEach(function (t) {
          var on = t.dataset.tab === name;
          t.setAttribute('aria-selected', on ? 'true' : 'false');
          t.tabIndex = on ? 0 : -1;
          if (on && focus) t.focus();
        });
        panels.forEach(function (p) { p.hidden = p.dataset.panel !== name; });
      }
      tabs.forEach(function (t, i) {
        t.addEventListener('click', function () { select(t.dataset.tab); });
        t.addEventListener('keydown', function (e) {
          var d = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0;
          if (!d) return;
          e.preventDefault();
          select(tabs[(i + d + tabs.length) % tabs.length].dataset.tab, true);
        });
      });
      var initial = tabs.filter(function (t) { return t.getAttribute('aria-selected') === 'true'; })[0] || tabs[0];
      if (initial) select(initial.dataset.tab);
    });
  }

  /* ---------- lightbox ---------- */
  function initLightbox() {
    var box = document.getElementById('lightbox');
    var img = box.querySelector('img');
    function close() { box.hidden = true; img.removeAttribute('src'); }
    document.querySelectorAll('.zoomable img').forEach(function (el) {
      el.addEventListener('click', function () {
        img.src = el.currentSrc || el.src;
        img.alt = el.alt;
        box.hidden = false;
      });
    });
    box.addEventListener('click', close);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !box.hidden) close(); });
  }

  /* ---------- nav ---------- */
  function initNav() {
    if (!('IntersectionObserver' in window)) return;
    var links = {};
    document.querySelectorAll('.nav-links a').forEach(function (a) { links[a.getAttribute('href').slice(1)] = a; });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting || !links[en.target.id]) return;
        Object.keys(links).forEach(function (k) { links[k].classList.toggle('active', k === en.target.id); });
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    Object.keys(links).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
  }

  function initReveal() {
    if (!('IntersectionObserver' in window)) return;
    var els = document.querySelectorAll('.tldr, .stats, .principle, .arch-plane, .eq-card, .card, .chain, .cmp, .insights, .explorer, .ablation, .fallback, .paper-row');
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
      });
    }, { rootMargin: '0px 0px -8% 0px' });
    els.forEach(function (el) { el.classList.add('reveal'); io.observe(el); });
  }

  /* ---------- BibTeX ---------- */
  function initCopy() {
    var btn = document.getElementById('copy-bib');
    var pre = document.getElementById('bib-text');
    btn.addEventListener('click', function () {
      var text = pre.textContent;
      var done = function () { btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = 'Copy'; }, 1600); };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done);
      } else {
        var r = document.createRange(); r.selectNodeContents(pre);
        var sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
        document.execCommand('copy'); sel.removeAllRanges(); done();
      }
    });
  }

  /* ---------- speedup explorer (Figs. 7-8) ---------- */
  function initExplorer() {
    var root = document.getElementById('explorer');
    var DATA = window.REAL_SPEEDUPS;
    if (!root || !DATA) return;

    var SYS = ['ESDG', 'DistDGL', 'BLAD', 'Real-G', 'Real-L'];
    var MODELS = [['EG', 'EvolveGCN'], ['WG', 'WD-GCN'], ['TG', 'TGCN'], ['GL', 'GAT-LSTM']];
    var SCALE = { g8: '8 GPUs', g32: '32 GPUs' };
    var DEPTH = { l1: '1 layer', l2: '2 layers' };
    // 8-step OKLab ramps of SJTU red; text ink flips at step 5 (contrast checked per step).
    var RAMP = {
      light: ['#fbe7e8', '#f8cacc', '#f2acb1', '#e78285', '#d64c51', '#bc142a', '#9b0f21', '#7a0a19'],
      dark: ['#3a2024', '#562126', '#721f28', '#942430', '#b92e3c', '#da4b55', '#ee8286', '#ffb3b8']
    };
    var INK = { light: ['#1d1617', '#ffffff'], dark: ['#f3eeee', '#170f11'] };
    var SVGNS = 'http://www.w3.org/2000/svg';
    var dark = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

    var state = { scale: 'g8', depth: 'l1', base: 2, ours: 4, sel: { ds: 'Reddit', m: 'EG' }, peek: null };

    var heat = document.getElementById('heat');
    var legend = document.getElementById('heat-legend');
    var detail = document.getElementById('detail');
    var tooltip = document.getElementById('ex-tooltip');

    function mode() { return dark && dark.matches ? 'dark' : 'light'; }
    function cfg() { return DATA[state.scale + '_' + state.depth]; }
    function fmt(v) { return v.toFixed(2) + '×'; }
    function modelName(m) { return MODELS.filter(function (x) { return x[0] === m; })[0][1]; }
    function ratio(row, a, b) { return row[a] === null || row[b] === null ? null : row[a] / row[b]; }

    function el(name, attrs, parent) {
      var n = document.createElementNS(SVGNS, name);
      for (var a in attrs) n.setAttribute(a, attrs[a]);
      if (parent) parent.appendChild(n);
      return n;
    }
    function div(cls, text, parent) {
      var n = document.createElement('div');
      if (cls) n.className = cls;
      if (text !== undefined) n.textContent = text;
      if (parent) parent.appendChild(n);
      return n;
    }

    function cells() {
      var c = cfg(), out = [];
      Object.keys(c).forEach(function (ds) {
        MODELS.forEach(function (m) { out.push({ ds: ds, m: m[0], v: ratio(c[ds][m[0]], state.ours, state.base) }); });
      });
      return out;
    }

    function domain(list) {
      var max = 1;
      list.forEach(function (c) { if (c.v !== null && c.v > max) max = c.v; });
      return [1, Math.max(1.5, Math.ceil(max * 2) / 2)];
    }
    function bin(v, dom) {
      var t = Math.log(Math.max(v, dom[0]) / dom[0]) / Math.log(dom[1] / dom[0]);
      return Math.max(0, Math.min(7, Math.floor(t * 8)));
    }

    /* readout: reproduces the ranges/averages quoted in the paper */
    function readout(list) {
      var vals = list.filter(function (c) { return c.v !== null; });
      var oom = list.length - vals.length;
      var lo = Infinity, hi = -Infinity, sum = 0, best = null;
      vals.forEach(function (c) { lo = Math.min(lo, c.v); sum += c.v; if (c.v > hi) { hi = c.v; best = c; } });
      var ours = SYS[state.ours], base = SYS[state.base];
      var ro = document.getElementById('ex-readout');
      ro.textContent = '';
      [
        [lo.toFixed(2) + '–' + fmt(hi), ours + ' speedup over ' + base + ', min to max'],
        [fmt(sum / vals.length), 'average over ' + vals.length + ' settings' + (oom ? ' (' + base + ' out of memory in ' + oom + ')' : '')],
        [best.ds + ' · ' + best.m, 'largest gain: ' + modelName(best.m) + ' on ' + best.ds]
      ].forEach(function (it) {
        var d = div('ro', undefined, ro);
        div('ro-v', it[0], d); div('ro-l', it[1], d);
      });
    }

    function drawHeat(list) {
      var c = cfg(), dom = domain(list), m = mode();
      heat.textContent = '';
      var t = document.getElementById('heat-title');
      t.textContent = '';
      t.appendChild(document.createTextNode(SYS[state.ours] + ' speedup over ' + SYS[state.base] + ' '));
      var sub = document.createElement('span'); sub.className = 'sub';
      sub.textContent = '· ' + SCALE[state.scale] + ', ' + DEPTH[state.depth];
      t.appendChild(sub);

      div('heat-h', '', heat);
      MODELS.forEach(function (md) {
        var h = div('heat-h', undefined, heat);
        var ab = document.createElement('abbr'); ab.title = md[1]; ab.textContent = md[0]; h.appendChild(ab);
      });
      var dsList = Object.keys(c);
      dsList.forEach(function (ds, r) {
        div('heat-r', ds, heat);
        MODELS.forEach(function (md, k) {
          var cell = list[r * MODELS.length + k];
          var b = document.createElement('button');
          b.type = 'button';
          b.className = 'heat-cell';
          b.dataset.r = r; b.dataset.k = k;
          if (cell.v === null) {
            b.classList.add('na');
            b.textContent = 'OOM';
            b.setAttribute('aria-label', ds + ', ' + md[1] + ': ' + SYS[state.base] + ' runs out of memory');
          } else {
            var i = bin(cell.v, dom);
            b.style.background = RAMP[m][i];
            b.style.color = INK[m][i >= 5 ? 1 : 0];
            b.textContent = cell.v.toFixed(2);
            b.setAttribute('aria-label', ds + ', ' + md[1] + ': ' + SYS[state.ours] + ' is ' + cell.v.toFixed(2) + ' times as fast as ' + SYS[state.base]);
          }
          if (ds === state.sel.ds && md[0] === state.sel.m) { b.classList.add('sel'); b.setAttribute('aria-pressed', 'true'); }
          else b.setAttribute('aria-pressed', 'false');
          b.tabIndex = ds === state.sel.ds && md[0] === state.sel.m ? 0 : -1;
          b.addEventListener('mouseenter', function () { peek({ ds: ds, m: md[0] }); });
          b.addEventListener('mouseleave', function () { peek(null); });
          b.addEventListener('focus', function () { peek({ ds: ds, m: md[0] }); });
          b.addEventListener('blur', function () { peek(null); });
          b.addEventListener('click', function () { state.sel = { ds: ds, m: md[0] }; state.peek = null; render(); focusCell(r, k); });
          b.addEventListener('keydown', function (e) {
            var dr = e.key === 'ArrowDown' ? 1 : e.key === 'ArrowUp' ? -1 : 0;
            var dk = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
            if (!dr && !dk) return;
            e.preventDefault();
            focusCell(Math.max(0, Math.min(dsList.length - 1, r + dr)), Math.max(0, Math.min(MODELS.length - 1, k + dk)));
          });
          heat.appendChild(b);
        });
      });

      legend.textContent = '';
      var lo = document.createElement('span'); lo.textContent = fmt(dom[0]); legend.appendChild(lo);
      var ramp = document.createElement('span'); ramp.className = 'ramp';
      RAMP[m].forEach(function (col) { var s = document.createElement('span'); s.style.background = col; ramp.appendChild(s); });
      legend.appendChild(ramp);
      var hi = document.createElement('span'); hi.textContent = fmt(dom[1]); legend.appendChild(hi);
      if (list.some(function (c) { return c.v === null; })) {
        var na = document.createElement('span'); na.className = 'na-key';
        var i = document.createElement('i'); na.appendChild(i);
        na.appendChild(document.createTextNode(SYS[state.base] + ' out of memory'));
        legend.appendChild(na);
      }
    }

    function focusCell(r, k) {
      var b = heat.querySelector('.heat-cell[data-r="' + r + '"][data-k="' + k + '"]');
      if (!b) return;
      heat.querySelectorAll('.heat-cell').forEach(function (o) { o.tabIndex = o === b ? 0 : -1; });
      b.focus();
    }

    function peek(cell) {
      state.peek = cell;
      drawDetail();
    }

    /* per-setting bar chart: all five systems for one dataset x model */
    function drawDetail() {
      var cur = state.peek || state.sel;
      var row = cfg()[cur.ds][cur.m];
      var baseIdx = row[state.base] === null ? 0 : state.base;
      var vals = row.map(function (v) { return v === null ? null : v / row[baseIdx]; });

      var title = document.getElementById('detail-title');
      title.textContent = '';
      title.appendChild(document.createTextNode(cur.ds + ' · ' + modelName(cur.m) + ' '));
      var sub = document.createElement('span'); sub.className = 'sub';
      sub.textContent = '· normalized to ' + SYS[baseIdx];
      title.appendChild(sub);
      document.getElementById('detail-note').textContent = baseIdx !== state.base
        ? SYS[state.base] + ' runs out of memory on this setting, so the bars are normalized to ESDG.'
        : 'Hover or focus a cell to preview it, click to pin it. Dashed line = ' + SYS[baseIdx] + '.';

      var w = detail.clientWidth, h = detail.clientHeight;
      var m = { l: 34, r: 8, t: 22, b: 30 };
      var max = 0;
      vals.forEach(function (v) { if (v !== null && v > max) max = v; });
      var step = max > 8 ? 2 : max > 4 ? 1 : max > 2 ? 0.5 : 0.25;
      var top = Math.ceil(max * 1.08 / step) * step;
      var y = function (v) { return m.t + (1 - v / top) * (h - m.t - m.b); };
      var bw = (w - m.l - m.r) / SYS.length;
      var barW = Math.min(46, bw * 0.62);

      detail.textContent = '';
      var svg = el('svg', { viewBox: '0 0 ' + w + ' ' + h, role: 'img', 'aria-label': 'Speedup of each system on ' + cur.ds + ', ' + modelName(cur.m) + ', normalized to ' + SYS[baseIdx] }, detail);
      var grid = el('g', { class: 'grid' }, svg);
      for (var v = 0; v <= top + 1e-9; v += step) {
        var yy = y(v);
        if (v > 0) el('line', { x1: m.l, x2: w - m.r, y1: yy, y2: yy }, grid);
        var t = el('text', { x: m.l - 7, y: yy + 4, 'text-anchor': 'end' }, svg);
        t.textContent = (Math.round(v * 100) / 100) + '×';
      }
      el('line', { class: 'baseline', x1: m.l, x2: w - m.r, y1: y(0), y2: y(0) }, svg);

      var colors = ['var(--bar-base)', 'var(--bar-base)', 'var(--bar-base)', 'var(--bar-g)', 'var(--bar-l)'];
      SYS.forEach(function (name, i) {
        var cx = m.l + bw * (i + 0.5);
        var cat = el('text', { class: 'cat' + (i >= 3 ? ' cat-us' : ''), x: cx, y: h - m.b + 18, 'text-anchor': 'middle' }, svg);
        cat.textContent = name;
        if (i >= 3) cat.style.fontVariantCaps = 'small-caps';
        if (vals[i] === null) {
          var o = el('text', { class: 'oom', x: cx, y: y(0) - 8, 'text-anchor': 'middle' }, svg);
          o.textContent = 'OOM';
          return;
        }
        var x0 = cx - barW / 2, y0 = y(vals[i]), yb = y(0), r = Math.min(4, barW / 2, yb - y0);
        var d = 'M' + x0 + ',' + yb + 'V' + (y0 + r) + 'Q' + x0 + ',' + y0 + ' ' + (x0 + r) + ',' + y0 +
          'H' + (x0 + barW - r) + 'Q' + (x0 + barW) + ',' + y0 + ' ' + (x0 + barW) + ',' + (y0 + r) + 'V' + yb + 'Z';
        el('path', { class: 'bar', d: d, fill: colors[i] }, svg);
        var lab = el('text', { class: 'val', x: cx, y: y0 - 6, 'text-anchor': 'middle' }, svg);
        lab.textContent = fmt(vals[i]);
        var hit = el('rect', { x: cx - bw / 2, y: m.t, width: bw, height: h - m.t - m.b, fill: 'transparent' }, svg);
        hit.addEventListener('pointermove', function (ev) { showTip(ev, cur, i, row, vals[i], baseIdx); });
        hit.addEventListener('pointerleave', function () { tooltip.hidden = true; });
      });

      var yr = y(1);
      el('line', { class: 'ref', x1: m.l, x2: w - m.r, y1: yr, y2: yr }, svg);
    }

    function showTip(ev, cur, i, row, v, baseIdx) {
      tooltip.textContent = '';
      div('tt-head', SYS[i] + ' · ' + cur.ds + ' · ' + cur.m, tooltip);
      var rows = [['vs ' + SYS[baseIdx], v]];
      if (baseIdx !== 0) rows.push(['vs ESDG (as in the paper)', row[i] / row[0]]);
      rows.forEach(function (r) {
        var d = div('tt-row', undefined, tooltip);
        var s = document.createElement('span'); s.textContent = r[0]; d.appendChild(s);
        var b = document.createElement('b'); b.textContent = fmt(r[1]); d.appendChild(b);
      });
      tooltip.hidden = false;
      var box = root.getBoundingClientRect();
      var tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
      var px = ev.clientX - box.left + 14, py = ev.clientY - box.top - th - 12;
      if (px + tw > box.width - 8) px = ev.clientX - box.left - tw - 14;
      if (py < 8) py = ev.clientY - box.top + 16;
      tooltip.style.left = px + 'px'; tooltip.style.top = py + 'px';
    }

    function table() {
      var c = cfg();
      var t = document.getElementById('ex-table');
      t.textContent = '';
      var cap = t.createCaption();
      cap.textContent = SCALE[state.scale] + ', ' + DEPTH[state.depth] + '. Speedup normalized to ' + SYS[state.base] + '; OOM = out of memory.';
      cap.style.captionSide = 'bottom'; cap.style.textAlign = 'left'; cap.style.paddingTop = '8px'; cap.style.color = 'var(--muted)';
      var head = t.createTHead().insertRow();
      ['Dataset', 'Model'].concat(SYS).forEach(function (h) {
        var th = document.createElement('th'); th.scope = 'col'; th.textContent = h; head.appendChild(th);
      });
      var body = t.createTBody();
      Object.keys(c).forEach(function (ds) {
        MODELS.forEach(function (md) {
          var row = c[ds][md[0]], tr = body.insertRow();
          tr.insertCell().textContent = ds;
          tr.insertCell().textContent = md[1];
          SYS.forEach(function (s, i) {
            var td = tr.insertCell();
            var v = ratio(row, i, state.base);
            td.textContent = row[i] === null ? 'OOM' : v === null ? '—' : v.toFixed(2);
            if (i === state.ours) td.className = 'us';
          });
        });
      });
    }

    function render() {
      var c = cfg();
      if (!c[state.sel.ds]) state.sel = { ds: Object.keys(c)[0], m: state.sel.m };
      var list = cells();
      readout(list);
      drawHeat(list);
      drawDetail();
      table();
    }

    root.querySelectorAll('.seg[data-key]').forEach(function (seg) {
      var key = seg.dataset.key;
      var btns = seg.querySelectorAll('button');
      btns.forEach(function (b) {
        b.setAttribute('aria-pressed', b.classList.contains('on') ? 'true' : 'false');
        b.addEventListener('click', function () {
          state[key] = key === 'base' || key === 'ours' ? +b.dataset.v : b.dataset.v;
          btns.forEach(function (o) { var on = o === b; o.classList.toggle('on', on); o.setAttribute('aria-pressed', on ? 'true' : 'false'); });
          render();
        });
      });
    });

    var pending = false;
    var onResize = function () {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function () { pending = false; drawDetail(); });
    };
    if ('ResizeObserver' in window) new ResizeObserver(onResize).observe(detail);
    else window.addEventListener('resize', onResize);
    if (dark) {
      var onScheme = function () { render(); };
      if (dark.addEventListener) dark.addEventListener('change', onScheme); else if (dark.addListener) dark.addListener(onScheme);
    }

    render();
  }

  function init() {
    initTabs();
    initLightbox();
    initNav();
    initCopy();
    initExplorer();
    renderMath();
    initReveal();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
