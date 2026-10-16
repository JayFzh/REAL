# REAL project page

Source of the project page for **REAL: Efficient Distributed Training of Dynamic GNNs with Reuse-Aware Load Balancing Scheduling** (SC ’26), served by GitHub Pages from this `gh-pages` branch at <https://jayfzh.github.io/REAL/>.

The code and artifact live on the [`main`](https://github.com/JayFzh/REAL/tree/main) branch.

- Static site, no build step: `index.html`, `static/css/style.css`, `static/js/main.js`.
- `static/js/data.js` holds the measured speedups behind the interactive explorer. The values were read from the vector bars of the paper's Figs. 7–8 (normalized to ESDG).
- Math is rendered with a self-hosted copy of [KaTeX](https://katex.org) (MIT). The title face is [Cinzel](https://github.com/google/fonts/tree/main/ofl/cinzel) (SIL OFL), subset to Latin.
- Colors follow the Shanghai Jiao Tong University visual identity (交大红 `#C8161E`). The SJTU emblem and logotype belong to Shanghai Jiao Tong University.

Preview locally with `python3 -m http.server` and open <http://localhost:8000>.
