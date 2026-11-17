# Saathi CRM (demo site)

Marketing site for Saathi, a fictional AI-native CRM. Two pages: product (`index.html`) and pricing (`pricing.html`), with prices in Indian rupees.

- Edit the sources in `src/`: `index.html` and `pricing.html` (page head, then `<!--/HEAD-->`, then the body), plus the shared `header.html` and `footer.html`.
- Shared styles and scripts live in `assets/site.css` and `assets/site.js`.
- Run `python3 build.py` to regenerate `index.html` and `pricing.html`.
- Pushing to `main` deploys to GitHub Pages through `.github/workflows/pages.yml`.

No dependencies or build tools beyond Python 3.
