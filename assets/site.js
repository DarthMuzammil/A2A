/* Saathi — shared behaviour: announce bar, nav theme, mobile menu, reveals, buttons */
(() => {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  window.SA = { reduce, $, $$ };

  const bar = $('.announce');
  if (bar) {
    try { if (sessionStorage.getItem('sa-announce') === 'x') bar.hidden = true; } catch (e) {}
    $('.x', bar)?.addEventListener('click', () => { bar.hidden = true; try { sessionStorage.setItem('sa-announce', 'x'); } catch (e) {} });
  }

  const nav = $('.nav');
  const burger = $('.burger'), mnav = $('.mnav');
  burger?.addEventListener('click', () => {
    const open = !mnav.classList.contains('open');
    mnav.classList.toggle('open', open); burger.setAttribute('aria-expanded', open);
  });
  $$('.mnav a').forEach(a => a.addEventListener('click', () => { mnav.classList.remove('open'); burger.setAttribute('aria-expanded', 'false'); }));

  /* nav goes dark over dark sections */
  const darks = $$('[data-dark]');
  function navTheme() {
    if (!nav) return;
    const y = nav.getBoundingClientRect().bottom;
    nav.classList.toggle('dark', darks.some(s => { const r = s.getBoundingClientRect(); return r.top <= y && r.bottom > y; }));
  }
  addEventListener('scroll', navTheme, { passive: true }); addEventListener('resize', navTheme); navTheme();

  /* reveal */
  if ('IntersectionObserver' in window && !reduce) {
    document.documentElement.classList.add('js');
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { rootMargin: '0px 0px -8% 0px' });
    $$('.rv').forEach(el => io.observe(el));
  }

  /* when-visible helper */
  window.SA.onView = (el, fn, threshold = 0.35) => {
    if (!el) return;
    if (!('IntersectionObserver' in window)) { fn(); return; }
    const io = new IntersectionObserver(es => { if (es[0].isIntersecting) { io.disconnect(); fn(); } }, { threshold });
    io.observe(el);
  };

  /* typewriter helper */
  window.SA.type = (el, text, speed = 34) => new Promise(res => {
    if (reduce) { el.textContent = text; res(); return; }
    el.textContent = ''; let i = 0;
    const t = setInterval(() => { el.textContent = text.slice(0, ++i); if (i >= text.length) { clearInterval(t); res(); } }, speed);
  });
})();
