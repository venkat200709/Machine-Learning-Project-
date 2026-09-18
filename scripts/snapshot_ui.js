/**
 * Render each dashboard view to a static HTML snapshot for visual review.
 *
 * jsdom boots the real page against the running API and executes app.js, so
 * every chart, table and animated value is baked into the DOM. The result is
 * plain HTML+CSS that a renderer (or a browser) can display without scripting.
 *
 *   node scripts/snapshot_ui.js http://127.0.0.1:8000 /tmp/snapshots
 */
const fs = require('fs');
const path = require('path');

const BASE = process.argv[2] || 'http://127.0.0.1:8000';
const OUT = process.argv[3] || path.join(__dirname, '..', 'reports', 'snapshots');
const ROOT = path.join(__dirname, '..');
const { JSDOM } = require('jsdom');

const VIEWS = ['overview', 'predict', 'analytics', 'model', 'map', 'batch', 'about'];

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const html = fs.readFileSync(path.join(ROOT, 'frontend', 'index.html'), 'utf8');
  const js = fs.readFileSync(path.join(ROOT, 'frontend', 'app.js'), 'utf8');
  const css = fs.readFileSync(path.join(ROOT, 'frontend', 'styles.css'), 'utf8');

  const dom = new JSDOM(html, { runScripts: 'outside-only', pretendToBeVisual: true, url: BASE });
  const { window } = dom;
  const doc = window.document;

  window.fetch = (url, opts) => fetch(new URL(url, BASE).href, opts);
  window.FormData = FormData;
  window.scrollTo = () => {};
  window.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
  window.IntersectionObserver = class { observe() {} unobserve() {} disconnect() {} };
  window.HTMLCanvasElement.prototype.getContext = () => ({
    clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, arc() {},
    fill() {}, setTransform() {}, createRadialGradient: () => ({ addColorStop() {} }),
  });

  window.eval(js);
  doc.dispatchEvent(new window.Event('DOMContentLoaded'));
  await new Promise(r => setTimeout(r, 6000));

  // Run one prediction so the result panel is populated in the snapshot.
  doc.querySelector('.nav-item[data-view="predict"]')
     .dispatchEvent(new window.Event('click', { bubbles: true }));
  doc.querySelector('#btnPredict').dispatchEvent(new window.Event('click', { bubbles: true }));
  await new Promise(r => setTimeout(r, 6000));

  for (const view of VIEWS) {
    doc.querySelector(`.nav-item[data-view="${view}"]`)
       .dispatchEvent(new window.Event('click', { bubbles: true }));
    await new Promise(r => setTimeout(r, 500));

    const clone = doc.documentElement.cloneNode(true);
    clone.querySelectorAll('script').forEach(s => s.remove());
    clone.querySelectorAll('link[rel="stylesheet"]').forEach(l => l.remove());
    clone.querySelector('#boot')?.remove();
    // Freeze reveal animations so nothing renders at opacity 0.
    clone.querySelectorAll('.reveal').forEach(el => el.classList.add('in'));
    clone.querySelector('#shell')?.classList.add('ready');

    const style = clone.ownerDocument.createElement('style');
    style.textContent = css + `
      /* snapshot overrides: no animation, static viewport */
      *{animation:none!important;transition:none!important;}
      .reveal{opacity:1!important;transform:none!important;}
      .shell{opacity:1!important;}
      body{width:1440px;}
      .aurora{position:absolute;height:2400px;}
    `;
    clone.querySelector('head').appendChild(style);

    const file = path.join(OUT, `${view}.html`);
    fs.writeFileSync(file, '<!DOCTYPE html>' + clone.outerHTML);
    console.log(`  wrote ${file}`);
  }
  console.log('Snapshots complete');
  process.exit(0);
})();
