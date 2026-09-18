/**
 * Headless smoke test for the RiskRadar frontend.
 *
 * Loads index.html into jsdom, stubs fetch against the live API, executes
 * app.js and asserts that every view actually renders. Catches the class of
 * bug a Python test suite never sees: a typo in a selector, a chart helper
 * that throws, a null dereference during boot.
 *
 *   node tests/smoke_frontend.js [http://127.0.0.1:8000]
 *
 * Requires jsdom (`npm install jsdom`). Skipped automatically if absent.
 */
const fs = require('fs');
const path = require('path');

const BASE = process.argv[2] || 'http://127.0.0.1:8000';
const ROOT = path.join(__dirname, '..');

let JSDOM;
try { ({ JSDOM } = require('jsdom')); }
catch { console.log('SKIP: jsdom not installed (npm install jsdom)'); process.exit(0); }

const failures = [];
const check = (name, cond, extra = '') => {
  console.log(`${cond ? '  ok  ' : '  FAIL'} ${name}${extra ? ' — ' + extra : ''}`);
  if (!cond) failures.push(name);
};

(async () => {
  // The dashboard is a single self-contained file: pull the inlined
  // controller out of it rather than loading a separate app.js.
  const html = fs.readFileSync(path.join(ROOT, 'frontend', 'index.html'), 'utf8');
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
  if (scripts.length !== 1) {
    console.log(`FAIL: expected exactly 1 inline <script>, found ${scripts.length}`);
    process.exit(1);
  }
  const js = scripts[0];

  const dom = new JSDOM(html, {
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    url: BASE,
  });
  const { window } = dom;

  const errors = [];
  window.addEventListener('error', e => errors.push(String(e.error || e.message)));

  // Real network calls against the running API.
  window.fetch = (url, opts) => fetch(new URL(url, BASE).href, opts);
  window.FormData = FormData;
  window.Blob = Blob;
  window.URL.createObjectURL = () => 'blob:stub';
  window.URL.revokeObjectURL = () => {};
  // Full-enough 2D context stub that the particle field and the risk map
  // both execute for real instead of silently no-oping.
  window.HTMLCanvasElement.prototype.getContext = () => ({
    canvas: { width: 1440, height: 900 },
    clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
    arc() {}, fill() {}, fillRect() {}, setTransform() {}, save() {}, restore() {},
    translate() {}, rotate() {}, scale() {}, closePath() {}, measureText: () => ({ width: 10 }),
    fillText() {}, strokeText() {}, drawImage() {}, clip() {}, rect() {},
    createRadialGradient: () => ({ addColorStop() {} }),
    createLinearGradient: () => ({ addColorStop() {} }),
    // The route planner dashes the comparison path and shadows the glow, so
    // the stub has to cover those too — otherwise the draw call throws and
    // the failure looks like a product bug rather than a missing stub.
    setLineDash() {}, getLineDash: () => [], ellipse() {}, quadraticCurveTo() {},
    bezierCurveTo() {}, arcTo() {}, createPattern: () => null,
    strokeRect() {}, clearRect() {}, roundRect() {}, isPointInPath: () => false,
    globalCompositeOperation: 'source-over', globalAlpha: 1,
    fillStyle: '', strokeStyle: '', lineWidth: 1, font: '',
    textAlign: 'start', textBaseline: 'alphabetic',
    lineJoin: 'miter', lineCap: 'butt',
    shadowColor: 'transparent', shadowBlur: 0,
  });
  window.scrollTo = () => {};
  window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  window.IntersectionObserver = class {
    observe() {} unobserve() {} disconnect() {}
  };

  try {
    window.eval(js);
    window.document.dispatchEvent(new window.Event('DOMContentLoaded'));
  } catch (e) {
    check('app.js executes', false, e.message);
    process.exit(1);
  }

  // Give boot() time to finish its network round-trips.
  await new Promise(r => setTimeout(r, 6000));

  const $ = s => window.document.querySelector(s);
  const $$ = s => [...window.document.querySelectorAll(s)];

  console.log('\nSelf-contained file');
  check('stylesheet is inlined', /<style>[\s\S]{5000,}<\/style>/.test(html));
  check('controller is inlined', js.length > 20000, `${js.length} chars`);
  check('no external asset requests', !/(href|src)="[^"]*\.(css|js)[?"]/.test(html));

  // ── Desktop layout ────────────────────────────────────────────────
  // A whole block of flagship CSS was once accidentally nested inside
  // `@media (max-width:940px)`. Every rule parsed, the file was valid, and the
  // dashboard rendered completely unstyled on any real monitor — unconstrained
  // SVGs, white canvases, labels running into values. Asserting on *computed*
  // style at desktop width is the only thing that catches that class of bug.
  console.log('\nDesktop layout (guards against mis-nested CSS)');
  {
    const cs = el => (el ? window.getComputedStyle(el) : null);
    const layout = [
      ['.panel is padded', '#view-route .panel', s => s.padding && s.padding !== '0px'],
      ['.route-grid is two columns', '.route-grid',
        s => s.display === 'grid' && /340px/.test(s.gridTemplateColumns)],
      ['.compare-cards is three columns', '.compare-cards',
        s => /190px/.test(s.gridTemplateColumns)],
      ['.ops-top is two columns', '.ops-top', s => /320px/.test(s.gridTemplateColumns)],
      ['.unc-grid is two columns', '.unc-grid', s => s.display === 'grid'],
      ['.set-band is laid out', '.set-band', s => s.display === 'flex'],
      ['metric labels stack above values', '.cmp-metrics label', s => s.display === 'block'],
      ['throughput canvas has height', '.spark', s => s.height === '158px'],
      ['health dial is centred', '.health-dial-svg', s => s.margin.includes('auto')],
    ];
    for (const [name, sel, ok] of layout) {
      const el = $(sel);
      check(name, !!el && ok(cs(el)), el ? '' : `${sel} missing`);
    }
    // Chart text must be filled, never stroked — a stroked glyph reads as blurry.
    check('svg text is not stroked', /svg\s+text\s*\{[^}]*stroke:\s*none/.test(html));

    // ── Style collisions ────────────────────────────────────────────
    // The flagship views were added to a stylesheet that already had a
    // `.map-legend` (absolutely positioned) and its own `.panel`, `.btn` and
    // `.data-table`. Reusing those names silently re-skinned existing views
    // and floated the new legend over the summary tiles. A duplicate base
    // definition is the tell, so assert on it directly.
    const css = html.match(/<style>([\s\S]*?)<\/style>/)[1];
    const defs = sel => (css.match(new RegExp(`^\\${sel}\\{`, 'gm')) || []).length;
    for (const sel of ['.panel', '.btn', '.data-table', '.map-legend',
                       '.gauge', '.gauge-wrap', '.kpi-row', '.prob-bars',
                       '.spinner', '.chip', '.tag']) {
      check(`${sel} defined exactly once`, defs(sel) === 1, `${defs(sel)} definitions`);
    }
    // The new legend must sit in normal flow, not inherit absolute positioning.
    const legend = $('#view-route .risk-legend');
    check('risk legend is in normal flow',
      !!legend && cs(legend).position === 'static',
      legend ? cs(legend).position : 'missing');

    // The health dial must not pick up the Assess Area gauge's rotation or its
    // fixed 168px square — that is what bent the semicircle into a "C".
    const dialSvg = $('.health-dial-svg');
    check('health dial is not rotated',
      !!dialSvg && !/rotate/.test(cs(dialSvg).transform || ''),
      dialSvg ? cs(dialSvg).transform : 'missing');
    check('health dial scales with its container',
      !!dialSvg && cs(dialSvg).width === '100%', dialSvg ? cs(dialSvg).width : '');
    // The readout is <text> inside the same viewBox, so it cannot drift.
    check('health readout lives inside the svg',
      !!$('svg.health-dial-svg text#healthScore'));
    check('health grade lives inside the svg',
      !!$('svg.health-dial-svg text#healthGrade'));
  }

  console.log('\nPlain-language map affordances');
  check('risk map has an explainer', !!$('.map-explainer'));
  check('risk map has compass markers', $$('.compass').length === 4);
  check('risk map has a summary strip', !!$('#mapSummary'));

  console.log('\nAttribution');
  check('authors credited', /N\.\s*Venkatesan/.test(html) && /Neethivendhan\s*T\./.test(html));

  console.log('\nBoot');
  check('no uncaught runtime errors', errors.length === 0, errors.join(' | '));
  check('boot overlay dismissed', $('#boot').classList.contains('done'));
  check('shell revealed', $('#shell').classList.contains('ready'));

  console.log('\nForm');
  check('all 29 inputs rendered', $$('[data-key]').length === 29,
    `found ${$$('[data-key]').length}`);
  check('sweep field options populated', $('#sweepField').children.length > 4);
  check('preset applied (hour set)', $('#f_Hour').value !== '');

  console.log('\nOverview');
  check('accuracy counter populated', /\d/.test($('#kpiAccuracy').textContent));
  check('leaderboard chart drawn', $('#leaderChart').querySelectorAll('rect').length > 3);
  check('importance chart drawn', $('#importanceChart').querySelectorAll('rect').length > 3);
  check('hourly chart drawn', $('#hourChart').querySelectorAll('path').length >= 2);
  check('health list populated', $('#healthList').children.length === 6);
  check('sidebar accuracy set', $('#sideAccuracy').textContent.includes('%'));

  console.log('\nView switching');
  for (const view of ['analytics', 'model', 'map', 'batch', 'about', 'predict']) {
    $(`.nav-item[data-view="${view}"]`).dispatchEvent(new window.Event('click', { bubbles: true }));
    await new Promise(r => setTimeout(r, 320));
    check(`${view} view активates`.replace('активates', 'activates'),
      $(`#view-${view}`).classList.contains('active'));
  }

  console.log('\nAnalytics render');
  check('hourly risk chart', $('#aHour').querySelectorAll('path').length >= 2);
  check('crime-by-hour columns', $('#aCrimeHour').querySelectorAll('rect').length > 20);
  check('weather chart', $('#aWeather').querySelectorAll('rect').length >= 4);
  check('crime mix donut', $('#aCrimeMix').querySelectorAll('circle').length >= 4);
  check('persistence chart', $('#aPersist').querySelectorAll('rect').length >= 9);

  console.log('\nModel card render');
  check('confusion matrix cells', $('#confusion').querySelectorAll('rect').length === 9);
  check('per-class table rows', $('#perClassTable').querySelectorAll('tbody tr').length === 3);
  check('leaderboard table rows', $('#leaderTable').querySelectorAll('tbody tr').length >= 5);
  check('winner row highlighted', $('#leaderTable').querySelectorAll('tr.best').length === 1);
  check('spec list populated', $('#specList').children.length >= 10);

  console.log('\nPrediction round-trip');
  $('.nav-item[data-view="predict"]').dispatchEvent(new window.Event('click', { bubbles: true }));
  $('#btnPredict').dispatchEvent(new window.Event('click', { bubbles: true }));
  await new Promise(r => setTimeout(r, 5000));
  check('result panel shown', $('#resultBody').hidden === false);
  check('risk pill filled', /risk/i.test($('#riskPill').textContent));
  check('probability bars', $('#probBars').children.length === 3);
  check('SHAP waterfall rows', $('#waterfall').children.length >= 5);
  check('narrative written', $('#narrative').textContent.length > 10);
  check('sweep chart drawn', $('#sweepChart').querySelectorAll('path').length >= 2);
  check('gauge arc animated', $('#gaugeFill').style.strokeDashoffset !== '');

  // ── Flagship views ────────────────────────────────────────────────
  // These drive the v2 platform API, so a failure here means either the
  // frontend controller or the backend route is broken — both worth catching
  // before anyone opens a browser.
  const show = async (view, ms = 900) => {
    $(`.nav-item[data-view="${view}"]`).dispatchEvent(
      new window.Event('click', { bubbles: true }));
    await new Promise(r => setTimeout(r, ms));
  };

  console.log('\nFlagship navigation');
  for (const view of ['route', 'optimise', 'uncertainty', 'ops', 'governance']) {
    await show(view, 400);
    check(`${view} view activates`, $(`#view-${view}`).classList.contains('active'));
  }

  console.log('\nUncertainty view');
  await show('uncertainty', 3000);
  check('conformal method badge set', $('#confMethodBadge').textContent.trim() !== '—');
  check('coverage ladder rendered', $('#uncLadder').querySelectorAll('.ladder-row').length >= 5);
  check('probability bars rendered', $('#uncProbs').querySelectorAll('.bar-row').length === 3);
  check('calibration evidence shown', $('#calGrid').querySelectorAll('.cal-cell').length >= 4);
  check('a prediction set is highlighted',
    $$('#setStage .set-band.in').length >= 1 || /no band clears/i.test($('#setVerdict').textContent));
  {
    // Moving the slider must actually change the answer, not just the label.
    const slider = $('#covSlider');
    const before = $('#setVerdict').textContent;
    slider.value = '99';
    slider.dispatchEvent(new window.Event('input', { bubbles: true }));
    await new Promise(r => setTimeout(r, 250));
    check('coverage slider updates the readout', $('#covValue').textContent === '99');
    check('coverage slider re-evaluates the set',
      $('#setVerdict').textContent !== before || /99%/.test($('#setVerdict').textContent));
  }

  console.log('\nRoute planner');
  await show('route', 6000);
  // The controller's state is const-scoped inside the module, so assert on
  // what the DOM can actually observe: the canvas was sized and the loading
  // overlay cleared, which only happens once the grid has been scored.
  check('risk grid loaded',
    $('#routeCanvas').width > 0 && !$('#routeLoader').classList.contains('on'));
  {
    $('#routeDemo').dispatchEvent(new window.Event('click', { bubbles: true }));
    await new Promise(r => setTimeout(r, 9000));
    check('route computed', $('#routeResults').hidden === false);
    check('safest distance reported', /km/.test($('#cmpSafeDist').textContent));
    check('shortest distance reported', /km/.test($('#cmpFastDist').textContent));
    check('trade-off ring animated', $('#vsArc').style.strokeDashoffset !== '');
    check('recommendation written', $('#routeVerdict').textContent.length > 20);
    check('exposure profile drawn', $('#routeProfile').querySelectorAll('path').length >= 2);
  }

  console.log('\nCity planner');
  await show('optimise', 1500);
  check('intervention catalogue loaded', $('#optLevers').querySelectorAll('[data-lever]').length >= 4);
  {
    $('#optRun').dispatchEvent(new window.Event('click', { bubbles: true }));
    await new Promise(r => setTimeout(r, 20000));
    check('optimiser returned a plan', $('#optResults').hidden === false);
    check('KPI tiles rendered', $('#optKpis').querySelectorAll('.kpi-tile').length >= 4);
    check('spend breakdown drawn', $('#optSpend').querySelectorAll('.bar-row').length >= 1);
    check('before/after bands drawn', $('#optShift').querySelectorAll('.shift-seg').length === 6);
    check('allocation table populated', $('#optPlan').querySelectorAll('tbody tr').length >= 1);
  }

  console.log('\nCommand centre');
  await show('ops', 4000);
  check('health gauge animated', $('#healthGaugeFill').style.strokeDashoffset !== '');
  check('health score populated', /\d/.test($('#healthScore').textContent));
  check('status tiles rendered', $('#opsTiles').querySelectorAll('.stat-tile').length >= 6);
  check('drift panel populated', $('#driftBody').innerHTML.length > 60);
  check('live feed rendered', $('#feedTable').querySelectorAll('tr').length >= 1);

  console.log('\nGovernance');
  await show('governance', 800);
  {
    $('#fairRun').dispatchEvent(new window.Event('click', { bubbles: true }));
    await new Promise(r => setTimeout(r, 12000));
    check('fairness audit ran', $('#fairBody').querySelectorAll('.fair-dim').length >= 3);
    check('fairness verdict shown', /pass|review|fail/i.test($('#govVerdict').textContent));
    check('fairness metrics rendered', $('#fairBody').querySelectorAll('.fm').length >= 10);

    $('#govTabs').querySelector('[data-gov="registry"]')
      .dispatchEvent(new window.Event('click', { bubbles: true }));
    await new Promise(r => setTimeout(r, 1800));
    check('registry listed', $('#regBody').querySelectorAll('.reg-item').length >= 1);
    check('champion marked', $('#regBody').querySelectorAll('.stage-pill.champion').length === 1);

    $('#govTabs').querySelector('[data-gov="readiness"]')
      .dispatchEvent(new window.Event('click', { bubbles: true }));
    await new Promise(r => setTimeout(r, 1800));
    check('readiness audit rendered', $('#readyBody').innerHTML.length > 60);
    check('readiness score shown', /\d/.test($('#readyScore').textContent));
  }

  console.log('\nIntegrity');
  check('no duplicate element ids', (() => {
    const seen = new Set(), dupes = [];
    $$('[id]').forEach(el => { if (seen.has(el.id)) dupes.push(el.id); seen.add(el.id); });
    return dupes.length === 0;
  })());
  check('no uncaught errors after full tour', errors.length === 0, errors.join(' | '));

  console.log(`\n${failures.length ? `FAILED (${failures.length}): ${failures.join(', ')}` : 'ALL FRONTEND CHECKS PASSED'}`);
  process.exit(failures.length ? 1 : 0);
})();

