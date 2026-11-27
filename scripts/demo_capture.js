// F8.4 demo capture: screenshots of the live dashboard and scorer while a replay runs.
// Runs inside zenika/alpine-chrome:with-puppeteer on the compose network (see scripts/record_demo.sh).
// Writes JPEG frames + manifest.jsonl ({seg, file, t}) into /out.
const fs = require('fs');
const puppeteer = require('puppeteer');

const OUT = '/out';
const DASH = process.env.DASH_URL || 'http://dashboard:8501';
const SCORER = process.env.SCORER_URL || 'http://scorer:8000';
const PUBLIC = process.env.PUBLIC_URL || 'https://aiml01-fraud-dashboard.streamlit.app';
const FPS = 2;
const t0 = Date.now();
let n = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const log = m => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s] ${m}`);

async function shot(page, seg) {
  const file = `f${String(n++).padStart(5, '0')}.jpg`;
  await page.screenshot({path: `${OUT}/${file}`, type: 'jpeg', quality: 85});
  fs.appendFileSync(`${OUT}/manifest.jsonl`,
    JSON.stringify({seg, file, t: (Date.now() - t0) / 1000}) + '\n');
}

async function film(page, seg, seconds) {
  const end = Date.now() + seconds * 1000;
  while (Date.now() < end) {
    const next = Date.now() + 1000 / FPS;
    await shot(page, seg);
    const wait = next - Date.now();
    if (wait > 0) await sleep(wait);
  }
}

async function clickText(page, selector, text) {
  const els = await page.$$(selector);
  for (const e of els) {
    const t = (await page.evaluate(x => x.innerText, e)).trim();
    if ((t === text || t.startsWith(text)) && await e.boundingBox()) {
      await e.click();
      return true;
    }
  }
  log(`not found: ${selector} "${text}"`);
  return false;
}

// Streamlit scrolls an inner container, so scroll with the mouse wheel over the content.
async function wheel(page, dy) {
  await page.mouse.move(720, 500);
  await page.mouse.wheel({deltaY: dy});
  await sleep(700);
}

async function tab(page, name) {
  await wheel(page, -5000);
  await clickText(page, 'button[role="tab"]', name);
  await sleep(2500);
}

async function pickRing(page, index) {
  // The "Ring network" selectbox (other tabs have hidden selectboxes too).
  const ok = await page.evaluate(() => {
    const box = [...document.querySelectorAll('[data-testid="stSelectbox"]')]
      .find(b => b.innerText.includes('Ring network') && b.offsetParent !== null);
    if (!box) return false;
    box.querySelector('div[data-baseweb="select"]').setAttribute('data-demo', 'ring');
    return true;
  });
  if (!ok) { log('ring selectbox not found'); return; }
  await page.click('[data-demo="ring"]');
  await sleep(600);
  for (let i = 0; i < index; i++) await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Enter');
  await sleep(3000);
}

(async () => {
  fs.writeFileSync(`${OUT}/manifest.jsonl`, '');
  const b = await puppeteer.launch({args: ['--no-sandbox', '--disable-dev-shm-usage'], headless: 'new'});
  const p = await b.newPage();
  await p.setViewport({width: 1440, height: 900});

  // Architecture figure as an image for the video.
  await p.goto('file:///repo/docs/figures/architecture.svg');
  await sleep(500);
  await p.screenshot({path: `${OUT}/architecture.png`, fullPage: true});

  log('dashboard');
  await p.goto(DASH, {waitUntil: 'networkidle2', timeout: 120000});
  await p.waitForSelector('button[role="tab"]', {timeout: 120000});
  await sleep(2000);
  await film(p, 'overview_live', Number(process.env.OVERVIEW_S || 50));

  log('alert queue');
  await clickText(p, 'button', 'Refresh queue');
  await sleep(2500);
  await wheel(p, 560);
  await film(p, 'alerts', 6);
  await clickText(p, 'button', 'Confirm fraud');
  await sleep(1500);
  await film(p, 'alerts_confirm', 4);
  await clickText(p, 'button', 'Dismiss');
  await sleep(1500);
  await film(p, 'alerts_dismiss', 4);

  log('rings');
  await tab(p, 'Fraud rings');
  await wheel(p, 520);
  await film(p, 'rings_table', 6);
  await pickRing(p, 2);
  await wheel(p, 700);
  await film(p, 'rings_graph', 8);

  log('cases + model');
  await tab(p, 'Cases');
  await wheel(p, 520);
  await film(p, 'cases', 5);
  await tab(p, 'Model');
  await wheel(p, 520);
  await film(p, 'model', 5);

  // Wait for the replay (incl. drift) to finish, then the final overview.
  const finalAt = Number(process.env.FINAL_AT_S || 140);
  while ((Date.now() - t0) / 1000 < finalAt) await sleep(1000);
  log('final overview');
  await p.goto(DASH, {waitUntil: 'networkidle2'});
  await p.waitForSelector('button[role="tab"]');
  await sleep(4000);
  await film(p, 'overview_final', 8);

  log('scorer docs');
  await p.goto(`${SCORER}/docs`, {waitUntil: 'networkidle2'});
  await sleep(1500);
  await film(p, 'docs', 3);
  await clickText(p, '.opblock-summary-path, .opblock-summary', '/score');
  await sleep(1500);
  await film(p, 'docs_score', 5);

  log('public app');
  try {
    await p.goto(PUBLIC, {waitUntil: 'networkidle2', timeout: 90000});
    for (let i = 0; i < 60; i++) {
      const ok = await p.evaluate(() => !!document.querySelector('iframe') ||
        !!document.querySelector('button[role="tab"]'));
      if (ok) break;
      await sleep(2000);
    }
    await sleep(8000);
    await film(p, 'public', 4);
  } catch (e) { log('public app failed: ' + e.message); }

  await b.close();
  log('done');
})().catch(e => { console.error(e); process.exit(1); });
