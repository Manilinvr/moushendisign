// Рендер SVG из build/manifest.json в PNG через Chromium (Playwright).
//   node src/render.cjs
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const assets = JSON.parse(fs.readFileSync(path.join(root, 'build', 'manifest.json'), 'utf8'));

(async () => {
  const browser = await chromium.launch();
  for (const a of assets) {
    const page = await browser.newPage({
      viewport: { width: a.w, height: a.h },
      deviceScaleFactor: a.scale,
    });
    const svg = fs.readFileSync(path.join(root, a.svg), 'utf8');
    await page.setContent(`<!doctype html><html><body style="margin:0;background:transparent">${svg}</body></html>`);
    await page.screenshot({
      path: path.join(root, a.png),
      omitBackground: a.transparent,
      clip: { x: 0, y: 0, width: a.w, height: a.h },
    });
    await page.close();
    console.log(`${a.png}  ${Math.round(a.w * a.scale)}×${Math.round(a.h * a.scale)}`);
  }
  await browser.close();
})();
