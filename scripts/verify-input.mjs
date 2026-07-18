// Definitive input check via the window.__app debug handle.
import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
await page.goto("http://localhost:5173/generals/");
await page.waitForFunction(() => "__app" in window, { timeout: 15000 });
await page.waitForFunction(
  () => /ready/.test(document.querySelector("#status")?.textContent ?? ""),
  { timeout: 30000 },
);
await page.waitForTimeout(2500); // let the general accumulate armies

// locate the P0 general precisely from the true state
const gen = await page.evaluate(() => {
  const app = window.__app;
  const st = app.session.state;
  const gp = st.generalPositions;
  return { row: gp[0], col: gp[1], armies: st.armies[gp[0] * 10 + gp[1]] };
});
console.log("general:", JSON.stringify(gen));

const box = await page.locator("#board").boundingBox();
await page.mouse.click(
  box.x + ((gen.col + 0.5) / 10) * box.width,
  box.y + ((gen.row + 0.5) / 10) * box.height,
);
await page.waitForTimeout(100);
const anchor = await page.evaluate(() => window.__app.input?.anchor);
console.log("anchor after click:", anchor, "(expected", gen.row * 10 + gen.col + ")");

for (const k of ["ArrowRight", "ArrowRight", "ArrowDown", "ArrowDown", "ArrowLeft", "ArrowUp"]) {
  await page.keyboard.press(k);
  await page.waitForTimeout(120);
}
const qlen = await page.evaluate(() => window.__app.session.queueOf(0).length);
console.log("queued moves:", qlen);
await page.waitForTimeout(4000);
const after = await page.evaluate(() => {
  const app = window.__app;
  const t = app.session.totals();
  return { land0: t.land[0], turn: app.session.turn };
});
console.log("after:", JSON.stringify(after));
console.log(after.land0 > 1 && qlen > 0 ? "INPUT OK" : "INPUT BROKEN");
await browser.close();
