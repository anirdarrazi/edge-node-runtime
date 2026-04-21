import { defineConfig } from "@playwright/test";

const channel =
  process.env.PLAYWRIGHT_CHANNEL ||
  (process.platform === "win32" ? "msedge" : undefined);

export default defineConfig({
  testDir: ".",
  testMatch: /.*\.smoke\.spec\.mjs/,
  fullyParallel: false,
  timeout: 30_000,
  expect: {
    timeout: 5_000,
  },
  reporter: "list",
  use: {
    browserName: "chromium",
    channel,
    headless: true,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    viewport: {
      width: 1440,
      height: 1100,
    },
  },
});
