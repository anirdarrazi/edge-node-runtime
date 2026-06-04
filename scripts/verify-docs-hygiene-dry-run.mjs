#!/usr/bin/env node
import { runVerifyDocsHygiene } from './lib/verify-docs-hygiene.mjs';

runVerifyDocsHygiene({
  repoName: 'edge-node-runtime',
  repoRoot: process.cwd(),
  dryRun: true,
});
