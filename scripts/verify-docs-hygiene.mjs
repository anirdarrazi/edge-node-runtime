#!/usr/bin/env node
import { runVerifyDocsHygiene } from '../../scripts/verify-docs-hygiene.mjs';

runVerifyDocsHygiene({
  repoName: 'edge-node-runtime',
  repoRoot: process.cwd(),
});
