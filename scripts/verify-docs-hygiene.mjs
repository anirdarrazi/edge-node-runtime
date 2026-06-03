#!/usr/bin/env node
import { runVerifyDocsHygiene } from '../../scripts/verify-docs-hygiene.mjs';
import { resolve } from 'node:path';

runVerifyDocsHygiene({
  repoName: 'edge-node-runtime',
  repoRoot: process.cwd(),
  policyPath: resolve(process.cwd(), '..', 'scripts', 'clean-room-policy.json'),
});
