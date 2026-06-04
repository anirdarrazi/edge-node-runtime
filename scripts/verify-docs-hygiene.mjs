#!/usr/bin/env node
import { runVerifyDocsHygiene } from './lib/verify-docs-hygiene.mjs';
import { resolve } from 'node:path';

runVerifyDocsHygiene({
  repoName: 'edge-node-runtime',
  repoRoot: process.cwd(),
  policyPath: resolve(process.cwd(), 'scripts', 'lib', 'clean-room-policy.json'),
});
