#!/usr/bin/env node
import { runVerifyTrackedLineEndings } from './lib/verify-tracked-line-endings.mjs';

runVerifyTrackedLineEndings({
  repoName: 'edge-node-runtime',
  repoRoot: process.cwd(),
});
