#!/usr/bin/env node
import { runVerifyTrackedArtifactSize } from './lib/verify-tracked-artifact-size.mjs';

runVerifyTrackedArtifactSize({
  repoName: 'edge-node-runtime',
  repoRoot: process.cwd(),
  maxBytes: 8 * 1024 * 1024,
});
