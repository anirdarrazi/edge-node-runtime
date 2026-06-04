import { resolve } from 'node:path';
import { runCleanRepo } from './lib/clean-repo.mjs';

runCleanRepo({
  repoName: 'edge-node-runtime',
  repoRoot: process.cwd(),
  dryRun: process.argv.includes('--dry-run'),
  policyPath: resolve(process.cwd(), 'scripts', 'lib', 'clean-room-policy.json'),
});
