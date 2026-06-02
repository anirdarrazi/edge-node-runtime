import { resolve } from 'node:path';
import { runCleanRepo } from '../../scripts/clean-repo.mjs';

runCleanRepo({
  repoName: 'edge-node-runtime',
  repoRoot: process.cwd(),
  dryRun: process.argv.includes('--dry-run'),
  policyPath: resolve(process.cwd(), '..', 'scripts', 'clean-room-policy.json'),
});
