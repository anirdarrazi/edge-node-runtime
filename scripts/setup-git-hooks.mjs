#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
execFileSync('git', ['-C', repoRoot, 'config', 'core.hooksPath', '.githooks'], {
  stdio: 'inherit',
});

console.log('[hooks] core.hooksPath set to .githooks');
