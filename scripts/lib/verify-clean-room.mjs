#!/usr/bin/env node
import { existsSync, readFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const DEFAULT_POLICY = resolve(dirname(fileURLToPath(import.meta.url)), 'clean-room-policy.json');

function parseArgs(argv) {
  const args = { repoName: basename(process.cwd()), strict: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--strict') {
      args.strict = true;
      continue;
    }
    if (arg === '--repo' && i + 1 < argv.length) {
      args.repoName = argv[i + 1];
      i += 1;
      continue;
    }
    if (arg.startsWith('--repo=')) {
      args.repoName = arg.slice('--repo='.length);
    }
  }
  return args;
}

function normalizePattern(pattern) {
  return pattern.trim();
}

function readPatterns(root) {
  const ignorePath = resolve(root, '.gitignore');
  if (!existsSync(ignorePath)) {
    throw new Error(`Missing .gitignore file: ${ignorePath}`);
  }

  return readFileSync(ignorePath, 'utf8')
    .split('\n')
    .map((line) => normalizePattern(line).replace(/\r$/, ''))
    .filter((line) => line && !line.startsWith('#'));
}

function hasPattern(patterns, required) {
  if (patterns.has(required)) {
    return true;
  }
  if (required.endsWith('/')) {
    return patterns.has(required.replace(/\/$/, ''));
  }

  return patterns.has(`${required}/`);
}

export function runVerifyCleanRoom(args) {
  const { repoName, strict = false, policyPath = DEFAULT_POLICY, repoRoot = process.cwd() } = args;

  const policy = JSON.parse(readFileSync(policyPath, 'utf8'));
  const knownRepo = repoName in policy.repos ? repoName : basename(repoRoot);
  const requiredPatterns = [
    ...policy.base,
    ...(policy.repos[knownRepo] ?? policy.repos['edge-node-runtime'] ?? []),
  ];

  const ignoreLines = new Set(
    readPatterns(repoRoot).map((line) => (strict ? normalizePattern(line) : line)),
  );

  const missingPatterns = requiredPatterns.filter((pattern) => !hasPattern(ignoreLines, pattern));

  if (missingPatterns.length > 0) {
    console.error(`[clean-room] Missing required ignore patterns for ${knownRepo}`);
    for (const pattern of missingPatterns) {
      console.error(`  - ${pattern}`);
    }
    console.error('[clean-room] Add these entries to .gitignore before proceeding.');
    process.exit(1);
  }

  console.log(`[clean-room] .gitignore includes required ignore patterns for ${knownRepo}.`);
}

export function main() {
  const args = parseArgs(process.argv.slice(2));
  runVerifyCleanRoom({
    repoName: args.repoName,
    strict: args.strict,
    repoRoot: process.cwd(),
    policyPath: DEFAULT_POLICY,
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
