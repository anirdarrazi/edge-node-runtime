#!/usr/bin/env node
import { existsSync, statSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

import { isBinaryPath, isTextLikePath, hasCrLfLineEndings } from './text-utils.mjs';

function getTrackedFiles(repoRoot) {
  const output = execFileSync('git', ['ls-files', '-z'], {
    cwd: repoRoot,
    encoding: 'utf8',
  });

  return output
    .split('\0')
    .filter(Boolean)
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function parseDryRunArgs(argv) {
  const args = {
    repoName: basename(process.cwd()),
    includeBinary: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if ((arg === '--repo' || arg === '-r') && index + 1 < argv.length) {
      args.repoName = argv[index + 1];
      index += 1;
      continue;
    }

    if (arg === '--include-binary') {
      args.includeBinary = true;
      continue;
    }
  }

  return args;
}

export function runVerifyTrackedLineEndings({
  repoRoot = process.cwd(),
  repoName = basename(repoRoot),
  includeBinary = false,
} = {}) {
  const files = getTrackedFiles(repoRoot);
  const violations = [];

  for (const file of files) {
    const absolutePath = resolve(repoRoot, file);
    if (!existsSync(absolutePath)) {
      continue;
    }

    if (!isTextLikePath(file)) {
      if (!includeBinary) {
        continue;
      }
    }

    if (!includeBinary && isBinaryPath(file)) {
      continue;
    }

    const size = statSync(absolutePath).size;
    if (size === 0) {
      continue;
    }

    if (!hasCrLfLineEndings(absolutePath)) {
      continue;
    }

    violations.push(file);
  }

  if (violations.length === 0) {
    console.log(`[tracked-line-endings] Tracked text files in ${repoName} all use LF line endings.`);
    return;
  }

  console.error(`[tracked-line-endings] CRLF was detected in tracked text files for ${repoName}:`);
  for (const file of violations) {
    console.error(`  - ${file}`);
  }
  throw new Error('[tracked-line-endings] CRLF policy violation in tracked files.');
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const args = parseDryRunArgs(process.argv.slice(2));
    runVerifyTrackedLineEndings({
      repoName: args.repoName,
      repoRoot: process.cwd(),
      includeBinary: args.includeBinary,
    });
    process.exit(0);
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
}
