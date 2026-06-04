#!/usr/bin/env node
import { existsSync, statSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

import { isTextLikePath } from './text-utils.mjs';

const DEFAULT_MAX_BYTES = 8 * 1024 * 1024;

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

function parseArgs(argv) {
  const args = {
    repoRoot: process.cwd(),
    repoName: basename(process.cwd()),
    maxBytes: DEFAULT_MAX_BYTES,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if ((arg === '--repo' || arg === '-r') && index + 1 < argv.length) {
      args.repoName = argv[index + 1];
      index += 1;
      continue;
    }

    if (arg === '--max-bytes' && index + 1 < argv.length) {
      const parsed = Number.parseInt(argv[index + 1], 10);
      if (Number.isFinite(parsed) && parsed > 0) {
        args.maxBytes = parsed;
        index += 1;
        continue;
      }
    }

    if (arg === '--max-mb' && index + 1 < argv.length) {
      const parsed = Number.parseFloat(argv[index + 1]);
      if (Number.isFinite(parsed) && parsed > 0) {
        args.maxBytes = Math.round(parsed * 1024 * 1024);
        index += 1;
        continue;
      }
    }
  }

  return args;
}

function asMegabytes(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function runVerifyTrackedArtifactSize({
  repoRoot = process.cwd(),
  repoName = basename(repoRoot),
  maxBytes = DEFAULT_MAX_BYTES,
} = {}) {
  const files = getTrackedFiles(repoRoot);
  const violations = [];
  for (const file of files) {
    const absolutePath = resolve(repoRoot, file);
    if (!existsSync(absolutePath) || !isTextLikePath(file)) {
      continue;
    }

    const sizeBytes = statSync(absolutePath).size;
    if (sizeBytes > maxBytes) {
      violations.push({
        file,
        bytes: sizeBytes,
      });
    }
  }

  if (violations.length === 0) {
    console.log(
      `[tracked-artifact-size] No tracked text files exceed ${asMegabytes(maxBytes)} in ${repoName}.`,
    );
    return;
  }

  console.error(
    `[tracked-artifact-size] Large tracked text files detected for ${repoName} (threshold ${asMegabytes(maxBytes)}).`,
  );
  for (const violation of violations.sort((left, right) => right.bytes - left.bytes)) {
    console.error(`  - ${violation.file} (${asMegabytes(violation.bytes)})`);
  }
  throw new Error('[tracked-artifact-size] Tracked artifact-size guard failed.');
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const args = parseArgs(process.argv.slice(2));
    runVerifyTrackedArtifactSize({
      repoName: args.repoName,
      repoRoot: process.cwd(),
      maxBytes: args.maxBytes,
    });
    process.exit(0);
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
}
