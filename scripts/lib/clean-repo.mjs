#!/usr/bin/env node
import {
  existsSync,
  readFileSync,
  readdirSync,
  rmSync,
} from 'node:fs';
import { basename, relative, resolve } from 'node:path';

const DEFAULT_POLICY_PATH = resolve(process.cwd(), 'scripts', 'clean-room-policy.json');

function parseArgs(argv) {
  const args = {
    repoName: basename(process.cwd()),
    dryRun: false,
    policyPath: DEFAULT_POLICY_PATH,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === '--dry-run') {
      args.dryRun = true;
      continue;
    }

    if ((arg === '--repo' || arg === '-r') && index + 1 < argv.length) {
      args.repoName = argv[index + 1];
      index += 1;
      continue;
    }

    if (arg.startsWith('--repo=')) {
      args.repoName = arg.slice('--repo='.length);
      continue;
    }

    if ((arg === '--policy' || arg === '-p') && index + 1 < argv.length) {
      args.policyPath = argv[index + 1];
      index += 1;
      continue;
    }

    if (arg.startsWith('--policy=')) {
      args.policyPath = arg.slice('--policy='.length);
      continue;
    }
  }

  return args;
}

function normalizePath(value) {
  return value.replace(/\\/g, '/');
}

function escapeRegexChar(value) {
  return value.replace(/[.+^${}()|[\]\\]/g, '\\$&');
}

function wildcardToRegexSource(pattern) {
  let regex = '';
  const normalized = normalizePath(pattern);

  for (let index = 0; index < normalized.length; index += 1) {
    const char = normalized[index];

    if (char === '*' && normalized[index + 1] === '*') {
      regex += '.*';
      index += 1;
      continue;
    }

    if (char === '*') {
      regex += '[^/]*';
      continue;
    }

    if (char === '?') {
      regex += '[^/]';
      continue;
    }

    regex += escapeRegexChar(char);
  }

  return regex;
}

function wildcardMatch(pattern, value) {
  const regex = new RegExp(`^${wildcardToRegexSource(pattern)}$`);
  return regex.test(normalizePath(value));
}

function matchesPattern(pattern, relativePath, isDirectory) {
  const rawPattern = String(pattern || '').trim();
  if (!rawPattern || rawPattern.startsWith('#')) {
    return false;
  }

  const normalizedPattern = normalizePath(rawPattern);
  const normalizedPath = normalizePath(relativePath);

  if (normalizedPattern.endsWith('/')) {
    const directoryPattern = normalizedPattern.slice(0, -1);
    const regex = new RegExp(`^${wildcardToRegexSource(directoryPattern)}(?:/.*)?$`);
    return regex.test(normalizedPath);
  }

  if (!normalizedPattern.includes('/')) {
    const baseName = normalizedPath.split('/').at(-1);
    if (isDirectory && normalizedPattern === baseName) {
      return true;
    }
    return wildcardMatch(normalizedPattern, baseName);
  }

  return wildcardMatch(normalizedPattern, normalizedPath);
}

function loadPatterns(policyPath, repoName) {
  const policy = JSON.parse(readFileSync(policyPath, 'utf8'));
  const basePatterns = Array.isArray(policy.base) ? policy.base : [];
  const repoPatterns = Array.isArray(policy.repos?.[repoName]) ? policy.repos[repoName] : [];
  const fallbackPatterns = Array.isArray(policy.repos?.['edge-node-runtime'])
    ? policy.repos['edge-node-runtime']
    : [];

  return [...new Set([...basePatterns, ...repoPatterns, ...fallbackPatterns])]
    .map((pattern) => normalizePath(String(pattern || '').trim()))
    .filter(Boolean);
}

function collectCleanCandidates(rootDir, patterns) {
  const stack = [{ absolutePath: rootDir, relativePath: '' }];
  const candidates = new Set();

  while (stack.length > 0) {
    const current = stack.pop();
    const entries = readdirSync(current.absolutePath, { withFileTypes: true, recursive: false });

    for (const entry of entries) {
      if (entry.name === '.git') {
        continue;
      }

      const childRelative = current.relativePath
        ? `${current.relativePath}/${entry.name}`
        : entry.name;
      const childAbsolute = resolve(current.absolutePath, entry.name);
      const isDirectory = entry.isDirectory();

      const shouldDelete = patterns.some((pattern) => matchesPattern(pattern, childRelative, isDirectory));
      if (shouldDelete) {
        candidates.add(childAbsolute);
        continue;
      }

      if (isDirectory) {
        stack.push({ absolutePath: childAbsolute, relativePath: childRelative });
      }
    }
  }

  return [...candidates].sort((left, right) => right.length - left.length);
}

export function runCleanRepo({
  repoName = basename(process.cwd()),
  repoRoot = process.cwd(),
  dryRun = false,
  policyPath = DEFAULT_POLICY_PATH,
} = {}) {
  const rootDir = resolve(repoRoot);
  const gitignorePath = resolve(rootDir, '.gitignore');

  if (!existsSync(gitignorePath)) {
    throw new Error(`[clean:repo] Missing .gitignore at ${gitignorePath}.`);
  }

  if (!existsSync(policyPath)) {
    throw new Error(`[clean:repo] Missing clean-room policy at ${policyPath}.`);
  }

  const patterns = loadPatterns(policyPath, repoName);
  const candidates = collectCleanCandidates(rootDir, patterns);

  if (candidates.length === 0) {
    console.log(`[clean:repo] No ephemeral paths found for ${repoName}.`);
    return;
  }

  const relTargets = candidates.map((candidate) => normalizePath(relative(rootDir, candidate)));

  if (dryRun) {
    console.log(`[clean:repo] Dry run: ${candidates.length} path(s) would be removed in ${repoName}.`);
    for (const path of relTargets) {
      console.log(`  - ${path}`);
    }
    return;
  }

  for (const candidate of candidates) {
    rmSync(candidate, { recursive: true, force: true });
  }

  console.log(`[clean:repo] Removed ${candidates.length} path(s) in ${repoName}.`);
  for (const path of relTargets) {
    console.log(`  - ${path}`);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const args = parseArgs(process.argv.slice(2));
    runCleanRepo({
      repoRoot: process.cwd(),
      repoName: args.repoName,
      dryRun: args.dryRun,
      policyPath: args.policyPath,
    });
    process.exit(0);
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
}
