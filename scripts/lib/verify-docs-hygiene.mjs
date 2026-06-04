#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';

import { runVerifyAgentClaudeSync } from './verify-agent-claude-sync.mjs';
import { runVerifyCleanRoom } from './verify-clean-room.mjs';
import { runVerifyToolchainLockfile } from './verify-toolchain-lockfile.mjs';
import { runVerifyWorkflowCommands } from './verify-workflow-commands.mjs';

const DEFAULT_DOCS_FILES = 'AGENTS.md,CLAUDE.md';
const REQUIRED_GITATTRIBUTES = [
  '* text eol=lf',
  '*.md text eol=lf working-tree-encoding=UTF-8',
  '*.json text eol=lf working-tree-encoding=UTF-8',
  '*.js text eol=lf working-tree-encoding=UTF-8',
  '*.mjs text eol=lf working-tree-encoding=UTF-8',
  '*.ts text eol=lf working-tree-encoding=UTF-8',
  '*.py text eol=lf working-tree-encoding=UTF-8',
  '*.png -text',
  '*.jpg -text',
  '*.jpeg -text',
  '*.gif -text',
  '*.webp -text',
  '*.pdf -text',
];
const REQUIRED_GITIGNORE_PATTERNS = [
  '*.log',
  '*.out',
  '*.err',
  '*.err.txt',
  '*.jsonl',
  '*.json.err',
  '*.json.err.txt',
  '*stress*.json',
  '*stress*.log',
  'stress*.json',
  'codex-*.json',
  'vite-dev*.log',
  'coverage/',
  'dist/',
  'build/',
  'runtime-state/',
  'state/',
  'test artifacts/',
  'test-artifacts/',
  'test-output/',
  'logs/',
];

function parseRepoName() {
  if (process.argv.includes('--repo') && process.argv.indexOf('--repo') + 1 < process.argv.length) {
    return process.argv[process.argv.indexOf('--repo') + 1];
  }

  return undefined;
}

function parseCliOptions() {
  const options = {
    repoName: parseRepoName(),
    dryRun: false,
  };

  if (process.argv.includes('--dry-run')) {
    options.dryRun = true;
  }

  return options;
}

function normalizePattern(line) {
  return line.trim().replace(/\s+/g, ' ');
}

function normalizeIgnorePattern(line) {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith('#')) {
    return null;
  }

  return trimmed;
}

function verifyGitattributes(repoRoot) {
  const gitattributesPath = resolve(repoRoot, '.gitattributes');
  if (!existsSync(gitattributesPath)) {
    throw new Error('[docs-hygiene] Missing .gitattributes file.');
  }

  const lines = readFileSync(gitattributesPath, 'utf8')
    .split('\n')
    .map(normalizePattern)
    .filter((line) => line && !line.startsWith('#'));
  const ruleSet = new Set(lines);

  const missing = REQUIRED_GITATTRIBUTES.filter((rule) => !ruleSet.has(normalizePattern(rule)));
  if (missing.length > 0) {
    console.error('[docs-hygiene] Missing .gitattributes hygiene rules:');
    for (const rule of missing) {
      console.error(`  - ${rule}`);
    }
    throw new Error('[docs-hygiene] .gitattributes rules are incomplete.');
  }
}

function verifyGitignore(repoRoot) {
  const gitignorePath = resolve(repoRoot, '.gitignore');
  if (!existsSync(gitignorePath)) {
    throw new Error('[docs-hygiene] Missing .gitignore file.');
  }

  const lines = readFileSync(gitignorePath, 'utf8')
    .split('\n')
    .map(normalizeIgnorePattern)
    .filter(Boolean);
  const ruleSet = new Set(lines);

  const missing = REQUIRED_GITIGNORE_PATTERNS.filter((pattern) => !ruleSet.has(pattern));
  if (missing.length > 0) {
    console.error('[docs-hygiene] Missing .gitignore hygiene rules:');
    for (const rule of missing) {
      console.error(`  - ${rule}`);
    }
    throw new Error('[docs-hygiene] .gitignore rules are incomplete.');
  }
}

function runNodeCommand(repoRoot, command, args) {
  execFileSync(command, args, {
    cwd: repoRoot,
    stdio: 'inherit',
  });
}

function runCommandChecks(repoRoot) {
  const packagePath = resolve(repoRoot, 'package.json');
  if (existsSync(packagePath)) {
    runVerifyAgentClaudeSync({ repoRoot, strict: true });
    runNodeCommand(
      repoRoot,
      process.execPath,
      [resolve(repoRoot, 'scripts', 'verify-doc-commands.mjs'), '--root', '.', '--docs', DEFAULT_DOCS_FILES],
    );
    return;
  }

  const pythonCheck = resolve(repoRoot, 'scripts', 'verify_agent_claude_sync.py');
  if (!existsSync(pythonCheck)) {
    throw new Error('[docs-hygiene] Missing AGENTS/CLAUDE sync verifier for this repository.');
  }

  runNodeCommand(repoRoot, 'python', [
    'scripts/verify_agent_claude_sync.py',
    '--expected-no-npm-run-only',
    '--strict',
  ]);
}

function executeCheck(name, check, failures) {
  try {
    check();
    console.log(`[docs-hygiene] ${name}: pass`);
    return;
  } catch (error) {
    failures.push({ name, error });
    console.error(`[docs-hygiene] ${name}: fail`);
    console.error(`  ${error.message}`);
  }
}

export function runVerifyDocsHygiene({
  repoRoot = process.cwd(),
  repoName = basename(repoRoot),
  dryRun = false,
  policyPath,
} = {}) {
  const root = resolve(repoRoot);
  const failures = [];

  const run = dryRun ? (name, fn) => executeCheck(name, fn, failures) : (name, fn) => {
    try {
      fn();
      console.log(`[docs-hygiene] ${name}: pass`);
    } catch (error) {
      throw error;
    }
  };

  run('toolchain-lockfile', () => runVerifyToolchainLockfile({ repoName, repoRoot: root }));
  run('workflow-commands', () => runVerifyWorkflowCommands({ repoName, repoRoot: root, strict: true }));
  run('clean-room', () => runVerifyCleanRoom({
    repoName,
    repoRoot: root,
    strict: true,
    ...(policyPath ? { policyPath } : {}),
  }));
  run('gitattributes', () => verifyGitattributes(root));
  run('gitignore', () => verifyGitignore(root));
  run('command-contract', () => runCommandChecks(root));

  if (dryRun && failures.length > 0) {
    console.error(`[docs-hygiene] Dry-run found ${failures.length} issue(s) for ${repoName}.`);
    return;
  }

  if (dryRun) {
    console.log(`[docs-hygiene] Dry-run found no blocking issues for ${repoName}.`);
    return;
  }

  console.log(`[docs-hygiene] Docs hygiene checks passed for ${repoName}.`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const options = parseCliOptions();
    runVerifyDocsHygiene({
      repoRoot: process.cwd(),
      repoName: options.repoName,
      dryRun: options.dryRun,
    });
    process.exit(0);
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
}
