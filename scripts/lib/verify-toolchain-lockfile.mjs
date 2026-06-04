#!/usr/bin/env node
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { basename, extname, relative, resolve } from 'node:path';
import { trimLine } from './text-utils.mjs';

const LOCKFILE_FOR_MANAGER = {
  npm: 'package-lock.json',
  pnpm: 'pnpm-lock.yaml',
  yarn: 'yarn.lock',
};

const TOOLCHAIN_RE = /\b(?<manager>npm|pnpm|yarn)\s+(?<command>ci|i|install)\b/g;

export function runVerifyToolchainLockfile({
  repoRoot = process.cwd(),
  repoName = basename(repoRoot),
} = {}) {
  const root = resolve(repoRoot);
  const workflowDir = resolve(root, '.github', 'workflows');
  if (!existsSync(workflowDir)) {
    throw new Error(`[toolchain-lockfile] Missing workflow directory: ${workflowDir}`);
  }

  const files = readdirSync(workflowDir)
    .filter((name) => ['.yml', '.yaml'].includes(extname(name)))
    .map((name) => resolve(workflowDir, name));

  const refs = [];
  for (const workflowPath of files) {
    const lines = readFileSync(workflowPath, 'utf8').split('\n');
    for (let index = 0; index < lines.length; index += 1) {
      const line = trimLine(lines[index]);
      if (!line || line.startsWith('#')) {
        continue;
      }

      for (const match of line.matchAll(TOOLCHAIN_RE)) {
        const manager = match.groups.manager;
        const command = match.groups.command === 'i' ? 'install' : match.groups.command;
        refs.push({
          workflow: relative(root, workflowPath),
          line: index + 1,
          manager,
          command,
          text: line,
        });
      }
    }
  }

  if (refs.length === 0) {
    console.log(`[toolchain-lockfile] No package-manager install commands found for ${repoName}.`);
    return;
  }

  const managers = [...new Set(refs.map((ref) => ref.manager))];
  if (managers.length > 1) {
    console.error(
      '[toolchain-lockfile] Multiple package managers detected in workflows. Use one toolchain per repo.',
    );
    for (const ref of refs) {
      console.error(
        `  repo=${repoName} file=${ref.workflow} line=${ref.line} manager=${ref.manager} command=${ref.command}`,
      );
    }
    throw new Error('[toolchain-lockfile] Package manager mix detected.');
  }

  const lockFiles = {
    npm: existsSync(resolve(root, LOCKFILE_FOR_MANAGER.npm)),
    pnpm: existsSync(resolve(root, LOCKFILE_FOR_MANAGER.pnpm)),
    yarn: existsSync(resolve(root, LOCKFILE_FOR_MANAGER.yarn)),
  };

  const primaryManager = managers[0];
  const primaryRefs = refs.filter((ref) => ref.manager === primaryManager);

  const failures = [];
  if (primaryManager === 'npm') {
    if (primaryRefs.some((ref) => ref.command === 'ci') && !lockFiles.npm) {
      failures.push(`workflow references npm ci but ${LOCKFILE_FOR_MANAGER.npm} is missing`);
    }
  } else if (primaryManager === 'pnpm') {
    if (!lockFiles.pnpm) {
      failures.push(`workflow references pnpm install but ${LOCKFILE_FOR_MANAGER.pnpm} is missing`);
    }
  } else if (primaryManager === 'yarn') {
    if (!lockFiles.yarn) {
      failures.push(`workflow references yarn install but ${LOCKFILE_FOR_MANAGER.yarn} is missing`);
    }
  }

  for (const otherManager of Object.keys(LOCKFILE_FOR_MANAGER)) {
    if (otherManager !== primaryManager && lockFiles[otherManager]) {
      failures.push(
        `workflow uses ${primaryManager} but ${LOCKFILE_FOR_MANAGER[otherManager]} suggests another package manager`,
      );
    }
  }

  if (failures.length > 0) {
    console.error(`[toolchain-lockfile] Lockfile sanity failed for ${repoName}.`);
    for (const line of failures) {
      console.error(`  - ${line}`);
    }
    console.error('[toolchain-lockfile] Toolchain references found:');
    for (const ref of refs) {
      console.error(
        `  repo=${repoName} file=${ref.workflow} line=${ref.line} manager=${ref.manager} command=${ref.command} text=${ref.text}`,
      );
    }
    throw new Error('[toolchain-lockfile] Lockfile and package-manager expectations diverge.');
  }

  const lockLabel = lockFiles[primaryManager]
    ? LOCKFILE_FOR_MANAGER[primaryManager]
    : 'no lockfile (allowed for npm install)';
  console.log(
    `[toolchain-lockfile] ${repoName} toolchain checks align with workflow commands (${primaryManager}/${lockLabel}).`,
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const args = {
      repoName: process.argv.includes('--repo')
        ? process.argv[process.argv.indexOf('--repo') + 1]
        : undefined,
    };
    runVerifyToolchainLockfile(args);
    process.exit(0);
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
}
