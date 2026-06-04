#!/usr/bin/env node
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { basename, extname, relative, resolve } from 'node:path';

const SCRIPT_RE = /\bnpm\s+run\s+([A-Za-z0-9][A-Za-z0-9:._-]*)(?=(?:\s|$|[;&|]))/g;
const MISSING_COMMAND_REASON_BY_COMMAND = {
  "check:ci":
    "npm run check:ci appears in a workflow without a corresponding `check:ci` script in package.json",
};

function parseBoolArg(name, defaultValue = false) {
  const arg = `--${name}`;
  return process.argv.includes(arg) || defaultValue;
}

export function runVerifyWorkflowCommands(args = {}) {
  const { repoRoot = process.cwd(), repoName = basename(repoRoot), strict = false } = args;

  const root = resolve(repoRoot);
  const workflowDir = resolve(root, '.github', 'workflows');
  const missingEntries = [];
  const refs = [];

  if (!existsSync(workflowDir)) {
    throw new Error(`[workflow-commands] Missing workflow directory: ${workflowDir}`);
  }

  const files = readdirSync(workflowDir)
    .filter((name) => ['.yml', '.yaml'].includes(extname(name)))
    .map((name) => resolve(workflowDir, name));

  for (const workflowPath of files) {
    const lines = readFileSync(workflowPath, 'utf8').split('\n');
    lines.forEach((line, index) => {
      if (line.trimStart().startsWith('#')) {
        return;
      }
      for (const match of line.matchAll(SCRIPT_RE)) {
        const command = match[1];
        refs.push({
          workflow: relative(root, workflowPath),
          line: index + 1,
          command,
          text: line.trim(),
        });
      }
    });
  }

  if (refs.length === 0) {
    console.log('[workflow-commands] No `npm run` references found in workflow files.');
    return;
  }

  const packagePath = resolve(root, 'package.json');
  if (!existsSync(packagePath)) {
    if (strict || refs.length > 0) {
      for (const ref of refs) {
        missingEntries.push({
          ...ref,
          reason: 'package.json missing',
        });
      }
      console.error('[workflow-commands] `npm run` references found but no package.json exists in this repo.');
      printFailures(missingEntries, repoName);
      throw new Error('[workflow-commands] Workflow command contract check failed.');
    }
    return;
  }

  const packageData = JSON.parse(readFileSync(packagePath, 'utf8'));
  const packageScripts = new Set(Object.keys(packageData.scripts || {}));

  for (const ref of refs) {
    if (!packageScripts.has(ref.command)) {
      const reason =
        MISSING_COMMAND_REASON_BY_COMMAND[ref.command] || "script missing from package.json";
      missingEntries.push({
        ...ref,
        reason,
      });
    }
  }

  if (missingEntries.length > 0) {
    console.error('[workflow-commands] Missing package scripts referenced in workflow files:');
    printFailures(missingEntries, repoName);
    throw new Error('[workflow-commands] Workflow command contract check failed.');
  }

  console.log(
    `[workflow-commands] All workflow ` +
      `npm run references resolve to package.json scripts for ${repoName}.`,
  );

  if (strict && !missingEntries.length) {
    console.log('[workflow-commands] Strict mode enabled. Workflow contract checks are complete.');
  }
}

function printFailures(failures, repoName) {
  for (const item of failures) {
    console.error(
      `  repo=${repoName} file=${item.workflow} line=${item.line} command=${item.command} reason=${item.reason}`,
    );
    if (item.text) {
      console.error(`    line: ${item.text}`);
    }
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const args = {
      repoName: process.argv.includes('--repo')
        ? process.argv[process.argv.indexOf('--repo') + 1]
        : undefined,
      strict: parseBoolArg('strict'),
    };
    runVerifyWorkflowCommands(args);
    process.exit(0);
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
}
