#!/usr/bin/env node
import { runVerifyWorkflowCommands } from '../../scripts/verify-workflow-commands.mjs';

runVerifyWorkflowCommands({
  repoName: 'edge-node-runtime',
  repoRoot: process.cwd(),
});
