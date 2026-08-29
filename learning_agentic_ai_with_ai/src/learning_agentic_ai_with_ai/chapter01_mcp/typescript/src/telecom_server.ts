/**
 * Telecom-ops MCP server (TypeScript, official SDK).
 *
 * Same protocol, different language: this server exposes telecom tools over
 * stdio using the official TypeScript SDK. The Python client in this chapter
 * connects to it exactly as it connects to the Python servers - proving MCP
 * is a true interop protocol, not a Python library convention.
 *
 * Run:  node --experimental-strip-types src/telecom_server.ts
 * (from this directory, after `npm install`)
 */
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

/** Minimal deterministic telecom dataset (mirrors the Python mock DB). */
interface Site {
  name: string;
  region: string;
  status: string;
  battery: number;
}

const SITES: Record<string, Site> = {
  'CS-11': { name: 'Lisboa Centro', region: 'lisbon', status: 'healthy', battery: 8.0 },
  'CS-44': { name: 'Porto Norte', region: 'porto', status: 'degraded', battery: 4.0 },
  'CS-55': { name: 'Faro Algarve', region: 'algarve', status: 'healthy', battery: 5.0 },
  'CS-77': { name: 'Braga Tecnopolo', region: 'braga', status: 'degraded', battery: 2.5 },
};

interface Tech {
  name: string;
  region: string;
  onDuty: boolean;
}

const TECHS: Record<string, Tech> = {
  'T-01': { name: 'Ana Rocha', region: 'lisbon', onDuty: true },
  'T-03': { name: 'Carla Costa', region: 'porto', onDuty: true },
  'T-06': { name: 'Filipe Braga', region: 'braga', onDuty: true },
  'T-04': { name: 'Diogo Lima', region: 'porto', onDuty: false },
};

let dispatchSeq = 1;

const server = new McpServer({
  name: 'telecom-ops-ts',
  version: '1.0.0',
});

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

server.tool(
  'telecom_site_status',
  'Return status and latest metrics for one cell site.',
  { site_id: z.string().describe('Cell site id, e.g. CS-77') },
  async ({ site_id }: { site_id: string }) => {
    const site = SITES[site_id];
    if (!site) {
      return {
        content: [{ type: 'text' as const, text: `unknown site: ${site_id}` }],
        isError: true,
      };
    }
    const degraded = site.status === 'degraded';
    const payload = {
      site_id,
      ...site,
      latency_ms: degraded ? 312.4 : 38.2,
      packet_loss_pct: degraded ? 4.8 : 0.2,
      active_users: degraded ? 1720 : 6210,
    };
    return { content: [{ type: 'text' as const, text: JSON.stringify(payload) }] };
  },
);

server.tool(
  'telecom_degraded_sites',
  'List cell sites currently degraded, optionally filtered by region.',
  { region: z.string().optional().describe('Optional region filter') },
  async ({ region }: { region?: string }) => {
    const sites = Object.entries(SITES)
      .filter(([, s]) => s.status === 'degraded' && (!region || s.region === region))
      .map(([sid, s]) => ({ site_id: sid, ...s }));
    return { content: [{ type: 'text' as const, text: JSON.stringify({ sites }) }] };
  },
);

server.tool(
  'telecom_dispatch_technician',
  'Dispatch an on-duty technician to a cell site (write tool).',
  {
    site_id: z.string().describe('Cell site id'),
    tech_id: z.string().describe('Technician id'),
    priority: z.enum(['low', 'medium', 'high']).describe('Dispatch priority'),
    note: z.string().max(300).optional().describe('Short note'),
  },
  async (args: { site_id: string; tech_id: string; priority: 'low' | 'medium' | 'high'; note?: string }) => {
    const { site_id, tech_id, priority, note } = args;
    const site = SITES[site_id];
    if (!site) {
      return {
        content: [{ type: 'text' as const, text: `unknown site: ${site_id}` }],
        isError: true,
      };
    }
    const tech = TECHS[tech_id];
    if (!tech) {
      return {
        content: [{ type: 'text' as const, text: `unknown technician: ${tech_id}` }],
        isError: true,
      };
    }
    if (!tech.onDuty) {
      return {
        content: [{ type: 'text' as const, text: `technician ${tech_id} is off duty` }],
        isError: true,
      };
    }
    const dispatch = {
      dispatch_id: dispatchSeq++,
      site_id,
      tech_id,
      priority,
      note: note ?? '',
      status: 'dispatched',
      created_at: new Date().toISOString(),
    };
    return { content: [{ type: 'text' as const, text: JSON.stringify(dispatch) }] };
  },
);

async function main(): Promise<void> {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  // NOTICE: nothing else may be printed to stdout - stdio IS the protocol.
  console.error('telecom-ops-ts server ready on stdio');
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
