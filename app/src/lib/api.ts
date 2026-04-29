// Native fetch only — never axios. (Harness rule from Paths.)
// All Worker calls route through this so there's one place to swap the base URL.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8787";

/**
 * Server-side fetch using the Cloudflare service binding (Worker-to-Worker).
 * Falls back to plain fetch for client components and local dev.
 */
export async function serverFetch(path: string, init?: RequestInit) {
  // In a Cloudflare Worker, use the service binding to avoid the
  // same-account workers.dev routing issue.
  if (typeof globalThis !== "undefined") {
    try {
      const { getCloudflareContext } = await import(
        "@opennextjs/cloudflare"
      );
      const { env } = await getCloudflareContext({ async: true });
      const api = (env as Record<string, unknown>).API as
        | { fetch: (req: Request) => Promise<Response> }
        | undefined;
      if (api) {
        return api.fetch(
          new Request(`https://inside-api${path}`, init)
        );
      }
    } catch {
      // Not in Cloudflare context (local dev) — fall through to plain fetch.
    }
  }
  return fetch(`${API_BASE}${path}`, init);
}

export async function apiFetch(path: string, init?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  }
  return res;
}
