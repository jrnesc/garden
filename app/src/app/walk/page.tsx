import Walker from "@/components/Walker";
import { serverFetch } from "@/lib/api";
import { JourneyRecord, pickSplat } from "@/lib/journey";

const SAMPLE_SPZ = "https://sparkjs.dev/assets/splats/butterfly.spz";

type SearchParams = Promise<{ ulid?: string; splatUrl?: string }>;


async function fetchJourney(ulid: string): Promise<JourneyRecord | null> {
  const res = await serverFetch(`/splats/${ulid}`, { cache: "no-store" });
  if (!res.ok) return null;
  return (await res.json()) as JourneyRecord;
}

export default async function WalkPage({
  searchParams,
}: {
  searchParams: SearchParams;
}) {
  const { ulid, splatUrl } = await searchParams;

  let url: string = SAMPLE_SPZ;
  let userPrompt: string | null = null;
  let sceneCaption: string | null = null;
  let colliderMeshUrl: string | null = null;
  let metricScale: number | null = null;
  let splatPaged = false;
  const backHref = ulid ? "/splats" : "/";

  if (ulid) {
    const rec = await fetchJourney(ulid);
    const resolved = rec ? pickSplat(rec) : null;
    if (resolved) {
      url = resolved.url;
      splatPaged = resolved.paged;
      userPrompt = rec?.prompt ?? null;
      sceneCaption = rec?.world?.assets?.caption ?? null;
      colliderMeshUrl = rec?.world?.assets?.mesh?.collider_mesh_url ?? null;
      const meta = rec?.world?.assets?.splats?.semantics_metadata;
      metricScale = meta?.metric_scale_factor ?? null;
    }
  } else if (splatUrl) {
    url = splatUrl;
  }

  return (
    <div className="fixed inset-0 bg-black text-zinc-100">
      <Walker
        key={ulid ?? splatUrl ?? "default"}
        splatUrl={url}
        splatPaged={splatPaged}
        colliderMeshUrl={colliderMeshUrl}
        metricScale={metricScale}
        userPrompt={userPrompt}
        sceneCaption={sceneCaption}
        backHref={backHref}
      />
    </div>
  );
}
