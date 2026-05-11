export type MarbleWorld = {
  id?: string;
  display_name?: string;
  world_marble_url?: string;
  assets?: {
    caption?: string;
    thumbnail_url?: string;
    splats?: {
      spz_urls?: Record<string, string>;
      lod_url?: string;
      lod_urls?: Record<string, string>;
      lod_meta_url?: string;
      lod_metadata_url?: string;
      lod_meta_json_url?: string;
      semantics_metadata?: {
        metric_scale_factor?: number;
        ground_plane_offset?: number;
      };
    };
    mesh?: { collider_mesh_url?: string };
    imagery?: { pano_url?: string };
  };
};

export type JourneyRecord = {
  ulid: string;
  operationId: string;
  prompt: string;
  style: string | null;
  imageUrl: string;
  createdAt: string;
  status: "pending" | "done" | "error";
  error?: string;
  world?: MarbleWorld | null;
};

const RES_ORDER = ["full_res", "500k", "150k", "100k"] as const;
const LOD_KEYS = [
  "lod",
  "lod_meta",
  "lod-meta",
  "lod_meta_json",
  "lod_metadata",
  "full_res",
  "500k",
] as const;

export type PickedSplat = {
  url: string;
  paged: boolean;
};

function looksPagedLodUrl(url: string): boolean {
  return /\.lod-meta\.json($|\?)/.test(url) || /lod[-_]?meta.*\.json($|\?)/.test(url);
}

export function pickSplat(rec: JourneyRecord): PickedSplat | null {
  const splats = rec.world?.assets?.splats;
  if (!splats) return null;

  const directLod =
    splats.lod_meta_json_url ??
    splats.lod_metadata_url ??
    splats.lod_meta_url ??
    splats.lod_url;
  if (directLod) {
    return { url: directLod, paged: looksPagedLodUrl(directLod) };
  }

  const lodUrls = splats.lod_urls;
  if (lodUrls) {
    for (const key of LOD_KEYS) {
      if (lodUrls[key]) return { url: lodUrls[key], paged: looksPagedLodUrl(lodUrls[key]) };
    }
  }

  const urls = splats.spz_urls;
  if (!urls) return null;
  for (const key of RES_ORDER) {
    if (urls[key]) return { url: urls[key], paged: false };
  }
  return null;
}

export function pickSplatUrl(rec: JourneyRecord): string | null {
  return pickSplat(rec)?.url ?? null;
}

export function pickJourneyImage(rec: JourneyRecord): string {
  return (
    rec.world?.assets?.thumbnail_url ??
    rec.world?.assets?.imagery?.pano_url ??
    rec.imageUrl
  );
}
