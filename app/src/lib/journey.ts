export type MarbleWorld = {
  id?: string;
  display_name?: string;
  world_marble_url?: string;
  assets?: {
    caption?: string;
    thumbnail_url?: string;
    splats?: {
      spz_urls?: Record<string, string>;
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

// Prefer highest quality. Fragment cost dominates over gaussian count once antialias is off
// and pixelRatio is tuned, so loading full_res directly is fine.
const RES_ORDER = ["full_res", "500k", "150k", "100k"] as const;

export function pickSplatUrl(rec: JourneyRecord): string | null {
  const urls = rec.world?.assets?.splats?.spz_urls;
  if (!urls) return null;
  for (const key of RES_ORDER) {
    if (urls[key]) return urls[key];
  }
  return null;
}
