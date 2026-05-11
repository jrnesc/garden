import Nav from "@/components/Nav";
import { serverFetch } from "@/lib/api";
import { JourneyRecord, pickJourneyImage } from "@/lib/journey";
import Link from "next/link";

async function fetchSplats(): Promise<JourneyRecord[]> {
  try {
    const res = await serverFetch("/splats", { cache: "no-store" });
    if (!res.ok) return [];
    const data = (await res.json()) as { splats?: JourneyRecord[]; journeys?: JourneyRecord[] };
    return data.splats ?? data.journeys ?? [];
  } catch {
    return [];
  }
}

export default async function SplatsPage() {
  const splats = await fetchSplats();

  return (
    <div className="min-h-screen">
      <Nav />
      <main className="mx-auto max-w-5xl px-8 pb-24 pt-20">
        <h1 className="mb-10 text-5xl tracking-tight lowercase">splats</h1>

        {splats.length === 0 ? (
          <p className="text-sm text-[var(--muted)]">nothing yet.</p>
        ) : (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {splats.map((splat) => {
              const clickable = splat.status === "done";
              const imageUrl = pickJourneyImage(splat);
              const card = (
                <div
                  className={`overflow-hidden rounded-md ${
                    clickable ? "hover:opacity-90" : "opacity-50"
                  }`}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imageUrl}
                    alt=""
                    className="aspect-video w-full bg-[var(--foreground)]/10 object-cover"
                  />
                </div>
              );
              return clickable ? (
                <Link key={splat.ulid} href={`/walk?ulid=${splat.ulid}`}>
                  {card}
                </Link>
              ) : (
                <div key={splat.ulid}>{card}</div>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}
