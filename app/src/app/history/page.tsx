import Nav from "@/components/Nav";
import { serverFetch } from "@/lib/api";
import { JourneyRecord } from "@/lib/journey";
import Link from "next/link";

async function fetchJourneys(): Promise<JourneyRecord[]> {
  const res = await serverFetch("/journeys", { cache: "no-store" });
  if (!res.ok) return [];
  const data = (await res.json()) as { journeys: JourneyRecord[] };
  return data.journeys;
}

export default async function HistoryPage() {
  const journeys = await fetchJourneys();

  return (
    <div className="min-h-screen">
      <Nav />
      <main className="mx-auto max-w-5xl px-8 pb-24 pt-20">
        <h1 className="mb-10 text-5xl tracking-tight lowercase">gallery</h1>

        {journeys.length === 0 ? (
          <p className="text-sm text-[var(--muted)]">nothing yet.</p>
        ) : (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {journeys.map((j) => {
              const clickable = j.status === "done";
              const card = (
                <div
                  className={`overflow-hidden rounded-md ${
                    clickable ? "hover:opacity-90" : "opacity-50"
                  }`}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={j.imageUrl}
                    alt=""
                    className="aspect-video w-full object-cover"
                  />
                </div>
              );
              return clickable ? (
                <Link key={j.ulid} href={`/walk?ulid=${j.ulid}`}>
                  {card}
                </Link>
              ) : (
                <div key={j.ulid}>{card}</div>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}
