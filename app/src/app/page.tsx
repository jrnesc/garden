import Link from "next/link";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <nav className="flex items-center justify-end gap-6 px-8 py-6 text-sm text-[var(--muted)]">
        <Link href="/splats" className="hover:text-[var(--foreground)]">
          splats
        </Link>
        <Link href="/character" className="hover:text-[var(--foreground)]">
          character
        </Link>
      </nav>

      <main className="flex flex-1 flex-col items-start justify-end px-8 pb-12">
        <h1 className="text-[clamp(3rem,10vw,7rem)] leading-[0.9] tracking-tight lowercase">
          garden
        </h1>
      </main>
    </div>
  );
}
