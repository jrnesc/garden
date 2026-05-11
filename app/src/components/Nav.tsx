import Link from "next/link";

export default function Nav() {
  return (
    <nav className="flex items-center justify-between px-8 py-6 text-sm">
      <Link href="/" className="text-[var(--foreground)]">
        garden
      </Link>
      <div className="flex gap-6 text-[var(--muted)]">
        <Link href="/splats" className="hover:text-[var(--foreground)]">
          splats
        </Link>
        <Link href="/character" className="hover:text-[var(--foreground)]">
          character
        </Link>
        <Link href="/nvidia-motion" className="hover:text-[var(--foreground)]">
          nvidia motion
        </Link>
      </div>
    </nav>
  );
}
