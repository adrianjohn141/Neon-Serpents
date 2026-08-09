"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function Nav() {
  const pathname = usePathname();
  return (
    <header className="site-header">
      <Link href="/" className="brand" aria-label="Neon Serpents home">
        <span className="brand-mark">S</span>
        <span><strong>NEON SERPENTS</strong><small>Autonomous battle league</small></span>
      </Link>
      <nav>
        <Link className={pathname === "/" ? "active" : ""} href="/"><span>01</span> Battle Arena</Link>
        <Link className={pathname.startsWith("/train") ? "active" : ""} href="/train"><span>02</span> Train AI</Link>
        <Link className={pathname.startsWith("/server-training") ? "active" : ""} href="/server-training"><span>03</span> Server Lab</Link>
        <Link className={pathname.startsWith("/intelligence") ? "active" : ""} href="/intelligence"><span>04</span> Intelligence</Link>
      </nav>
      <div className="system-status"><i /> SYSTEM ONLINE</div>
    </header>
  );
}
