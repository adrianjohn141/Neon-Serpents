import type { Metadata } from "next";
import { Nav } from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "Neon Serpents — Autonomous Snake Battle",
  description: "A dynamic reinforcement-learning Snake Battle arena for up to eight agents.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><Nav />{children}<footer className="site-footer"><span>NEON SERPENTS // AI COMBAT SIMULATION</span><span>ADAPTIVE ARENA · INDIVIDUAL BRAINS · LAST SNAKE WINS</span></footer></body></html>;
}
