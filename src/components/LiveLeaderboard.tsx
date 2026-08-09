import type { Brain, GameState, PersistedProfile } from "@/game/types";

type Props = { state: GameState; profiles: PersistedProfile[]; brains: Brain[]; trainer?: boolean };

export function LiveLeaderboard({ state, profiles, brains, trainer = false }: Props) {
  const rows = state.snakes.map((snake) => {
    const profile = profiles.find((entry) => entry.snakeId === snake.id);
    const brain = brains.find((entry) => entry.snakeId === snake.id) ?? profile?.brain;
    return { snake, profile, brain };
  }).sort((a, b) => b.snake.score - a.snake.score || Number(b.snake.alive) - Number(a.snake.alive));

  return (
    <aside className="leaderboard glass-card">
      <div className="panel-heading">
        <div><span className="eyebrow">Live ranking</span><h2>{trainer ? "Training run" : "Leaderboard"}</h2></div>
        <span className="live-pill"><i /> LIVE</span>
      </div>
      <div className="leader-list">
        {rows.map(({ snake, profile, brain }, index) => {
          const benchmarkRate = brain?.lastBenchmark ? Math.round(brain.lastBenchmark.winRate * 100) : null;
          return <div className={`leader-row ${!snake.alive ? "eliminated" : ""}`} key={snake.id}>
            <span className="rank">{String(index + 1).padStart(2, "0")}</span>
            <span className="snake-swatch" style={{ "--snake-color": snake.color } as React.CSSProperties} />
            <div className="leader-name"><strong>{snake.name}</strong><span>{snake.alive ? `${snake.segments.length} cells` : "ELIMINATED"}</span></div>
            <div className="leader-score"><strong>{snake.score}</strong><span>PTS</span></div>
            <div className="brain-meter" title={benchmarkRate === null ? "Not benchmarked yet" : `Benchmark win rate ${benchmarkRate}%`}>
              <i style={{ width: `${benchmarkRate ?? 0}%` }} />
            </div>
            {!trainer && <span className="career-wins">{profile?.wins ?? 0}W</span>}
          </div>;
        })}
      </div>
      <div className="leader-foot"><span>Rank awareness feeds each AI brain</span><strong>DEEP Q-NETWORK</strong></div>
    </aside>
  );
}
