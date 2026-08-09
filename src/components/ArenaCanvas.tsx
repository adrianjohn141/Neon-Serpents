"use client";

import { useEffect, useRef } from "react";
import { POWER_UP_META } from "@/game/constants";
import { boundsForInset, hazardCells } from "@/game/adaptive-arena";
import type { GameState } from "@/game/types";

type Props = { state: GameState; compact?: boolean };

function roundedRect(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radius: number) {
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, radius);
}

export function ArenaCanvas({ state, compact = false }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const draw = () => {
      const bounds = canvas.getBoundingClientRect();
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      const width = Math.max(1, Math.round(bounds.width * ratio));
      const height = Math.max(1, Math.round(bounds.height * ratio));
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      const cell = Math.min(bounds.width / state.width, bounds.height / state.height);
      const boardWidth = cell * state.width;
      const boardHeight = cell * state.height;
      const ox = (bounds.width - boardWidth) / 2;
      const oy = (bounds.height - boardHeight) / 2;

      const gradient = context.createLinearGradient(0, 0, bounds.width, bounds.height);
      gradient.addColorStop(0, "#071817");
      gradient.addColorStop(0.55, "#081310");
      gradient.addColorStop(1, "#11120d");
      context.fillStyle = gradient;
      context.fillRect(0, 0, bounds.width, bounds.height);

      context.strokeStyle = "rgba(103, 247, 193, 0.045)";
      context.lineWidth = 1;
      for (let x = 0; x <= state.width; x += 4) {
        context.beginPath(); context.moveTo(ox + x * cell, oy); context.lineTo(ox + x * cell, oy + boardHeight); context.stroke();
      }
      for (let y = 0; y <= state.height; y += 4) {
        context.beginPath(); context.moveTo(ox, oy + y * cell); context.lineTo(ox + boardWidth, oy + y * cell); context.stroke();
      }

      context.shadowBlur = 0;
      if (state.arena.enabled) {
        const safe = boundsForInset(state.width, state.height, state.arena.safeZone.pendingInset);
        if (state.tick >= state.arena.safeZone.telegraphAt && state.arena.safeZone.pendingInset > state.arena.safeZone.inset) {
          context.fillStyle = "rgba(255, 71, 98, .10)";
          context.fillRect(ox, oy, boardWidth, safe.minY * cell);
          context.fillRect(ox, oy + (safe.maxY + 1) * cell, boardWidth, (state.height - safe.maxY - 1) * cell);
          context.fillRect(ox, oy + safe.minY * cell, safe.minX * cell, (safe.maxY - safe.minY + 1) * cell);
          context.fillRect(ox + (safe.maxX + 1) * cell, oy + safe.minY * cell, (state.width - safe.maxX - 1) * cell, (safe.maxY - safe.minY + 1) * cell);
          context.strokeStyle = "rgba(255, 103, 123, .9)"; context.lineWidth = Math.max(1, cell * .12); context.setLineDash([cell * .7, cell * .5]);
          context.strokeRect(ox + safe.minX * cell, oy + safe.minY * cell, (safe.maxX - safe.minX + 1) * cell, (safe.maxY - safe.minY + 1) * cell);
          context.setLineDash([]);
        }
        if (state.arena.dormantObstacles.length && state.arena.phase !== "opening" && state.tick < state.arena.obstaclesActivateAt) {
          context.fillStyle = "rgba(255, 209, 102, .14)"; context.strokeStyle = "rgba(255, 209, 102, .55)";
          for (const point of state.arena.dormantObstacles) {
            context.fillRect(ox + point.x * cell + cell * .12, oy + point.y * cell + cell * .12, cell * .76, cell * .76);
            context.strokeRect(ox + point.x * cell + cell * .12, oy + point.y * cell + cell * .12, cell * .76, cell * .76);
          }
        }
        for (const hazard of state.arena.hazards) {
          const active = state.tick >= hazard.activatesAt;
          const points = hazardCells(hazard, active ? state.tick : hazard.activatesAt);
          context.fillStyle = active ? "rgba(255, 62, 91, .55)" : "rgba(255, 209, 102, .16)";
          context.strokeStyle = active ? "#ff4967" : "rgba(255, 209, 102, .75)";
          context.lineWidth = Math.max(1, cell * .08);
          for (const point of points) {
            context.fillRect(ox + point.x * cell + cell * .08, oy + point.y * cell + cell * .08, cell * .84, cell * .84);
            context.strokeRect(ox + point.x * cell + cell * .08, oy + point.y * cell + cell * .08, cell * .84, cell * .84);
          }
        }
        if (state.arena.objective) {
          const objective = state.arena.objective;
          const cx = ox + (objective.position.x + .5) * cell;
          const cy = oy + (objective.position.y + .5) * cell;
          context.shadowColor = "#62e6ff"; context.shadowBlur = cell * 1.5;
          context.strokeStyle = "rgba(98, 230, 255, .85)"; context.lineWidth = Math.max(1.2, cell * .12);
          context.beginPath(); context.arc(cx, cy, cell * (objective.radius + .2), 0, Math.PI * 2); context.stroke();
          context.fillStyle = "#d7fbff"; context.beginPath(); context.arc(cx, cy, Math.max(2, cell * .3), 0, Math.PI * 2); context.fill();
        }
      }

      context.shadowBlur = 0;
      state.obstacles.forEach((point) => {
        const x = ox + point.x * cell;
        const y = oy + point.y * cell;
        context.fillStyle = "#222c28";
        context.strokeStyle = "rgba(167, 199, 185, .22)";
        roundedRect(context, x + cell * .08, y + cell * .08, cell * .84, cell * .84, Math.max(1, cell * .16));
        context.fill(); context.stroke();
      });

      state.food.forEach((food) => {
        const cx = ox + (food.position.x + .5) * cell;
        const cy = oy + (food.position.y + .5) * cell;
        const color = food.kind === "rare" ? "#ffd45b" : "#ff8078";
        context.shadowColor = color; context.shadowBlur = cell * (food.kind === "rare" ? 1.4 : .9);
        context.fillStyle = color;
        context.beginPath(); context.arc(cx, cy, Math.max(1.7, cell * (food.kind === "rare" ? .34 : .25)), 0, Math.PI * 2); context.fill();
        if (food.kind === "rare") {
          context.strokeStyle = "#fff2a8"; context.lineWidth = Math.max(1, cell * .08);
          context.beginPath(); context.arc(cx, cy, Math.max(2.2, cell * .46), 0, Math.PI * 2); context.stroke();
        }
      });

      if (state.powerUp) {
        const meta = POWER_UP_META[state.powerUp.kind];
        const cx = ox + (state.powerUp.position.x + .5) * cell;
        const cy = oy + (state.powerUp.position.y + .5) * cell;
        const pulse = 1 + Math.sin(state.tick * .24) * .16;
        context.shadowColor = meta.color; context.shadowBlur = cell * 1.6;
        context.strokeStyle = meta.color; context.lineWidth = Math.max(1.4, cell * .13);
        context.beginPath(); context.arc(cx, cy, Math.max(3, cell * .42 * pulse), 0, Math.PI * 2); context.stroke();
        if (cell > 8) {
          context.fillStyle = meta.color; context.font = `700 ${Math.max(8, cell * .62)}px system-ui`;
          context.textAlign = "center"; context.textBaseline = "middle"; context.fillText(meta.icon, cx, cy + .3);
        }
      }

      state.snakes.forEach((snake) => {
        context.shadowColor = snake.color; context.shadowBlur = snake.alive ? cell * .72 : 0;
        const alpha = snake.alive ? 1 : .16;
        snake.segments.forEach((segment, index) => {
          const taper = Math.max(.48, 1 - index / Math.max(14, snake.segments.length * 1.8));
          const size = cell * .76 * taper;
          const x = ox + (segment.x + .5) * cell - size / 2;
          const y = oy + (segment.y + .5) * cell - size / 2;
          context.globalAlpha = alpha * Math.max(.42, 1 - index * .025);
          context.fillStyle = index === 0 ? snake.accent : snake.color;
          roundedRect(context, x, y, size, size, Math.max(1.5, size * .3)); context.fill();
          if (index === 0 && snake.alive && cell > 7) {
            context.globalAlpha = 1; context.fillStyle = "#08100d";
            context.beginPath(); context.arc(x + size * .63, y + size * .3, Math.max(1, size * .08), 0, Math.PI * 2); context.fill();
          }
        });
        context.globalAlpha = 1;
        if (snake.buffs.shield > 0 && snake.alive) {
          const head = snake.segments[0];
          context.strokeStyle = "#62e6ff"; context.lineWidth = Math.max(1, cell * .12);
          context.beginPath(); context.arc(ox + (head.x + .5) * cell, oy + (head.y + .5) * cell, cell * .68, 0, Math.PI * 2); context.stroke();
        }
        if (state.arena.leaderId === snake.id && snake.alive) {
          const head = snake.segments[0];
          context.fillStyle = "#ffd45b"; context.font = `700 ${Math.max(9, cell * .8)}px system-ui`;
          context.textAlign = "center"; context.textBaseline = "bottom";
          context.fillText("♛", ox + (head.x + .5) * cell, oy + (head.y - .25) * cell);
        }
      });
      context.shadowBlur = 0;
      context.strokeStyle = "rgba(119, 255, 206, .24)";
      context.lineWidth = 1;
      context.strokeRect(ox + .5, oy + .5, boardWidth - 1, boardHeight - 1);
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [state]);

  return <canvas ref={canvasRef} className={`arena-canvas ${compact ? "compact" : ""}`} aria-label="Snake battle arena" />;
}
