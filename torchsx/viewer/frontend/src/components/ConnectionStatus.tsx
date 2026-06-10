// Server connection / latency indicator. Polls the tiny /api/ping on an interval,
// shows a status dot (green = healthy, amber = slow, red = disconnected, pulsing
// grey = connecting) plus the last measured round-trip in ms.

import { useEffect, useRef, useState } from "react";
import { ping } from "../api";

type Status = "connecting" | "online" | "slow" | "offline";

const SLOW_MS = 300;      // round-trip above this → flagged "slow" (tunnel lag)
const INTERVAL_MS = 3000; // ping cadence

export function ConnectionStatus() {
  const [status, setStatus] = useState<Status>("connecting");
  const [latency, setLatency] = useState<number | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const ms = await ping();
        if (cancelled) return;
        setLatency(ms);
        setStatus(ms > SLOW_MS ? "slow" : "online");
      } catch {
        if (cancelled) return;
        setLatency(null);
        setStatus("offline");
      }
    };
    void tick();
    timer.current = window.setInterval(tick, INTERVAL_MS);
    return () => {
      cancelled = true;
      if (timer.current) clearInterval(timer.current);
    };
  }, []);

  const label =
    status === "offline" ? "disconnected"
    : status === "connecting" ? "connecting…"
    : latency != null ? `${Math.round(latency)} ms` : "—";

  return (
    <div className={`conn ${status}`} title={`server connection: ${status}`}>
      <span className="dot" />
      <span className="conn-label">{label}</span>
      {status === "slow" && <span className="muted">slow</span>}
    </div>
  );
}
