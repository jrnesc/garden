"use client";

import { useEffect, useRef } from "react";

type Particle = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  phase: number;
  shine: boolean;
};

type Props = {
  label?: string;
  className?: string;
};

const BG = "#ffffff";
const INK = "74, 93, 35";

function createParticle(width: number, height: number): Particle {
  return {
    x: Math.random() * width,
    y: Math.random() * height,
    vx: (Math.random() - 0.5) * 0.18,
    vy: (Math.random() - 0.5) * 0.18,
    radius: Math.random() * 1.1 + 0.45,
    phase: Math.random() * Math.PI * 2,
    shine: Math.random() < 0.22,
  };
}

export default function LoadingParticles({ label = "loading", className = "" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const pointerRef = useRef({ x: -1000, y: -1000 });
  const frameRef = useRef(0);
  const timeRef = useRef(0);
  const sizeRef = useRef({ width: 0, height: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const targetCount = (width: number, height: number) =>
      Math.min(420, Math.max(120, Math.floor((width * height) / 2200)));

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      const width = window.innerWidth;
      const height = window.innerHeight;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const previous = sizeRef.current;
      const particles = particlesRef.current;
      const desired = targetCount(width, height);

      if (particles.length === 0) {
        for (let i = 0; i < desired; i += 1) particles.push(createParticle(width, height));
      } else if (previous.width > 0 && previous.height > 0) {
        const sx = width / previous.width;
        const sy = height / previous.height;
        for (const particle of particles) {
          particle.x *= sx;
          particle.y *= sy;
        }
      }

      while (particles.length < desired) particles.push(createParticle(width, height));
      if (particles.length > desired) particles.length = desired;
      sizeRef.current = { width, height };
    };

    const handlePointerMove = (event: PointerEvent) => {
      pointerRef.current = { x: event.clientX, y: event.clientY };
    };
    const handlePointerLeave = () => {
      pointerRef.current = { x: -1000, y: -1000 };
    };

    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerleave", handlePointerLeave);

    const tick = () => {
      const { width, height } = sizeRef.current;
      const particles = particlesRef.current;
      const pointer = pointerRef.current;
      timeRef.current += 0.01;
      const t = timeRef.current;
      const activePointer = pointer.x >= 0 && pointer.y >= 0;
      const lightX = activePointer ? pointer.x : width * 0.5 + Math.sin(t * 0.55) * width * 0.22;
      const lightY = activePointer ? pointer.y : height * 0.48 + Math.cos(t * 0.42) * height * 0.18;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = BG;
      ctx.fillRect(0, 0, width, height);

      const glow = ctx.createRadialGradient(lightX, lightY, 0, lightX, lightY, 320);
      glow.addColorStop(0, "rgba(255, 255, 255, 0.16)");
      glow.addColorStop(1, "rgba(255, 255, 255, 0)");
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, width, height);

      for (let i = 0; i < particles.length; i += 1) {
        const a = particles[i];
        let ax = 0;
        let ay = 0;

        for (let j = i + 1; j < particles.length; j += 1) {
          const b = particles[j];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const distSq = dx * dx + dy * dy;
          if (distSq <= 0 || distSq > 110 * 110) continue;

          const dist = Math.sqrt(distSq);
          const nx = dx / dist;
          const ny = dy / dist;
          if (dist < 20) {
            const force = 0.0032 * (1 - dist / 20);
            ax -= nx * force;
            ay -= ny * force;
            b.vx += nx * force;
            b.vy += ny * force;
          } else {
            const force = 0.00042 * (1 - dist / 110);
            const swirl = 0.00028 * (1 - dist / 110);
            ax += nx * force - ny * swirl;
            ay += ny * force + nx * swirl;
            b.vx -= nx * force - ny * swirl;
            b.vy -= ny * force + nx * swirl;
          }
        }

        const lx = lightX - a.x;
        const ly = lightY - a.y;
        const lightDistSq = lx * lx + ly * ly;
        if (lightDistSq < 170 * 170 && lightDistSq > 1) {
          const lightDist = Math.sqrt(lightDistSq);
          const pull = activePointer ? 0.0009 : 0.00035;
          ax += (lx / lightDist) * pull * (1 - lightDist / 170);
          ay += (ly / lightDist) * pull * (1 - lightDist / 170);
        }

        a.vx = (a.vx + ax + Math.sin(t + a.phase) * 0.00035) * 0.992;
        a.vy = (a.vy + ay + Math.cos(t * 0.8 + a.phase) * 0.00035) * 0.992;
        const speed = Math.hypot(a.vx, a.vy);
        if (speed > 0.55) {
          a.vx = (a.vx / speed) * 0.55;
          a.vy = (a.vy / speed) * 0.55;
        }

        a.x += a.vx;
        a.y += a.vy;
        if (a.x < 8 || a.x > width - 8) a.vx *= -0.9;
        if (a.y < 8 || a.y > height - 8) a.vy *= -0.9;
        a.x = Math.max(8, Math.min(width - 8, a.x));
        a.y = Math.max(8, Math.min(height - 8, a.y));
      }

      for (const p of particles) {
        const dx = p.x - lightX;
        const dy = p.y - lightY;
        const light = Math.max(0, 1 - Math.hypot(dx, dy) / 260);
        const pulse = Math.sin(t * 2 + p.phase) * 0.5 + 0.5;
        const opacity = 0.24 + pulse * 0.18 + light * 0.42;
        ctx.fillStyle = `rgba(${INK}, ${opacity})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fill();

        if (p.shine) {
          const len = 2.5 + light * 4;
          ctx.strokeStyle = `rgba(${INK}, ${opacity * 0.55})`;
          ctx.lineWidth = 0.6;
          ctx.beginPath();
          ctx.moveTo(p.x - len, p.y);
          ctx.lineTo(p.x + len, p.y);
          ctx.moveTo(p.x, p.y - len);
          ctx.lineTo(p.x, p.y + len);
          ctx.stroke();
        }
      }

      frameRef.current = requestAnimationFrame(tick);
    };

    frameRef.current = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(frameRef.current);
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerleave", handlePointerLeave);
    };
  }, []);

  return (
    <div className={`absolute inset-0 overflow-hidden ${className}`}>
      <canvas ref={canvasRef} className="absolute inset-0 block h-full w-full" />
      <div className="absolute inset-x-0 bottom-10 flex justify-center">
        <div className="rounded-full border border-[#4A5D23]/15 bg-white/70 px-4 py-2 text-[12px] lowercase tracking-[0.24em] text-[#4A5D23] backdrop-blur-md">
          {label}
        </div>
      </div>
    </div>
  );
}
