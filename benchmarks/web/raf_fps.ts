export function startFpsMeter(onUpdate: (fps: number) => void) {
  let last = performance.now();
  let frames = 0;

  function tick(now: number) {
    frames++;
    const dt = now - last;
    if (dt >= 1000) {
      const fps = Math.round((frames * 1000) / dt);
      onUpdate(fps);
      frames = 0;
      last = now;
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
