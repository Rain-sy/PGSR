"use strict";
document.querySelectorAll(".result").forEach((card) => {
  const stage = card.querySelector(".comparison");
  const slider = stage.querySelector("input");
  const sample = card.dataset.sample;
  const update = () => {
    stage.style.setProperty("--split", `${slider.value}%`);
    slider.setAttribute("aria-valuetext", `${Math.round(slider.value)}% LR, ${Math.round(100-slider.value)}% Ours`);
  };
  slider.addEventListener("input", update);
  // Pointer mapping spans the entire picture, not the native thumb's inset track.
  let pointer = null;
  function position(event) {
    const bounds = stage.getBoundingClientRect();
    slider.value = Math.max(0, Math.min(100, (event.clientX-bounds.left)/bounds.width*100));
    update();
  }
  slider.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    slider.focus({preventScroll:true});
    pointer = event.pointerId;
    slider.setPointerCapture(pointer);
    position(event);
  });
  slider.addEventListener("pointermove", (event) => { if (pointer === event.pointerId) position(event); });
  const release = () => { pointer = null; };
  slider.addEventListener("pointerup", release);
  slider.addEventListener("pointercancel", release);
  slider.addEventListener("lostpointercapture", release);
  let request = 0;
  card.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", async () => {
      const version = ++request;
      const view = button.dataset.view;
      const urls = ["ours", "lr"].map(kind => `images/${sample}-${view}-${kind}.webp`);
      stage.setAttribute("aria-busy", "true");
      try {
        const images = await Promise.all(urls.map(src => new Promise((resolve, reject) => {
          const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = src;
        })));
        if (version !== request) return;
        ["ours", "lr"].forEach((kind, index) => {
          const img = stage.querySelector(`.${kind}`);
          img.src = urls[index]; img.width = images[index].width; img.height = images[index].height;
          img.alt = `${kind === "lr" ? "LR input" : "PGSR restoration"}, ${view === "full" ? "full image" : "matched detail"}, sample ${sample}`;
        });
        stage.style.aspectRatio = `${images[0].width} / ${images[0].height}`;
        card.querySelectorAll("[data-view]").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
      } catch (error) {
        console.error("Could not load comparison images", error);
      } finally { if (version === request) stage.removeAttribute("aria-busy"); }
    });
  });
  update();
});
