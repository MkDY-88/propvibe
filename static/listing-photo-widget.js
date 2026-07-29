// PropVibe - Landing Page Photo Widget
// =====================================
//
// Isolated add-on for static/listing.html: shows the property photo the
// lead's post actually used, with a 3x3 time-of-day x weather toggle. Kept
// in its own file (loaded via a single <script src> tag) rather than
// inlined into listing.html, so it never conflicts with edits to that
// page's own script.
//
// Renders nothing if there's no tracking id, or the tracking id has no
// photo_filename recorded against it (e.g. an older post, or one published
// before this feature existed) - matches the page's existing showGeneric()
// graceful-degradation philosophy: absence of data means the widget quietly
// doesn't appear, never an error.

(function () {
  const trackingId = new URLSearchParams(window.location.search).get("tid") || "";
  if (!trackingId) return;

  const TIMES = ["morning", "evening", "night"];
  const WEATHERS = ["sunny", "cloudy", "rainy"];

  let photoFilename = null;
  let listing = null; // {price, address, features_text} from /listing-info

  // Reuses listing.html's existing --navy/--line CSS variables so the widget
  // matches the page, but the class names themselves (.toggle-cell, .active)
  // don't exist in listing.html's stylesheet - inject a small scoped block
  // once rather than editing that file's <style>.
  function injectStyles() {
    if (document.getElementById("photo-widget-styles")) return;
    const style = document.createElement("style");
    style.id = "photo-widget-styles";
    style.textContent = `
      #photo-widget .toggle-cell {
        font-size: 12px; padding: 8px 4px; border: 1px solid var(--line); border-radius: 6px;
        background: #eef1f6; color: var(--navy); cursor: pointer; text-transform: capitalize;
      }
      #photo-widget .toggle-cell.active { background: var(--navy); color: #fff; }
      #photo-widget .toggle-cell:disabled { opacity: 0.55; cursor: not-allowed; }
    `;
    document.head.appendChild(style);
  }

  function markup() {
    const cells = TIMES.flatMap((t) =>
      WEATHERS.map(
        (w) =>
          `<button type="button" class="toggle-cell" data-time="${t}" data-weather="${w}">${t} &middot; ${w}</button>`
      )
    ).join("");

    return `<section class="card" id="photo-widget">
      <h2 class="chat-title">Photos</h2>
      <img id="widget-photo" alt="Property photo"
           style="width:100%;border-radius:8px;border:1px solid var(--line);display:block;margin-bottom:12px;" />
      <div class="toggle-grid" id="widget-toggle-grid"
           style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;">${cells}</div>
      <p class="generic-copy" style="font-size:12px;margin-top:10px;">Lighting/weather images are AI-simulated.</p>
    </section>`;
  }

  async function init() {
    let payload;
    try {
      const resp = await fetch(`/listing-info/${encodeURIComponent(trackingId)}`);
      payload = await resp.json();
      if (!resp.ok || !payload.found || !payload.photo_filename) return;
    } catch (err) {
      return; // no photo to show - fail silently, same as the rest of this page
    }

    photoFilename = payload.photo_filename;
    listing = payload;

    injectStyles();
    document.querySelector("main").insertAdjacentHTML("afterbegin", markup());
    document.getElementById("widget-photo").src = `/pool-photo/${encodeURIComponent(photoFilename)}`;
    document.getElementById("widget-toggle-grid").addEventListener("click", onToggle);
  }

  async function onToggle(event) {
    const btn = event.target.closest(".toggle-cell");
    if (!btn) return;

    const cells = [...document.querySelectorAll("#widget-toggle-grid .toggle-cell")];
    cells.forEach((c) => (c.disabled = true));
    const original = btn.textContent;
    btn.textContent = "Generating…";

    try {
      const resp = await fetch("/toggle-photo-condition", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          photo_filename: photoFilename,
          time_of_day: btn.dataset.time,
          weather: btn.dataset.weather,
          price: listing.price,
          location: listing.address,
          features: listing.features_text,
        }),
      });
      const payload = await resp.json().catch(() => ({}));
      if (!resp.ok) return; // widget stays on the previously-shown image

      document.getElementById("widget-photo").src = "data:image/png;base64," + payload.poster_base64;
      cells.forEach((c) => c.classList.toggle("active", c === btn));
    } catch (err) {
      // network hiccup - widget stays on the previously-shown image
    } finally {
      btn.textContent = original;
      cells.forEach((c) => (c.disabled = false));
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
