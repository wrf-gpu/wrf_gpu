/* wrf_gpu User's Guide — client-side search + nav. Vanilla JS, no dependencies. */
(function () {
  "use strict";

  var BASE = document.documentElement.getAttribute("data-base") || "";
  var index = null;

  /* ---------- tokenization ---------- */
  function tokenize(s) {
    return (s || "")
      .toLowerCase()
      .replace(/[`*_>#|]/g, " ")
      .split(/[^a-z0-9]+/)
      .filter(function (t) { return t.length > 1; });
  }

  function loadIndex(cb) {
    if (index) { cb(index); return; }
    var xhr = new XMLHttpRequest();
    xhr.open("GET", BASE + "_assets/search-index.json", true);
    xhr.onreadystatechange = function () {
      if (xhr.readyState === 4) {
        try {
          index = JSON.parse(xhr.responseText);
          index.forEach(function (e) {
            e._all = {};
            tokenize(e.title).forEach(function (t) { e._all[t] = (e._all[t] || 0) + 8; });
            tokenize(e.crumb || "").forEach(function (t) { e._all[t] = (e._all[t] || 0) + 3; });
            tokenize(e.text).forEach(function (t) { e._all[t] = (e._all[t] || 0) + 1; });
            e._exact = tokenize(e.title).join(" ");
            e._hay = (e.title + " " + (e.crumb || "") + " " + e.text).toLowerCase();
          });
          cb(index);
        } catch (err) { /* index missing: search silently disabled */ }
      }
    };
    xhr.send();
  }

  /* ---------- ranking ---------- */
  function search(query, limit) {
    if (!index) return [];
    var qTokens = tokenize(query);
    if (!qTokens.length) return [];
    var phrase = query.toLowerCase().trim();
    var results = [];
    index.forEach(function (e) {
      var score = 0, matchedAll = true;
      qTokens.forEach(function (qt) {
        var best = 0;
        for (var tok in e._all) {
          if (tok === qt) { best = Math.max(best, e._all[tok] * 2); }
          else if (tok.indexOf(qt) === 0) { best = Math.max(best, e._all[tok]); }
          else if (tok.indexOf(qt) !== -1) { best = Math.max(best, e._all[tok] * 0.4); }
        }
        if (best === 0) matchedAll = false;
        score += best;
      });
      if (score === 0) return;
      if (matchedAll) score *= 3;
      if (qTokens.length > 1 && e._hay.indexOf(phrase) !== -1) score *= 2;
      if (e._exact === qTokens.join(" ")) score += 50;
      results.push({ e: e, score: score });
    });
    results.sort(function (a, b) { return b.score - a.score; });
    return results.slice(0, limit || 12).map(function (r) { return r.e; });
  }

  function escapeHtml(s) {
    return s.replace(/[&<>]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; });
  }

  function snippet(text, query) {
    var qTokens = tokenize(query);
    var low = text.toLowerCase();
    var pos = -1;
    for (var i = 0; i < qTokens.length && pos === -1; i++) pos = low.indexOf(qTokens[i]);
    if (pos === -1) pos = 0;
    var start = Math.max(0, pos - 40);
    var snip = (start > 0 ? "…" : "") + text.slice(start, start + 170) +
               (text.length > start + 170 ? "…" : "");
    // Wrap matches in ASCII sentinels, HTML-escape, then swap sentinels for <mark>.
    var OPEN = "", CLOSE = "";
    qTokens.forEach(function (qt) {
      var re = new RegExp("(" + qt.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig");
      snip = snip.replace(re, OPEN + "$1" + CLOSE);
    });
    return escapeHtml(snip).split(OPEN).join("<mark>").split(CLOSE).join("</mark>");
  }

  /* ---------- sidebar live search ---------- */
  function wireSidebarSearch() {
    var input = document.getElementById("search-input");
    var box = document.getElementById("search-results");
    if (!input || !box) return;
    var sel = -1;

    function render(items, q) {
      sel = -1;
      if (!q) { box.classList.remove("open"); box.innerHTML = ""; return; }
      if (!items.length) {
        box.innerHTML = '<div class="sr-empty">No matches for “' + escapeHtml(q) +
          '”. Try broader terms.</div>';
        box.classList.add("open"); return;
      }
      box.innerHTML = items.map(function (e) {
        return '<a href="' + BASE + e.url + '">' +
          '<span class="sr-title">' + escapeHtml(e.title) + '</span>' +
          '<span class="sr-crumb"> · ' + escapeHtml(e.crumb || "") + '</span>' +
          '<div class="sr-snip">' + snippet(e.text, q) + "</div></a>";
      }).join("");
      box.classList.add("open");
    }

    input.addEventListener("input", function () {
      var q = input.value.trim();
      loadIndex(function () { render(search(q, 8), q); });
    });
    input.addEventListener("keydown", function (ev) {
      var links = box.querySelectorAll("a");
      if (ev.key === "ArrowDown") { ev.preventDefault(); sel = Math.min(sel + 1, links.length - 1); }
      else if (ev.key === "ArrowUp") { ev.preventDefault(); sel = Math.max(sel - 1, 0); }
      else if (ev.key === "Enter") {
        if (sel >= 0 && links[sel]) { window.location.href = links[sel].getAttribute("href"); }
        else { window.location.href = BASE + "search.html?q=" + encodeURIComponent(input.value.trim()); }
        return;
      } else if (ev.key === "Escape") { box.classList.remove("open"); input.blur(); return; }
      links.forEach(function (l, i) { l.classList.toggle("sel", i === sel); });
      if (links[sel]) links[sel].scrollIntoView({ block: "nearest" });
    });
    document.addEventListener("click", function (ev) {
      if (!box.contains(ev.target) && ev.target !== input) box.classList.remove("open");
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "/" && document.activeElement !== input &&
          !/input|textarea/i.test(document.activeElement.tagName)) {
        ev.preventDefault(); input.focus();
      }
    });
  }

  /* ---------- dedicated results page ---------- */
  function wireResultsPage() {
    var container = document.getElementById("full-results");
    if (!container) return;
    var params = new URLSearchParams(window.location.search);
    var q = params.get("q") || "";
    var qInput = document.getElementById("full-search-input");
    if (qInput) {
      qInput.value = q;
      qInput.form.addEventListener("submit", function (ev) {
        ev.preventDefault();
        window.location.search = "?q=" + encodeURIComponent(qInput.value.trim());
      });
    }
    var heading = document.getElementById("results-heading");
    loadIndex(function () {
      var items = search(q, 40);
      if (heading) heading.textContent = q
        ? items.length + " result" + (items.length === 1 ? "" : "s") + " for “" + q + "”"
        : "Type a query to search the guide.";
      container.innerHTML = items.map(function (e) {
        return '<div class="result">' +
          '<a class="result-link" href="' + BASE + e.url + '">' + escapeHtml(e.title) + '</a>' +
          '<div class="sr-crumb">' + escapeHtml(e.crumb || "") + "</div>" +
          '<div class="sr-snip">' + snippet(e.text, q) + "</div></div>";
      }).join("") || (q ? "<p>No matches. Try broader or fewer terms.</p>" : "");
    });
  }

  /* ---------- mobile nav ---------- */
  function wireNav() {
    var toggle = document.querySelector(".menu-toggle");
    var scrim = document.querySelector(".scrim");
    if (toggle) toggle.addEventListener("click", function () { document.body.classList.toggle("nav-open"); });
    if (scrim) scrim.addEventListener("click", function () { document.body.classList.remove("nav-open"); });
    var active = document.querySelector("nav.toc a.active");
    if (active) active.scrollIntoView({ block: "center" });
  }

  document.addEventListener("DOMContentLoaded", function () {
    wireNav();
    wireSidebarSearch();
    wireResultsPage();
  });
})();
