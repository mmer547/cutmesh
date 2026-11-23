// Boundary-layer switches: one per figure and one for all of them.
(function () {
  var figs = document.querySelectorAll("figure.vfig");
  var all = document.getElementById("layer-switch");
  if (!all || !figs.length) return;
  var globalSwitch = all.querySelector(".vswitch");

  function make(onSet) {
    var g = globalSwitch.cloneNode(true);
    g.setAttribute("aria-label", all.dataset.figureLabel);
    g.addEventListener("click", function (e) {
      var b = e.target.closest("button");
      if (b) onSet(b.dataset.set === "layers");
    });
    return g;
  }
  function press(group, on) {
    group.querySelectorAll("button").forEach(function (b) {
      b.setAttribute("aria-pressed", String((b.dataset.set === "layers") === on));
    });
  }
  function setFig(f, on) {
    f.classList.toggle("layers-on", on);
    press(f.querySelector(".vswitch"), on);
  }
  figs.forEach(function (f) {
    f.insertBefore(make(function (on) { setFig(f, on); }), f.firstChild);
  });
  all.hidden = false;
  globalSwitch.addEventListener("click", function (e) {
    var b = e.target.closest("button");
    if (!b) return;
    var on = b.dataset.set === "layers";
    press(globalSwitch, on);
    figs.forEach(function (f) { setFig(f, on); });
  });
})();
