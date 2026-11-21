// meshDict form: builds the form from window.MESHDICT_SCHEMA (generated from cutmesh/schema.py),
// keeps the meshDict being edited in `state`, and shows it as JSON with the problems found.
// No libraries, no network: STL and meshDict files are read in the browser only.
(function () {
  "use strict";
  var S = window.MESHDICT_SCHEMA, EXAMPLES = window.MESHDICT_EXAMPLES || {};
  var P = S.properties;
  var STORE = "cutmesh.meshdict.v1";

  var state = {};            // the meshDict being edited (only keys the user set)
  var stl = null;            // {file, format, solids, ntri, lo, hi}
  var showAdvanced = false, withDefaults = false;

  // ---------------------------------------------------------------- small helpers
  function el(tag, attrs) {
    var e = document.createElement(tag);
    if (attrs) {
      for (var k in attrs) {
        var v = attrs[k];
        if (v == null || v === false) continue;
        if (k === "text") e.textContent = v;
        else if (k === "class") e.className = v;
        else if (k.slice(0, 2) === "on") e.addEventListener(k.slice(2), v);
        else e.setAttribute(k, v === true ? "" : v);
      }
    }
    for (var i = 2; i < arguments.length; i++) {
      var c = arguments[i];
      if (c == null || c === false) continue;
      if (Array.isArray(c)) c.forEach(function (x) { if (x) e.appendChild(x); });
      else e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return e;
  }
  function types(p) { return Array.isArray(p.type) ? p.type : [p.type]; }
  function isObj(v) { return v !== null && typeof v === "object" && !Array.isArray(v); }
  function isComment(k) { return k.charAt(0) === "_" || k.slice(0, 2) === "//"; }
  function clone(v) { return v === undefined ? undefined : JSON.parse(JSON.stringify(v)); }
  function same(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
  function show(v) { return v === undefined ? "" : JSON.stringify(v); }
  function pathName(path) {
    return path.map(function (k, i) {
      return typeof k === "number" ? "[" + k + "]" : (i ? "." : "") + k;
    }).join("");
  }
  function getAt(path) {
    var v = state;
    for (var i = 0; i < path.length; i++) {
      if (v == null || typeof v !== "object") return undefined;
      v = v[path[i]];
    }
    return v;
  }
  function setAt(path, val) {
    var v = state;
    for (var i = 0; i < path.length - 1; i++) {
      if (v[path[i]] == null || typeof v[path[i]] !== "object") {
        v[path[i]] = typeof path[i + 1] === "number" ? [] : {};
      }
      v = v[path[i]];
    }
    var last = path[path.length - 1];
    if (val === undefined) {
      if (Array.isArray(v)) v.splice(last, 1); else delete v[last];
    } else {
      v[last] = val;
    }
  }
  // "0.01" -> 0.01, "" -> undefined, "abc" -> "abc" (kept, so that the check reports it)
  function parseNum(s) {
    s = s.trim();
    if (s === "") return undefined;
    if (/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(s)) return Number(s);
    return s;
  }
  function fmtNum(x) {
    if (!isFinite(x)) return String(x);
    var a = Math.abs(x);
    return a !== 0 && (a < 1e-3 || a >= 1e6) ? x.toExponential(3) : String(+x.toPrecision(6));
  }

  // ---------------------------------------------------------------- changes
  var saveTimer = null;
  function changed() {
    update();
    clearTimeout(saveTimer);
    saveTimer = setTimeout(function () {
      try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) { /* storage off */ }
    }, 300);
  }
  function rerender() { render(); changed(); }

  // ---------------------------------------------------------------- form
  function visible(p, path) {
    return showAdvanced || !p["x-advanced"] || getAt(path) !== undefined;
  }

  function label(key, p, required) {
    return el("div", { class: "mdf-label" },
      el("code", { text: key }),
      required ? el("span", { class: "mdf-req", text: "必須" }) : null,
      p["x-advanced"] ? el("span", { class: "mdf-adv", text: "詳細" }) : null);
  }
  function help(p) {
    var bits = [p.description || ""];
    if (p["x-unit"]) bits.push("［" + p["x-unit"] + "］");
    var t = el("div", { class: "mdf-help" }, bits.join(" "));
    if ("default" in p && !(isObj(p.default) && !Object.keys(p.default).length) &&
        !(Array.isArray(p.default) && !p.default.length)) {
      t.appendChild(el("span", { class: "mdf-default" }, "既定: ", el("code", { text: show(p.default) })));
    }
    return t;
  }
  function row(key, p, path, required, input) {
    return el("div", { class: "mdf-field", "data-path": pathName(path) },
      label(key, p, required), el("div", { class: "mdf-input" }, input), help(p));
  }

  function textInput(path, p, opts) {
    opts = opts || {};
    var v = getAt(path);
    return el("input", {
      type: "text", class: "mdf-text" + (opts.num ? " mdf-num" : ""),
      inputmode: opts.num ? "decimal" : null, list: opts.list || null,
      value: v === undefined || v === null ? "" : (typeof v === "string" ? v : String(v)),
      placeholder: opts.placeholder != null ? opts.placeholder :
        ("default" in p && p.default !== null && typeof p.default !== "object" ? String(p.default) : ""),
      "aria-label": pathName(path),
      oninput: function (e) {
        var s = e.target.value;
        setAt(path, opts.num ? parseNum(s) : (s === "" ? undefined : s));
        changed();
      }
    });
  }

  function selectInput(path, p, options) {
    // options: [[value, label]]; the first option (value undefined) keeps the default
    var v = getAt(path);
    var sel = el("select", {
      "aria-label": pathName(path),
      onchange: function (e) {
        var o = options[e.target.selectedIndex];
        setAt(path, clone(o[0]));
        changed();
      }
    });
    var found = false;
    options.forEach(function (o, i) {
      var opt = el("option", { text: o[1] });
      if ((v === undefined && o[0] === undefined) || (v !== undefined && same(v, o[0]))) {
        opt.selected = true; found = true;
      }
      sel.appendChild(opt);
    });
    if (!found && v !== undefined) {        // a value the list does not have (from a loaded file)
      options.push([v, show(v) + "（不明な値）"]);
      sel.appendChild(el("option", { text: show(v) + "（不明な値）", selected: true }));
    }
    return sel;
  }

  function vecInput(path, p, n) {
    var v = getAt(path);
    var box = el("div", { class: "mdf-vec" });
    var names = n === 3 ? ["x", "y", "z"] : ["z0", "z1"];
    var dflt = Array.isArray(p.default) ? p.default : null;
    for (var i = 0; i < n; i++) {
      (function (i) {
        box.appendChild(el("input", {
          type: "text", class: "mdf-text mdf-num", inputmode: "decimal",
          "aria-label": pathName(path) + " " + names[i], placeholder: dflt ? String(dflt[i]) : names[i],
          value: Array.isArray(v) && v[i] !== undefined && v[i] !== null ? String(v[i]) : "",
          oninput: function () {
            var vals = Array.prototype.map.call(box.querySelectorAll("input"), function (x) {
              return parseNum(x.value);
            });
            setAt(path, vals.every(function (x) { return x === undefined; }) ? undefined :
              vals.map(function (x) { return x === undefined ? null : x; }));
            changed();
          }
        }));
      })(i);
    }
    return box;
  }

  function jsonInput(path, p) {
    var v = getAt(path);
    var ta = el("textarea", {
      class: "mdf-textarea", rows: 4, spellcheck: "false", "aria-label": pathName(path),
      placeholder: p.type === "array" ? '[{"rOuter": 0.06, "nr": 2}]' : '{"halfWidth": 0.0375, "n": 6}',
      oninput: function () {
        var s = ta.value.trim();
        if (s === "") { setAt(path, undefined); ta.classList.remove("mdf-bad"); changed(); return; }
        try { setAt(path, JSON.parse(s)); ta.classList.remove("mdf-bad"); }
        catch (e) { setAt(path, { "!json": s }); ta.classList.add("mdf-bad"); }
        changed();
      }
    });
    ta.value = v === undefined ? "" : (isObj(v) && "!json" in v ? v["!json"] : JSON.stringify(v));
    return ta;
  }

  // solid name -> value (surfaceRefinement, patchTypes)
  function mapInput(path, p, widget) {
    var box = el("div", { class: "mdf-map" });
    function draw() {
      box.textContent = "";
      var m = getAt(path) || {};
      var keys = Object.keys(m);
      keys.forEach(function (k) {
        var name = el("input", {
          type: "text", class: "mdf-text", value: k, "aria-label": pathName(path) + " の名前",
          list: widget === "solidLevels" ? "mdf-solids-default" : "mdf-solids", placeholder: "solid 名",
          onchange: function (e) {
            var nk = e.target.value.trim();
            var cur = getAt(path) || {};
            if (!nk || nk === k) { e.target.value = k; return; }
            var out = {};
            Object.keys(cur).forEach(function (kk) { out[kk === k ? nk : kk] = cur[kk]; });
            setAt(path, out); draw(); changed();
          }
        });
        var val = el("input", {
          type: "text", class: "mdf-text" + (widget === "solidLevels" ? " mdf-num" : ""),
          inputmode: widget === "solidLevels" ? "numeric" : null,
          list: widget === "solidTypes" ? "mdf-ptypes" : null,
          placeholder: widget === "solidLevels" ? "レベル" : "wall",
          value: m[k] === undefined ? "" : String(m[k]), "aria-label": pathName(path) + "." + k,
          oninput: function (e) {
            var cur = getAt(path) || {};
            cur[k] = widget === "solidLevels" ? parseNum(e.target.value) : e.target.value.trim();
            if (cur[k] === undefined) cur[k] = "";
            setAt(path, cur); changed();
          }
        });
        var del = el("button", {
          type: "button", class: "mdf-x", title: "この行を消す", "aria-label": k + " を消す",
          onclick: function () {
            var cur = getAt(path) || {};
            delete cur[k];
            setAt(path, Object.keys(cur).length ? cur : undefined); draw(); changed();
          }
        }, "×");
        box.appendChild(el("div", { class: "mdf-maprow" }, name, val, del));
      });
      var add = el("div", { class: "mdf-mapadd" },
        el("button", {
          type: "button", class: "mdf-link", onclick: function () {
            var cur = getAt(path) || {};
            var base = widget === "solidLevels" && !("default" in cur) ? "default" : "", nk = base, i = 1;
            if (!nk) {
              var unused = (stl ? stl.solids : []).filter(function (s) { return !(s in cur); });
              nk = unused.length ? unused[0] : "solid";
              while (nk in cur) nk = "solid" + (i++);
            }
            cur[nk] = widget === "solidLevels" ? 1 : "patch";
            setAt(path, cur); draw(); changed();
            var ins = box.querySelectorAll(".mdf-maprow input");
            if (ins.length) ins[ins.length - 2].focus();
          }
        }, "＋ 行を足す"));
      if (stl && stl.solids.length && widget === "solidLevels") {
        var missing = stl.solids.filter(function (s) { return !(s in m); });
        if (missing.length) {
          add.appendChild(el("button", {
            type: "button", class: "mdf-link", onclick: function () {
              var cur = getAt(path) || {};
              if (!("default" in cur)) cur["default"] = 1;
              missing.forEach(function (s) { cur[s] = cur["default"]; });
              setAt(path, cur); draw(); changed();
            }
          }, "＋ STL の solid をすべて足す"));
        }
      }
      box.appendChild(add);
    }
    draw();
    return box;
  }

  // list of solid names (nonRotatingPatches, addLayers.patches)
  function listInput(path, p) {
    var v = getAt(path);
    var box = el("div", { class: "mdf-list" });
    var inp = el("input", {
      type: "text", class: "mdf-text", "aria-label": pathName(path),
      placeholder: p.default === null ? "空欄: wall のパッチすべて" : "名前をカンマ区切りで",
      value: Array.isArray(v) ? v.join(", ") : "",
      oninput: function () { sync(); }
    });
    function sync() {
      var names = inp.value.split(",").map(function (s) { return s.trim(); }).filter(Boolean);
      setAt(path, names.length ? names : undefined);
      changed(); chips();
    }
    var chipBox = el("div", { class: "mdf-chips" });
    function chips() {
      chipBox.textContent = "";
      if (!stl || !stl.solids.length) return;
      var cur = inp.value.split(",").map(function (s) { return s.trim(); });
      stl.solids.forEach(function (s) {
        var on = cur.indexOf(s) >= 0;
        chipBox.appendChild(el("button", {
          type: "button", class: "mdf-chip", "aria-pressed": on ? "true" : "false",
          onclick: function () {
            var names = inp.value.split(",").map(function (x) { return x.trim(); }).filter(Boolean);
            var i = names.indexOf(s);
            if (i >= 0) names.splice(i, 1); else names.push(s);
            inp.value = names.join(", "); sync();
          }
        }, s));
      });
    }
    box.appendChild(inp); box.appendChild(chipBox); chips();
    return box;
  }

  // a nullable number whose default is not null: input + "null" switch
  function nullableNum(path, p) {
    var v = getAt(path);
    var box = el("div", { class: "mdf-inline" });
    var inp = textInput(path, p, { num: true });
    var cb = el("input", {
      type: "checkbox", checked: v === null,
      onchange: function (e) {
        inp.disabled = e.target.checked;
        if (e.target.checked) { inp.value = ""; setAt(path, null); } else setAt(path, undefined);
        changed();
      }
    });
    inp.disabled = v === null;
    box.appendChild(inp);
    box.appendChild(el("label", { class: "mdf-check" }, cb, " null（無効）"));
    return box;
  }

  function scalarField(key, p, path, required) {
    var t = types(p), w = p["x-widget"];
    if (w === "vec3") return row(key, p, path, required, vecInput(path, p, 3));
    if (t[0] === "array" && p.items && p.items.type === "number" && p.minItems === p.maxItems) {
      return row(key, p, path, required, vecInput(path, p, p.minItems));
    }
    if (w === "solidLevels" || w === "solidTypes") return row(key, p, path, required, mapInput(path, p, w));
    if (w === "solidList") return row(key, p, path, required, listInput(path, p));
    if (w === "json") return row(key, p, path, required, jsonInput(path, p));
    if (w === "stl") return row(key, p, path, required, textInput(path, p, { placeholder: "model.stl" }));
    if (p["enum"]) {
      var opts = [[undefined, "既定（" + show(p.default) + "）"]];
      p["enum"].forEach(function (x) { opts.push([x, typeof x === "string" ? x : show(x)]); });
      return row(key, p, path, required, selectInput(path, p, opts));
    }
    if (t[0] === "boolean" && t.length === 1) {
      return row(key, p, path, required, selectInput(path, p, [
        [undefined, "既定（" + show(p.default) + "）"], [true, "true"], [false, "false"]]));
    }
    if (t.indexOf("number") >= 0 || t.indexOf("integer") >= 0) {
      if (t.indexOf("null") >= 0 && p.default !== null && "default" in p) {
        return row(key, p, path, required, nullableNum(path, p));
      }
      return row(key, p, path, required, textInput(path, p, { num: true }));
    }
    return row(key, p, path, required, textInput(path, p));
  }

  function fields(props, path, required, box) {
    Object.keys(props).forEach(function (k) {
      var p = props[k], sub = path.concat([k]);
      if (!visible(p, sub)) return;
      box.appendChild(field(k, p, sub, required.indexOf(k) >= 0));
    });
    return box;
  }

  // an object that may be left out: a switch, then its fields
  function optionalObject(key, p, path, required) {
    var v = getAt(path);
    var on = v !== undefined;
    var t = types(p);
    var wrap = el("div", { class: "mdf-field mdf-sub", "data-path": pathName(path) });
    var head = el("div", { class: "mdf-subhead" }, label(key, p, required));
    if (t.indexOf("boolean") >= 0) {        // autoGapRefinement: off / true / settings
      var modes = [[undefined, "使わない"], [true, "true（既定値で使う）"], [{}, "値を指定する"]];
      var cur = v === undefined ? 0 : v === true ? 1 : isObj(v) ? 2 : -1;
      var sel = el("select", { "aria-label": pathName(path), onchange: function (e) {
        var i = e.target.selectedIndex;
        if (i < modes.length) { setAt(path, clone(modes[i][0])); rerender(); }
      } });
      modes.forEach(function (m, i) { sel.appendChild(el("option", { text: m[1], selected: i === cur })); });
      if (cur < 0) sel.appendChild(el("option", { text: show(v) + "（不明な値）", selected: true }));
      head.appendChild(sel);
      on = isObj(v);
    } else if (!required) {
      head.appendChild(el("label", { class: "mdf-check" }, el("input", {
        type: "checkbox", checked: on,
        onchange: function (e) { setAt(path, e.target.checked ? {} : undefined); rerender(); }
      }), " 使う"));
    }
    wrap.appendChild(head);
    wrap.appendChild(help(p));
    if (on || required) {
      if (!isObj(getAt(path))) setAt(path, {});
      wrap.appendChild(fields(p.properties, path, p.required || [], el("div", { class: "mdf-subbody" })));
    }
    return wrap;
  }

  function objectList(key, p, path, itemTitle) {
    var wrap = el("div", { class: "mdf-field mdf-listfield", "data-path": pathName(path) });
    wrap.appendChild(el("div", { class: "mdf-subhead" }, label(key, p, false)));
    wrap.appendChild(help(p));
    var arr = getAt(path);
    if (!Array.isArray(arr)) arr = [];
    arr.forEach(function (item, i) {
      var ip = path.concat([i]);
      var card = el("div", { class: "mdf-card" });
      var title = itemTitle(item, i);
      card.appendChild(el("div", { class: "mdf-cardhead" }, el("strong", { text: title }),
        el("button", {
          type: "button", class: "mdf-x", title: "消す", "aria-label": title + " を消す",
          onclick: function () {
            var a = getAt(path); a.splice(i, 1);
            setAt(path, a.length ? a : undefined); rerender();
          }
        }, "×")));
      if (key === "rotatingZones") card.appendChild(zoneShape(ip));
      var props = {};
      Object.keys(p.items.properties).forEach(function (k) {
        if (key === "rotatingZones") {
          var useStl = isObj(item) && "stl" in item;
          if (k === "cylinder" && useStl) return;
          if ((k === "stl" || k === "origin" || k === "axis") && !useStl) return;
        }
        props[k] = p.items.properties[k];
      });
      var req = (p.items.required || []).slice();
      if (key === "rotatingZones") req.push(isObj(item) && "stl" in item ? "stl" : "cylinder");
      fields(props, ip, req, card);
      wrap.appendChild(card);
    });
    wrap.appendChild(el("button", {
      type: "button", class: "btn mdf-small", onclick: function () {
        var a = getAt(path); if (!Array.isArray(a)) a = [];
        a.push(key === "rotatingZones" ? { name: "rotor" + (a.length ? a.length : ""), cylinder: {} } :
          { cylinder: {}, level: 1 });
        setAt(path, a); rerender();
      }
    }, key === "rotatingZones" ? "＋ 回転ゾーンを足す" : "＋ 細分化領域を足す"));
    return wrap;
  }

  function zoneShape(ip) {
    var item = getAt(ip) || {};
    var useStl = "stl" in item;
    var name = "shape-" + ip.join("-");
    function pick(stlMode) {
      var cur = getAt(ip) || {};
      if (stlMode) { delete cur.cylinder; if (!("stl" in cur)) cur.stl = ""; }
      else { delete cur.stl; delete cur.origin; delete cur.axis; if (!("cylinder" in cur)) cur.cylinder = {}; }
      setAt(ip, cur); rerender();
    }
    return el("div", { class: "mdf-shape", role: "radiogroup", "aria-label": "ゾーンの形" },
      el("span", { class: "mdf-shape-label", text: "形:" }),
      el("label", { class: "mdf-check" }, el("input", { type: "radio", name: name, checked: !useStl,
        onchange: function () { pick(false); } }), " 円筒"),
      el("label", { class: "mdf-check" }, el("input", { type: "radio", name: name, checked: useStl,
        onchange: function () { pick(true); } }), " 閉じた STL"));
  }

  function field(key, p, path, required) {
    var t = types(p);
    if (t[0] === "array" && p.items && p.items.type === "object" && p.items.properties) {
      return objectList(key, p, path, key === "rotatingZones" ?
        function (z, i) { return "ゾーン " + (i + 1) + (isObj(z) && z.name ? "（" + z.name + "）" : ""); } :
        function (r, i) { return "領域 " + (i + 1); });
    }
    if (t.indexOf("object") >= 0 && p.properties && p["x-widget"] !== "json") {
      return optionalObject(key, p, path, required);
    }
    return scalarField(key, p, path, required);
  }

  function render() {
    var form = document.getElementById("mdf-form");
    form.textContent = "";
    S["x-groups"].forEach(function (g) {
      var keys = Object.keys(P).filter(function (k) { return P[k]["x-group"] === g.key; });
      var sec = el("section", { class: "mdf-group", id: "g-" + g.key },
        el("h2", {}, g.title, " ", el("a", { class: "mdf-doc", href: "manual.html#" + g.doc, text: "マニュアル" })));
      var shown = 0, hidden = 0;
      keys.forEach(function (k) {
        if (!visible(P[k], [k])) { hidden++; return; }
        sec.appendChild(field(k, P[k], [k], S.required.indexOf(k) >= 0));
        shown++;
      });
      if (!shown) {
        sec.appendChild(el("p", { class: "mdf-hidden-note" },
          "この欄の項目はすべて詳細な設定です（「詳細な設定も表示」で表示）。"));
      } else if (hidden) {
        sec.appendChild(el("p", { class: "mdf-hidden-note" },
          "ほかに詳細な設定が " + hidden + " 項目あります。"));
      }
      form.appendChild(sec);
    });
    var extra = Object.keys(state).filter(function (k) { return !(k in P); });
    if (extra.length) {
      form.appendChild(el("section", { class: "mdf-group" }, el("h2", { text: "そのほかのキー" }),
        el("p", { class: "mdf-help" }, "コメント（_ や // で始まるキー）と、cutmesh の知らないキーはそのまま書き出します: ",
          extra.map(function (k) { return el("code", { text: k }); }).reduce(function (a, c) {
            if (a.length) a.push(document.createTextNode(", ")); a.push(c); return a;
          }, []))));
    }
    updateDatalists();
  }

  function updateDatalists() {
    var solids = stl ? stl.solids : [];
    function fill(id, names) {
      var dl = document.getElementById(id);
      dl.textContent = "";
      names.forEach(function (n) { dl.appendChild(el("option", { value: n })); });
    }
    fill("mdf-solids", solids);
    fill("mdf-solids-default", ["default"].concat(solids));
    fill("mdf-ptypes", S["x-patchTypes"]);
  }

  // ---------------------------------------------------------------- checks
  function typeOk(v, t) {
    switch (t) {
      case "null": return v === null;
      case "boolean": return typeof v === "boolean";
      case "integer": return typeof v === "number" && isFinite(v) && Math.floor(v) === v;
      case "number": return typeof v === "number" && isFinite(v);
      case "string": return typeof v === "string";
      case "array": return Array.isArray(v);
      case "object": return isObj(v);
    }
    return true;
  }
  var TYPE_JA = { "null": "null", "boolean": "true / false", "integer": "整数", "number": "数",
    "string": "文字列", "array": "リスト [...]", "object": "オブジェクト {...}" };

  function check(v, p, path, out) {
    var name = pathName(path);
    if (isObj(v) && "!json" in v) { out.push(["error", name + ": JSON として読めません"]); return; }
    var t = types(p);
    if (!t.some(function (x) { return typeOk(v, x); })) {
      out.push(["error", name + " は" + t.map(function (x) { return TYPE_JA[x]; }).join(" か ") +
        "にしてください（いま " + show(v) + "）"]);
      return;
    }
    if (p["enum"] && !p["enum"].some(function (x) { return same(x, v); })) {
      out.push(["error", name + " は " + p["enum"].map(show).join(", ") + " のどれかです（いま " + show(v) + "）"]);
    }
    if (typeof v === "number") {
      if ("minimum" in p && v < p.minimum) out.push(["error", name + " は " + p.minimum + " 以上です（いま " + v + "）"]);
      if ("exclusiveMinimum" in p && v <= p.exclusiveMinimum) out.push(["error", name + " は " + p.exclusiveMinimum + " より大きくしてください（いま " + v + "）"]);
      if ("maximum" in p && v > p.maximum) out.push(["error", name + " は " + p.maximum + " 以下です（いま " + v + "）"]);
      if ("exclusiveMaximum" in p && v >= p.exclusiveMaximum) out.push(["error", name + " は " + p.exclusiveMaximum + " より小さくしてください（いま " + v + "）"]);
    }
    if (typeof v === "string" && p.minLength && v.length < p.minLength) out.push(["error", name + " が空です"]);
    if (Array.isArray(v)) {
      if (p.minItems === p.maxItems && p.minItems !== undefined && v.length !== p.minItems) {
        out.push(["error", name + " は " + p.minItems + " 個の数です"]);
      }
      if (p.items) v.forEach(function (x, i) {
        if (x === null && p.items.type === "number") out.push(["error", name + "[" + i + "] が空です"]);
        else check(x, p.items, path.concat([i]), out);
      });
    }
    if (isObj(v)) {
      (p.required || []).forEach(function (k) {
        if (!(k in v)) out.push(["error", (name ? name + ": " : "") + k + " が必要です"]);
      });
      Object.keys(v).forEach(function (k) {
        var sub = path.concat([k]);
        if (p.properties && k in p.properties) check(v[k], p.properties[k], sub, out);
        else if (isComment(k)) { /* comment */ }
        else if (isObj(p.additionalProperties)) check(v[k], p.additionalProperties, sub, out);
        else if (p.additionalProperties === false) {
          out.push(["warn", pathName(sub) + " は cutmesh の知らないキーです（無視されます）"]);
        }
      });
    }
  }

  // the checks of config.validate_config that the schema does not express
  function crossChecks(cfg, out) {
    function cyl(c, name) {
      if (isObj(c) && Array.isArray(c.point1) && Array.isArray(c.point2) && same(c.point1, c.point2)) {
        out.push(["error", name + ": point1 と point2 が同じです（軸の長さが 0）"]);
      }
    }
    (Array.isArray(cfg.refinementRegions) ? cfg.refinementRegions : []).forEach(function (r, i) {
      if (isObj(r)) cyl(r.cylinder, "refinementRegions[" + i + "].cylinder");
    });
    var seen = {};
    (Array.isArray(cfg.rotatingZones) ? cfg.rotatingZones : []).forEach(function (z, i) {
      if (!isObj(z)) return;
      var nm = "rotatingZones[" + i + "]";
      if (("cylinder" in z) === ("stl" in z)) out.push(["error", nm + ": cylinder か stl のどちらか一方を書きます"]);
      if (typeof z.name === "string" && z.name) {
        if (seen[z.name]) out.push(["error", nm + ": ゾーン名 " + show(z.name) + " が重複しています"]);
        seen[z.name] = true;
      }
      cyl(z.cylinder, nm + ".cylinder");
      if (Array.isArray(z.axis) && z.axis.every(function (x) { return x === 0; })) {
        out.push(["error", nm + ".axis が 0 ベクトルです"]);
      }
      if (z.ami === true && z.conforming === false) {
        out.push(["error", nm + ": \"ami\": true は \"conforming\": false と一緒に使えません"]);
      }
    });
    var bm = cfg.baseMesh;
    if (isObj(bm) && bm.type === "cylindrical") {
      if (("core" in bm) !== ("blocks" in bm)) out.push(["error", "baseMesh: core と blocks は両方書くか、両方書かないかです"]);
      if ("core" in bm && ("rOuter" in bm || "cellSize" in bm)) {
        out.push(["error", "baseMesh: rOuter と cellSize は自動配置（core と blocks なし）のときだけ使えます"]);
      }
    }
    if (isObj(bm) && bm.type !== "cylindrical" && Object.keys(bm).some(function (k) { return k !== "type"; })) {
      out.push(["warn", "baseMesh: type が cylindrical でないので、ほかの値は使われません"]);
    }
    if (isObj(bm) && "nz" in bm && "dz" in bm) out.push(["warn", "baseMesh: nz と dz の両方があるときは nz が使われます"]);
    if (stl && stl.solids.length) {       // config.check_names
      var have = {};
      stl.solids.forEach(function (s) { have[s] = true; });
      function names(list, where) {
        list.forEach(function (k) {
          if (!isComment(k) && !have[k]) out.push(["warn", where + " の " + show(k) + " は STL の solid にありません（無視されます）"]);
        });
      }
      if (isObj(cfg.surfaceRefinement)) names(Object.keys(cfg.surfaceRefinement).filter(function (k) { return k !== "default"; }), "surfaceRefinement");
      if (isObj(cfg.patchTypes)) names(Object.keys(cfg.patchTypes), "patchTypes");
      (Array.isArray(cfg.rotatingZones) ? cfg.rotatingZones : []).forEach(function (z, i) {
        if (isObj(z) && Array.isArray(z.nonRotatingPatches)) names(z.nonRotatingPatches, "rotatingZones[" + i + "].nonRotatingPatches");
      });
      if (isObj(cfg.addLayers) && Array.isArray(cfg.addLayers.patches)) names(cfg.addLayers.patches, "addLayers.patches");
    }
    var zs = Array.isArray(cfg.rotatingZones) ? cfg.rotatingZones : [];
    if (zs.some(function (z) { return isObj(z) && z.ami === true; }) && !("openfoam" in cfg)) {
      out.push(["info", "スライディングメッシュ（ami）は openfoam の書き方を選べます（既定は com = openfoam.com）"]);
    }
  }

  // ---------------------------------------------------------------- output
  function ordered(v, p) {
    if (Array.isArray(v)) return v.map(function (x) { return ordered(x, p && p.items); });
    if (!isObj(v)) return v;
    var keys = Object.keys(v), props = (p && p.properties) || {};
    var known = Object.keys(props).filter(function (k) { return k in v; });
    var comments = keys.filter(function (k) { return !(k in props) && isComment(k); });
    var rest = keys.filter(function (k) { return !(k in props) && !isComment(k); });
    var out = {};
    comments.concat(known, rest).forEach(function (k) {
      out[k] = ordered(v[k], props[k] || (isObj(p && p.additionalProperties) ? p.additionalProperties : null));
    });
    return out;
  }
  function output() {
    var cfg = clone(state);
    if (withDefaults) {
      Object.keys(P).forEach(function (k) {
        if (!(k in cfg) && "default" in P[k]) cfg[k] = clone(P[k]["default"]);
      });
    }
    return ordered(cfg, S);
  }
  // JSON in the style of the examples: short objects and lists on one line
  function flat(v) {
    if (Array.isArray(v)) return "[" + v.map(flat).join(", ") + "]";
    if (isObj(v)) {
      var ks = Object.keys(v);
      return ks.length ? "{ " + ks.map(function (k) { return JSON.stringify(k) + ": " + flat(v[k]); }).join(", ") + " }" : "{}";
    }
    return JSON.stringify(v);
  }
  function pretty(v, ind, keyLen) {
    var f = flat(v);
    if (ind && ind.length + (keyLen || 0) + f.length <= 96) return f;
    var ni = ind + "  ";
    if (Array.isArray(v)) {
      return "[\n" + v.map(function (x) { return ni + pretty(x, ni, 0); }).join(",\n") + "\n" + ind + "]";
    }
    if (isObj(v)) {
      var ks = Object.keys(v);
      if (!ks.length) return "{}";
      return "{\n" + ks.map(function (k) {
        var kk = JSON.stringify(k) + ": ";
        return ni + kk + pretty(v[k], ni, kk.length);
      }).join(",\n") + "\n" + ind + "}";
    }
    return f;
  }
  function jsonText() { return pretty(output(), "", 0) + "\n"; }

  function update() {
    var text = jsonText();
    document.getElementById("mdf-json").textContent = text;
    var out = [];
    var cfg = output();
    check(cfg, S, [], out);
    crossChecks(cfg, out);
    var ul = document.getElementById("mdf-issues");
    ul.textContent = "";
    var nerr = out.filter(function (x) { return x[0] === "error"; }).length;
    if (!out.length) ul.appendChild(el("li", { class: "mdf-ok", text: "問題は見つかりませんでした" }));
    out.forEach(function (x) { ul.appendChild(el("li", { class: "mdf-" + x[0], text: x[1] })); });
    document.getElementById("mdf-download").title = nerr ? "エラーがありますが、書き出せます" : "";
    stlInfo();
  }

  // ---------------------------------------------------------------- STL
  function parseSTL(buf, fileName) {
    var bytes = new Uint8Array(buf);
    var head = String.fromCharCode.apply(null, bytes.subarray(0, 5)).toLowerCase();
    var binary = true;
    if (head === "solid") {
      binary = bytes.length >= 84 && 84 + 50 * new DataView(buf).getUint32(80, true) === bytes.length;
    }
    var lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    var solids = [], ntri = 0;
    function pt(x, y, z) {
      if (x < lo[0]) lo[0] = x; if (x > hi[0]) hi[0] = x;
      if (y < lo[1]) lo[1] = y; if (y > hi[1]) hi[1] = y;
      if (z < lo[2]) lo[2] = z; if (z > hi[2]) hi[2] = z;
    }
    if (binary) {
      if (bytes.length < 84) throw new Error("STL ではありません（84 バイトのヘッダーがない）");
      var dv = new DataView(buf), n = dv.getUint32(80, true);
      if (84 + 50 * n > bytes.length) throw new Error("バイナリ STL が途中で切れています");
      for (var i = 0; i < n; i++) {
        var o = 84 + 50 * i + 12;
        for (var j = 0; j < 3; j++, o += 12) pt(dv.getFloat32(o, true), dv.getFloat32(o + 4, true), dv.getFloat32(o + 8, true));
      }
      ntri = n;
      var h80 = bytes.subarray(0, 80), end = h80.indexOf(0);
      var name = new TextDecoder("latin1").decode(end >= 0 ? h80.subarray(0, end) : h80).replace(/solid/g, "").trim();
      solids = [name || "walls"];
    } else {
      var text = new TextDecoder().decode(bytes);
      var re = /^[ \t]*(solid|vertex)[ \t]*([^\r\n]*)/gm, m, nv = 0;
      while ((m = re.exec(text))) {
        if (m[1] === "solid") {
          var nm = m[2].trim().split(/\s+/)[0] || "walls";
          if (solids.indexOf(nm) < 0) solids.push(nm);
        } else {
          var c = m[2].trim().split(/\s+/).map(Number);
          pt(c[0], c[1], c[2]); nv++;
        }
      }
      ntri = Math.floor(nv / 3);
    }
    if (!ntri) throw new Error("三角形がありません");
    return { file: fileName, format: binary ? "バイナリ" : "ASCII", solids: solids, ntri: ntri, lo: lo, hi: hi };
  }

  function stlInfo() {
    var box = document.getElementById("mdf-stlinfo");
    if (!stl) { box.hidden = true; return; }
    box.hidden = false;
    box.textContent = "";
    var size = [0, 1, 2].map(function (i) { return stl.hi[i] - stl.lo[i]; });
    var L = Math.max.apply(null, size);
    var h = typeof state.baseCellSize === "number" && state.baseCellSize > 0 ? state.baseCellSize : null;
    var pad = typeof state.padding === "number" ? state.padding : P.padding["default"];
    var dl = el("dl", { class: "mdf-facts" },
      el("dt", { text: "ファイル" }), el("dd", {}, el("code", { text: stl.file }), "（" + stl.format + "、" + stl.ntri.toLocaleString() + " 三角形）"),
      el("dt", { text: "solid（パッチ）" }), el("dd", {}, stl.solids.map(function (s, i) {
        return el("span", {}, i ? ", " : "", el("code", { text: s }));
      }), stl.format === "バイナリ" ? el("span", { class: "mdf-muted", text: "（バイナリ STL は solid が 1 つだけ）" }) : null),
      el("dt", { text: "外接箱" }), el("dd", { text: "[" + stl.lo.map(fmtNum).join(", ") + "] 〜 [" + stl.hi.map(fmtNum).join(", ") + "]" }),
      el("dt", { text: "大きさ" }), el("dd", { text: size.map(fmtNum).join(" × ") }));
    if (h) {
      var n = size.map(function (s) { return Math.ceil(s / h) + 2 * pad; });
      var cells = n[0] * n[1] * n[2];
      dl.appendChild(el("dt", { text: "レベル 0 の格子" }));
      dl.appendChild(el("dd", {}, n.join(" × ") + " = " + cells.toLocaleString() + " セル",
        el("span", { class: "mdf-muted", text: "（外接箱いっぱいの数。STL の外は捨てられ、細分化で増えます）" })));
    }
    var sugg = el("div", { class: "mdf-sugg" }, el("span", { text: "baseCellSize の目安: " }));
    [20, 40, 80].forEach(function (k) {
      var v = +(L / k).toPrecision(2);
      sugg.appendChild(el("button", {
        type: "button", class: "mdf-chip", "aria-pressed": h === v ? "true" : "false",
        title: "最長辺の 1/" + k, onclick: function () { state.baseCellSize = v; rerender(); }
      }, fmtNum(v) + "（1/" + k + "）"));
    });
    box.appendChild(el("div", { class: "mdf-stlhead" }, el("strong", { text: "読み込んだ STL" }),
      el("button", { type: "button", class: "mdf-x", title: "閉じる", "aria-label": "STL の情報を閉じる",
        onclick: function () { stl = null; rerender(); } }, "×")));
    box.appendChild(dl);
    box.appendChild(sugg);
  }

  function loadSTL(file) {
    var r = new FileReader();
    r.onload = function () {
      try {
        stl = parseSTL(r.result, file.name);
      } catch (e) {
        alert(file.name + ": " + e.message); return;
      }
      if (!state.domain) state.domain = file.name;
      rerender();
    };
    r.readAsArrayBuffer(file);
  }

  function loadJSON(text, what) {
    var cfg;
    try { cfg = JSON.parse(text.replace(/^﻿/, "")); }
    catch (e) { alert(what + ": JSON として読めません: " + e.message); return; }
    if (!isObj(cfg)) { alert(what + ": meshDict は JSON のオブジェクト {...} です"); return; }
    state = cfg;
    rerender();
  }

  // ---------------------------------------------------------------- page
  function init() {
    try {
      var saved = localStorage.getItem(STORE);
      if (saved) { var v = JSON.parse(saved); if (isObj(v)) state = v; }
    } catch (e) { /* storage off */ }

    var exSel = document.getElementById("mdf-example");
    Object.keys(EXAMPLES).forEach(function (k) { exSel.appendChild(el("option", { value: k, text: k })); });
    exSel.addEventListener("change", function () {
      if (!exSel.value) return;
      if (Object.keys(state).length && !confirm("いまの入力を例題 " + exSel.value + " で置き換えますか？")) {
        exSel.value = ""; return;
      }
      state = clone(EXAMPLES[exSel.value]);
      exSel.value = "";
      rerender();
    });
    document.getElementById("mdf-stl").addEventListener("change", function (e) {
      if (e.target.files[0]) loadSTL(e.target.files[0]);
      e.target.value = "";
    });
    document.getElementById("mdf-open").addEventListener("change", function (e) {
      var f = e.target.files[0];
      if (!f) return;
      f.text().then(function (t) { loadJSON(t, f.name); });
      e.target.value = "";
    });
    document.getElementById("mdf-clear").addEventListener("click", function () {
      if (Object.keys(state).length && !confirm("入力をすべて消しますか？")) return;
      state = {}; rerender();
    });
    document.getElementById("mdf-advanced").addEventListener("change", function (e) {
      showAdvanced = e.target.checked; render();
    });
    document.getElementById("mdf-defaults").addEventListener("change", function (e) {
      withDefaults = e.target.checked; update();
    });
    document.getElementById("mdf-copy").addEventListener("click", function (e) {
      var btn = e.target, text = jsonText();
      function done() { btn.textContent = "コピーしました"; setTimeout(function () { btn.textContent = "コピー"; }, 1500); }
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text); done(); });
      } else { fallbackCopy(text); done(); }
    });
    document.getElementById("mdf-download").addEventListener("click", function () {
      var a = el("a", { href: URL.createObjectURL(new Blob([jsonText()], { type: "application/json" })),
        download: "meshDict.json" });
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
    });
    document.addEventListener("dragover", function (e) { e.preventDefault(); });
    document.addEventListener("drop", function (e) {
      e.preventDefault();
      Array.prototype.forEach.call(e.dataTransfer.files, function (f) {
        if (/\.stl$/i.test(f.name)) loadSTL(f);
        else f.text().then(function (t) { loadJSON(t, f.name); });
      });
    });
    render();
    update();
  }
  function fallbackCopy(text) {
    var ta = el("textarea", { style: "position:fixed;left:-9999px" });
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e) { /* ignore */ }
    ta.remove();
  }

  window.cutmeshMeshDict = { parseSTL: parseSTL, check: check, crossChecks: crossChecks,
    pretty: function (v) { return pretty(v, "", 0); }, schema: S };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
