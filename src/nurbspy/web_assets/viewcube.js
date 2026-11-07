/* ViewCube adapted from turbo_moc's camera-linked SVG orientation overlay.
 * Original: method_of_characteristics/src/turbo_moc/app/assets/orientation_cube.js
 * Adaptation: renderer-independent readCamera/writeCamera interface for nurbspy.
 * Camera navigation only; this module performs no NURBS or scientific evaluation.
 *
 * MIT License
 *
 * Copyright (c) 2026 Sustainable Thermal Power Research group at DTU Construct
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const unit = a => a.map(value => value / (Math.hypot(...a) || 1));
const subtract = (a, b) => a.map((value, i) => value - b[i]);
const add = (a, b) => a.map((value, i) => value + b[i]);
const scale = (a, factor) => a.map(value => value * factor);

function basis(back, up) {
  back = unit(back);
  const right = unit(cross(up, back));
  return {right, up: unit(cross(back, right)), back};
}

function targetBasis(direction, preferredUp = [0, 0, 1]) {
  const back = unit(direction);
  // A pole needs a different up vector. Pick a stable perpendicular axis,
  // including when the machine's preferred engineering view has Y up.
  const up = unit(preferredUp);
  const fallback = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    .find(axis => Math.abs(dot(back, axis)) < .9);
  return basis(back, Math.abs(dot(back, up)) > .999 ? fallback : up);
}

function rotate(vector, axis, degrees) {
  const angle = degrees * Math.PI / 180, cosine = Math.cos(angle), sine = Math.sin(angle);
  return add(add(scale(vector, cosine), scale(cross(axis, vector), sine)),
    scale(axis, dot(axis, vector) * (1-cosine)));
}

function quaternion({right: x, up: y, back: z}) {
  const trace = x[0] + y[1] + z[2];
  let q;
  if (trace > 0) {
    const s = 2 * Math.sqrt(trace + 1);
    q = [(y[2]-z[1])/s, (z[0]-x[2])/s, (x[1]-y[0])/s, s/4];
  } else if (x[0] > y[1] && x[0] > z[2]) {
    const s = 2 * Math.sqrt(1 + x[0] - y[1] - z[2]);
    q = [s/4, (y[0]+x[1])/s, (z[0]+x[2])/s, (y[2]-z[1])/s];
  } else if (y[1] > z[2]) {
    const s = 2 * Math.sqrt(1 + y[1] - x[0] - z[2]);
    q = [(y[0]+x[1])/s, s/4, (z[1]+y[2])/s, (z[0]-x[2])/s];
  } else {
    const s = 2 * Math.sqrt(1 + z[2] - x[0] - y[1]);
    q = [(z[0]+x[2])/s, (z[1]+y[2])/s, s/4, (x[1]-y[0])/s];
  }
  return unit(q);
}

function fromQuaternion([x, y, z, w]) {
  return {
    right: [1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)],
    up: [2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w)],
    back: [2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)]
  };
}

function interpolate(start, end, fraction) {
  let cosine = dot(start, end);
  if (cosine < 0) { end = scale(end, -1); cosine = -cosine; }
  if (cosine > .9995) return unit(add(scale(start, 1-fraction), scale(end, fraction)));
  const angle = Math.acos(Math.min(1, cosine));
  return add(scale(start, Math.sin((1-fraction)*angle)/Math.sin(angle)),
    scale(end, Math.sin(fraction*angle)/Math.sin(angle)));
}

// A cube with actual bevels: six square faces, twelve edge rectangles and
// eight corner triangles. Bevel targets remain visible in a face-on view.
const faces = [], bevel = .62;
for (const x of [-1, 0, 1]) for (const y of [-1, 0, 1]) for (const z of [-1, 0, 1]) {
  const direction = [x, y, z], axes = [0, 1, 2].filter(axis => direction[axis]);
  if (!axes.length) continue;
  const kind = [null, "face", "edge", "corner"][axes.length];
  const normal = unit(direction), vertices = [];
  if (kind === "face") {
    const axis = axes[0], tangent = (axis+1)%3, other = (axis+2)%3;
    for (const u of [-bevel, bevel]) for (const v of [-bevel, bevel]) {
      const point = [...direction]; point[tangent] = u; point[other] = v; vertices.push(point);
    }
  } else if (kind === "edge") {
    const other = [0, 1, 2].find(axis => !direction[axis]);
    for (const axis of axes) for (const end of [-bevel, bevel]) {
      const point = scale(direction, bevel); point[axis] = direction[axis]; point[other] = end;
      vertices.push(point);
    }
  } else {
    for (const axis of axes) {
      const point = scale(direction, bevel); point[axis] = direction[axis]; vertices.push(point);
    }
  }
  const center = scale(vertices.reduce(add, [0, 0, 0]), 1/vertices.length);
  const frame = targetBasis(normal);
  const angle = point => Math.atan2(dot(subtract(point, center), frame.up), dot(subtract(point, center), frame.right));
  vertices.sort((a, b) => angle(a)-angle(b));
  const label = direction.map((value, axis) => value ? (value > 0 ? "+" : "-") + "XYZ"[axis] : "").filter(Boolean).join(" ");
  faces.push({normal, direction, vertices, center, kind, axis: axes[0], label});
}

const svgNS = "http://www.w3.org/2000/svg";
function svgElement(name, attributes = {}, text) {
  const element = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text != null) element.textContent = text;
  return element;
}

export function createViewCube(parent, adapter) {
  const host = document.createElement("div");
  host.className = "orientation-widget";
  host.hidden = false;
  host.setAttribute("role", "group");
  host.setAttribute("aria-label", "3D camera orientation");
  const svg = svgElement("svg", {viewBox: "0 0 168 158", role: "group", tabindex: "0",
    "aria-label": "Orientation cube. Click a face, bevel or corner. Arrow keys orbit; Q and E roll; Home restores the engineering view; I selects isometric."});
  const title = svgElement("title", {}, "Click a face, bevel or corner. Arrows: orbit 30°. Q/E: roll 15°. Home: engineering view. I: isometric.");
  svg.appendChild(title);
  const faceLayer = svgElement("g"), controlLayer = svgElement("g"), axesLayer = svgElement("g");
  svg.append(faceLayer, controlLayer, axesLayer);
  const footer = document.createElement("div");
  footer.className = "orientation-footer";
  const homeButton = document.createElement("button"); homeButton.type = "button"; homeButton.textContent = "HOME";
  homeButton.title = "Restore the initial geometry view (Home; preserve zoom and focus)";
  homeButton.setAttribute("aria-label", "Restore initial view");
  const iso = document.createElement("button"); iso.type = "button"; iso.textContent = "ISO";
  iso.title = "Isometric view (preserve zoom and focus)";
  iso.setAttribute("aria-label", "Isometric view");
  footer.append(homeButton, iso); host.append(svg, footer); parent.appendChild(host);

  let frame = null, animation = null, destroyed = false;
  let preferredUp = [0,0,1], homeBasis = targetBasis([1,1,1]);
  let currentBasis = homeBasis, destination = null;
  const faceNodes = faces.map(face => {
    const group = svgElement("g", {class: "orientation-face orientation-" + face.kind, role: "button",
      "aria-label": "View from " + face.label, "data-direction": face.direction.join(","), "data-kind": face.kind});
    const polygon = svgElement("polygon", {class: "orientation-face-fill", "data-axis": "XYZ"[face.axis]});
    const text = svgElement("text", {"text-anchor": "middle", "dominant-baseline": "central"}, face.label);
    group.append(svgElement("title", {}, "View from " + face.label + (face.kind === "face" ? "" : " (" + face.kind + ")")), polygon);
    if (face.kind === "face") group.appendChild(text);
    group.addEventListener("click", () => snap(face.direction));
    group.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault(); svg.focus({preventScroll: true}); snap(face.direction);
      }
    });
    faceLayer.appendChild(group);
    return {face, group, polygon, text};
  });
  function control(action, label, x, y, path, callback) {
    const node = svgElement("g", {class: "orientation-control", role: "button", tabindex: "0",
      transform: `translate(${x},${y})`, "aria-label": label, "data-action": action});
    node.append(svgElement("title", {}, label), svgElement("circle", {r: 11}), svgElement("path", {d: path}));
    node.addEventListener("click", callback);
    node.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault(); svg.focus({preventScroll: true}); callback();
      }
    });
    controlLayer.appendChild(node);
  }
  control("orbit-left", "Orbit left by 30° (Left arrow)", 17, 76, "M 2,-4 -3,0 2,4", () => orbit(-30, 0));
  control("orbit-right", "Orbit right by 30° (Right arrow)", 151, 76, "M -2,-4 3,0 -2,4", () => orbit(30, 0));
  control("orbit-up", "Orbit up by 30° (Up arrow)", 84, 15, "M -4,2 0,-3 4,2", () => orbit(0, 30));
  control("orbit-down", "Orbit down by 30° (Down arrow)", 84, 138, "M -4,-2 0,3 4,-2", () => orbit(0, -30));
  control("roll-left", "Roll counterclockwise by 15° (Q)", 23, 22, "M 4,4 A 5,5 0 1 0 -4,-4 M -4,-8 -4,-4 0,-4", () => roll(-15));
  control("roll-right", "Roll clockwise by 15° (E)", 145, 22, "M -4,4 A 5,5 0 1 1 4,-4 M 4,-8 4,-4 0,-4", () => roll(15));
  const axisNodes = ["X", "Y", "Z"].map((label, axis) => {
    const line = svgElement("line", {class: "orientation-axis", "data-axis": label});
    const text = svgElement("text", {class: "orientation-axis-label", "data-axis": label}, label);
    axesLayer.append(line, text);
    return {axis, line, text};
  });

  function draw() {
    frame = null;
    if (destroyed) return;
    const camera = adapter.readCamera();
    currentBasis = basis(subtract(camera.position, camera.target), camera.up);
    const project = point => [84 + 30*dot(point, currentBasis.right), 76 - 30*dot(point, currentBasis.up)];
    for (const {face, group, polygon, text} of faceNodes) {
      const facing = dot(face.normal, currentBasis.back);
      const visible = facing > .015;
      group.style.display = visible ? "" : "none";
      group.setAttribute("tabindex", visible ? "0" : "-1");
      group.setAttribute("aria-hidden", String(!visible));
      if (!visible) continue;
      polygon.setAttribute("points", face.vertices.map(point => project(point).join(",")).join(" "));
      const center = project(face.center);
      text.setAttribute("x", center[0]); text.setAttribute("y", center[1]);
      text.style.display = facing > .2 ? "" : "none";
    }
    for (const {axis, line, text} of axisNodes) {
      const dx = currentBasis.right[axis], dy = -currentBasis.up[axis];
      line.setAttribute("x1", 29); line.setAttribute("y1", 137);
      line.setAttribute("x2", 29+14*dx); line.setAttribute("y2", 137+14*dy);
      text.setAttribute("x", 29+21*dx); text.setAttribute("y", 140+18*dy);
    }
    host.dataset.orientation = currentBasis.back.map(value => value.toFixed(4)).join(",");
    host.dataset.viewUp = currentBasis.up.map(value => value.toFixed(4)).join(",");
  }

  function queueDraw() { if (!destroyed && frame == null) frame = requestAnimationFrame(draw); }
  function cancel(notify = true) {
    const active = animation != null;
    if (active) cancelAnimationFrame(animation);
    animation = null;
    destination = null;
    if (active && notify) adapter.onCancel?.();
  }
  function animate(endBasis) {
    if (destroyed) return;
    cancel();
    const camera = adapter.readCamera(), focus = [...camera.target];
    const distance = Math.hypot(...subtract(camera.position, focus));
    const start = quaternion(basis(subtract(camera.position, focus), camera.up));
    const end = quaternion(endBasis);
    destination = endBasis;
    const duration = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 240;
    const started = performance.now();
    function step(now) {
      if (destroyed) return;
      const t = duration ? Math.min(1, (now-started)/duration) : 1;
      const orientation = fromQuaternion(interpolate(start, end, t*t*(3-2*t)));
      adapter.writeCamera({position: add(focus, scale(orientation.back, distance)),
        target: [...focus], up: [...orientation.up]}, t >= 1);
      queueDraw();
      animation = t < 1 ? requestAnimationFrame(step) : null;
      if (t >= 1) destination = null;
    }
    animation = requestAnimationFrame(step);
  }
  function snap(direction, up = preferredUp) { animate(targetBasis(direction, up)); }
  function setHome(direction, up = [0,0,1]) {
    preferredUp = unit(up); homeBasis = targetBasis(direction, preferredUp);
  }
  function home() { animate(homeBasis); }
  function cameraBasis() {
    if (destination) return destination;
    if (destroyed) return currentBasis;
    const camera = adapter.readCamera();
    return basis(subtract(camera.position, camera.target), camera.up);
  }
  function orbit(horizontalDegrees, verticalDegrees) {
    let frame = cameraBasis();
    if (horizontalDegrees) {
      frame = Object.fromEntries(Object.entries(frame).map(([name, vector]) => [name, rotate(vector, frame.up, horizontalDegrees)]));
    }
    if (verticalDegrees) {
      frame = Object.fromEntries(Object.entries(frame).map(([name, vector]) => [name, rotate(vector, frame.right, -verticalDegrees)]));
    }
    animate(frame);
  }
  function roll(degrees) {
    const frame = cameraBasis();
    animate(basis(frame.back, rotate(frame.up, frame.back, degrees)));
  }
  host.addEventListener("pointerdown", event => event.stopPropagation());
  host.addEventListener("dblclick", event => event.stopPropagation());
  iso.addEventListener("click", () => snap([1,1,1]));
  homeButton.addEventListener("click", home);
  svg.addEventListener("keydown", event => {
    const actions = {ArrowLeft: () => orbit(-30, 0), ArrowRight: () => orbit(30, 0),
      ArrowUp: () => orbit(0, 30), ArrowDown: () => orbit(0, -30),
      q: () => roll(-15), Q: () => roll(-15), e: () => roll(15), E: () => roll(15),
      Home: home, i: () => snap([1,1,1]), I: () => snap([1,1,1])};
    if (actions[event.key]) {
      event.preventDefault(); event.stopPropagation();
      // A previously focused face may turn away and become hidden.
      svg.focus({preventScroll: true}); actions[event.key]();
    }
  });

  // The host calls update() whenever its camera changes. The adapter keeps this
  // overlay independent of Three.js, VTK, or any scientific geometry backend.
  adapter.interactionElement?.addEventListener("pointerdown", cancel);
  function destroy() {
    if (destroyed) return;
    destroyed = true;
    cancel(false);
    if (frame != null) cancelAnimationFrame(frame);
    frame = null;
    adapter.interactionElement?.removeEventListener("pointerdown", cancel);
    host.remove();
  }
  queueDraw();
  return {host, update: queueDraw, cancel, setHome, home, snap, orbit, roll, destroy};
}

export {basis, targetBasis, quaternion, fromQuaternion, interpolate};
