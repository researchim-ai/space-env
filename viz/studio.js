import * as THREE from "./vendor/three.module.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const COLORS = ["#9fd4ff", "#f0c27a", "#c6ecd4", "#e7b4c3", "#d7c4f2", "#f3d2a4"];
const COMPANION_COLORS = { HOT: "#ffb15a", WARM: "#e4d3bc" };
let markedStation = "";
let liveTracks = null;

function liveSample(name, frameT) {
  const rows = liveTracks && liveTracks.vehicles && liveTracks.vehicles[name];
  if (!rows || !rows.length || frameT == null) return null;
  const last = rows[rows.length - 1];
  if (frameT > last.t + 1) return null;
  let chosen = rows[0];
  for (const row of rows) {
    if (row.t <= frameT + 1) chosen = row;
    else break;
  }
  return chosen;
}

export function mountStudio(stage, readout, clock, scrub, mapCanvas, onFrame) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  stage.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#070b12");
  const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 80);
  camera.position.set(2.7, 1.35, 2.15);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 1.45;
  controls.maxDistance = 8;
  controls.target.set(0, 0, 0);

  const sunLight = new THREE.DirectionalLight("#fff6e8", 2.4);
  scene.add(sunLight);
  scene.add(new THREE.AmbientLight("#c5d4e6", 1.15));
  scene.add(new THREE.HemisphereLight("#d7e6f5", "#1a2433", 0.55));

  const stars = new THREE.BufferGeometry();
  const starPositions = new Float32Array(900 * 3);
  for (let index = 0; index < 900; index += 1) {
    const radius = 22 + Math.random() * 10;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    starPositions[index * 3] = radius * Math.sin(phi) * Math.cos(theta);
    starPositions[index * 3 + 1] = radius * Math.cos(phi);
    starPositions[index * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);
  }
  stars.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
  scene.add(new THREE.Points(stars, new THREE.PointsMaterial({ color: "#d5deea", size: 0.03, sizeAttenuation: true })));

  const earthPivot = new THREE.Group();
  scene.add(earthPivot);
  const loader = new THREE.TextureLoader();
  const dayMap = loader.load("/viz/textures/earth_day.jpg");
  dayMap.colorSpace = THREE.SRGBColorSpace;
  earthPivot.add(new THREE.Mesh(
    new THREE.SphereGeometry(1, 96, 72),
    new THREE.MeshBasicMaterial({ map: dayMap }),
  ));
  scene.add(new THREE.Mesh(
    new THREE.SphereGeometry(1.04, 64, 48),
    new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.BackSide,
      uniforms: { glowColor: { value: new THREE.Color("#9ec9ff") } },
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vWorld;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vec4 world = modelMatrix * vec4(position, 1.0);
          vWorld = world.xyz;
          gl_Position = projectionMatrix * viewMatrix * world;
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        varying vec3 vWorld;
        uniform vec3 glowColor;
        void main() {
          vec3 viewDir = normalize(cameraPosition - vWorld);
          float rim = pow(1.0 - abs(dot(normalize(vNormal), viewDir)), 2.8);
          gl_FragColor = vec4(glowColor, rim * 0.85);
        }
      `,
    }),
  ));

  const openLines = makeLines("#8ef0c8", 0.95);
  const quietLines = makeLines("#8aa0b8", 0.35);
  scene.add(openLines, quietLines);

  let episode = null;
  let scale = 1;
  let playing = true;
  let frameIndex = 0;
  let accumulator = 0;
  const satellites = new Map();
  const orbitLines = [];
  const orbitByName = new Map();
  const stationMeshes = [];
  let arcMode = "flown";

  function resize() {
    const width = stage.clientWidth || window.innerWidth;
    const height = stage.clientHeight || window.innerHeight;
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  }

  function clearEpisode() {
    satellites.forEach((sat) => scene.remove(sat));
    satellites.clear();
    orbitLines.forEach((line) => {
      scene.remove(line);
      line.geometry.dispose();
    });
    orbitLines.length = 0;
    orbitByName.clear();
  }

  function bodyVector(vehicle, name, frameT) {
    const live = liveSample(name, frameT);
    if (live) return live.r;
    if (arcMode === "coast" && vehicle && vehicle.r_coast) return vehicle.r_coast;
    return vehicle.r;
  }

  function bodyVelocity(vehicle, name, frameT) {
    const live = liveSample(name, frameT);
    if (live) return live.v;
    if (arcMode === "coast" && vehicle && vehicle.v_coast) return vehicle.v_coast;
    return vehicle.v;
  }

  function eciToWorld(values) {
    return new THREE.Vector3(values[0], values[2], values[1]).multiplyScalar(scale);
  }

  function showFrame(index) {
    const frame = episode.frames[index];
    sunLight.position.copy(eciToWorld(frame.sun).normalize().multiplyScalar(8));
    earthPivot.rotation.y = -frame.gast;
    earthPivot.updateMatrixWorld(true);
    while (stationMeshes.length < frame.stations.length) {
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.014, 16, 12),
        new THREE.MeshBasicMaterial({ color: "#f3e2bc" }),
      );
      earthPivot.add(mesh);
      stationMeshes.push(mesh);
    }
    frame.stations.forEach((station, stationIndex) => {
      const world = eciToWorld(station.r);
      stationMeshes[stationIndex].position.copy(earthPivot.worldToLocal(world));
      stationMeshes[stationIndex].visible = true;
      const inPass = Object.values(frame.vehicles).some(
        (vehicle) => vehicle.contact && vehicle.station === station.name,
      );
      const marked = station.name === markedStation;
      stationMeshes[stationIndex].scale.setScalar(marked ? 1.8 : 1);
      stationMeshes[stationIndex].material.color.set(marked ? "#ffb15a" : (inPass ? "#f0c27a" : "#f6ead0"));
    });
    for (let extra = frame.stations.length; extra < stationMeshes.length; extra += 1) {
      stationMeshes[extra].visible = false;
    }
    const rows = [];
    Object.entries(frame.companions || {}).forEach(([name, body]) => {
      const sat = satellites.get(name);
      if (!sat || !body) return;
      sat.position.copy(eciToWorld(body.r));
    });
    Object.entries(frame.vehicles).forEach(([name, vehicle]) => {
      const sat = satellites.get(name);
      if (!sat) return;
      const live = liveSample(name, frame.t);
      const position = eciToWorld(bodyVector(vehicle, name, frame.t));
      sat.position.copy(position);
      const forward = eciToWorld(bodyVelocity(vehicle, name, frame.t)).normalize();
      if (forward.lengthSq() > 0) {
        sat.up.copy(sunLight.position).normalize();
        sat.lookAt(position.clone().add(forward));
      }
      rows.push({
        name,
        altitudeKm: live && live.altitude_km != null
          ? live.altitude_km
          : (arcMode === "coast" && vehicle.altitude_km_coast != null ? vehicle.altitude_km_coast : vehicle.altitude_km),
        eclipse: live ? live.eclipse : vehicle.eclipse,
        contact: live ? live.contact : vehicle.contact,
        station: live ? (live.station || "") : (vehicle.station || ""),
      });
    });
    drawPairs(frame);
    const openCount = (frame.pairs || []).filter((pair) => pair.open).length;
    clock.textContent = formatClock(frame.t);
    scrub.value = String(index);
    if (readout) readout.textContent = "";
    if (onFrame) {
      onFrame({
        time: frame.t,
        openLinks: openCount,
        vehicles: rows,
        neighbors: companionOffsets(frame, arcMode),
      });
    }
    drawMap(mapCanvas, episode, index, arcMode);
  }

  function drawPairs(frame) {
    const pairs = frame.pairs || [];
    paint(openLines, pairs.filter((pair) => pair.open), frame);
    paint(quietLines, pairs.filter((pair) => !pair.open), frame);
  }

  function paint(lines, pairs, frame) {
    const needed = Math.max(pairs.length, 1) * 6;
    let positions = lines.geometry.attributes.position.array;
    if (positions.length < needed) {
      positions = new Float32Array(needed);
      lines.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    }
    pairs.forEach((pair, index) => {
      const leader = frame.vehicles[pair.leader];
      const follower = frame.vehicles[pair.follower];
      if (!leader || !follower) return;
      const start = eciToWorld(bodyVector(leader, pair.leader, frame.t));
      const end = eciToWorld(bodyVector(follower, pair.follower, frame.t));
      const offset = index * 6;
      positions[offset] = start.x;
      positions[offset + 1] = start.y;
      positions[offset + 2] = start.z;
      positions[offset + 3] = end.x;
      positions[offset + 4] = end.y;
      positions[offset + 5] = end.z;
    });
    lines.geometry.attributes.position.needsUpdate = true;
    lines.geometry.setDrawRange(0, pairs.length * 2);
  }

  function load(next) {
    clearEpisode();
    episode = next;
    scale = 1 / episode.earth_radius_m;
    frameIndex = 0;
    scrub.max = String(Math.max(episode.frames.length - 1, 0));
    const names = Object.keys(episode.frames[0].vehicles);
    names.forEach((name, index) => {
      const color = COLORS[index % COLORS.length];
      const positions = episode.frames.map((frame) => eciToWorld(bodyVector(frame.vehicles[name], name, frame.t)));
      const orbit = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(positions),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.45 }),
      );
      scene.add(orbit);
      orbitLines.push(orbit);
      orbitByName.set(name, orbit);
      const sat = makeSatellite(color);
      sat.scale.setScalar(names.length > 12 ? 0.62 : 1.45);
      scene.add(sat);
      satellites.set(name, sat);
    });
    Object.keys(episode.frames[0].companions || {}).forEach((name) => {
      const color = COMPANION_COLORS[name] || "#f4f7fb";
      const positions = episode.frames
        .filter((frame) => frame.companions && frame.companions[name])
        .map((frame) => eciToWorld(frame.companions[name].r));
      const orbit = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(positions),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.35 }),
      );
      scene.add(orbit);
      orbitLines.push(orbit);
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 18, 14),
        new THREE.MeshBasicMaterial({ color }),
      );
      scene.add(marker);
      satellites.set(name, marker);
    });
    showFrame(0);
    playing = true;
  }

  function tick() {
    if (playing && episode && episode.frames.length > 1) {
      accumulator += 1;
      if (accumulator >= 8) {
        accumulator = 0;
        frameIndex = (frameIndex + 1) % episode.frames.length;
        showFrame(frameIndex);
      }
    }
    controls.update();
    renderer.render(scene, camera);
  }

  resize();
  const observer = new ResizeObserver(resize);
  observer.observe(stage);
  renderer.setAnimationLoop(tick);
  scrub.addEventListener("input", () => {
    if (!episode) return;
    frameIndex = Number(scrub.value);
    showFrame(frameIndex);
  });

  return {
    load,
    setPlaying(next) {
      playing = next;
      return playing;
    },
    setFocus(seconds, station) {
      markedStation = station || "";
      this.seekTime(seconds);
    },
    setFlight(flight) {
      liveTracks = flight && flight.vehicles ? flight : null;
      if (!episode) return;
      orbitByName.forEach((line, name) => {
        const points = episode.frames.map((frame) => eciToWorld(bodyVector(frame.vehicles[name], name, frame.t)));
        line.geometry.dispose();
        line.geometry = new THREE.BufferGeometry().setFromPoints(points);
      });
      showFrame(frameIndex);
    },
    setArc(mode) {
      const next = mode === "coast" ? "coast" : "flown";
      if (next === arcMode) return;
      arcMode = next;
      if (!episode) return;
      orbitByName.forEach((line, name) => {
        const points = episode.frames.map((frame) => eciToWorld(bodyVector(frame.vehicles[name], name, frame.t)));
        line.geometry.dispose();
        line.geometry = new THREE.BufferGeometry().setFromPoints(points);
      });
      showFrame(frameIndex);
    },
    seekTime(seconds) {
      if (!episode) return;
      let best = 0;
      let bestGap = Infinity;
      episode.frames.forEach((frame, index) => {
        const gap = Math.abs(frame.t - seconds);
        if (gap < bestGap) {
          bestGap = gap;
          best = index;
        }
      });
      frameIndex = best;
      playing = false;
      showFrame(best);
    },
    get playing() {
      return playing;
    },
  };
}

function makeLines(color, opacity) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
  geometry.setDrawRange(0, 0);
  return new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity }),
  );
}

function makeSatellite(color) {
  const group = new THREE.Group();
  const bus = new THREE.MeshStandardMaterial({
    color: "#e4e8ee",
    metalness: 0.72,
    roughness: 0.28,
  });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.022, 0.016, 0.03), bus);
  const stripe = new THREE.Mesh(
    new THREE.BoxGeometry(0.023, 0.004, 0.008),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35, metalness: 0.3, roughness: 0.4 }),
  );
  stripe.position.y = 0.008;
  const cells = new THREE.MeshStandardMaterial({
    color: "#16375f",
    metalness: 0.55,
    roughness: 0.22,
    emissive: "#0c2344",
    emissiveIntensity: 0.4,
  });
  const wingGeo = new THREE.BoxGeometry(0.05, 0.0014, 0.02);
  const left = new THREE.Mesh(wingGeo, cells);
  const right = new THREE.Mesh(wingGeo, cells);
  left.position.x = -0.04;
  right.position.x = 0.04;
  const hinge = new THREE.MeshStandardMaterial({ color: "#8d97a3", metalness: 0.8, roughness: 0.3 });
  const yoke = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.001, 0.004), hinge);
  const dish = new THREE.Mesh(
    new THREE.CylinderGeometry(0.007, 0.007, 0.0016, 16),
    new THREE.MeshStandardMaterial({ color: "#d7b56a", metalness: 0.85, roughness: 0.2 }),
  );
  dish.position.set(0, 0.0105, -0.006);
  const feed = new THREE.Mesh(
    new THREE.BoxGeometry(0.0014, 0.006, 0.0014),
    hinge,
  );
  feed.position.set(0, 0.012, -0.006);
  const instrument = new THREE.Mesh(
    new THREE.CylinderGeometry(0.004, 0.005, 0.006, 12),
    new THREE.MeshStandardMaterial({ color: "#1c2128", metalness: 0.4, roughness: 0.45 }),
  );
  instrument.position.set(0, -0.01, 0.006);
  group.add(body, stripe, left, right, yoke, dish, feed, instrument);
  return group;
}

function subsatellite(r, gast) {
  const worldX = r[0];
  const worldY = r[2];
  const worldZ = r[1];
  const cos = Math.cos(gast);
  const sin = Math.sin(gast);
  const localX = worldX * cos + worldZ * sin;
  const localY = worldY;
  const localZ = -worldX * sin + worldZ * cos;
  const lat = Math.atan2(localY, Math.hypot(localX, localZ)) * 180 / Math.PI;
  let phi = Math.atan2(localZ, -localX);
  if (phi < 0) phi += Math.PI * 2;
  return { u: 1 - phi / (Math.PI * 2), lat };
}

function drawMap(canvas, episode, frameIndex, arcMode) {
  if (!canvas || !episode) return;
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(rect.width * ratio));
  const height = Math.max(1, Math.floor(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  if (!canvas._map) {
    const image = new Image();
    image.src = "/viz/textures/earth_day.jpg";
    canvas._map = image;
  }
  if (canvas._map.complete && canvas._map.naturalWidth) {
    context.drawImage(canvas._map, 0, 0, width, height);
  } else {
    context.fillStyle = "#0c1824";
    context.fillRect(0, 0, width, height);
  }
  context.fillStyle = "rgba(5, 10, 16, 0.18)";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(232, 238, 246, 0.16)";
  context.lineWidth = ratio;
  for (let lat = -60; lat <= 60; lat += 30) {
    const y = ((90 - lat) / 180) * height;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
  for (let lon = -150; lon < 180; lon += 30) {
    const x = ((lon + 180) / 360) * width;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
  const names = Object.keys(episode.frames[0].vehicles);
  const now = episode.frames[frameIndex].t;
  const span = episode.frames[episode.frames.length - 1].t - episode.frames[0].t;
  const revSec = 6000;
  const windowed = span > revSec;
  const from = windowed ? now - revSec / 2 : episode.frames[0].t;
  const to = windowed ? now + revSec / 2 : episode.frames[episode.frames.length - 1].t;
  names.forEach((name, index) => {
    const color = COLORS[index % COLORS.length];
    let penDown = false;
    let previousX = 0;
    context.beginPath();
    context.lineWidth = 1.6 * ratio;
    context.strokeStyle = color;
    episode.frames.forEach((frame) => {
      if (frame.t < from - 1e-6 || frame.t > to + 1e-6) return;
      const vehicle = frame.vehicles[name];
      if (!vehicle) return;
      const live = liveSample(name, frame.t);
      const track = live ? live.r : (arcMode === "coast" && vehicle.r_coast ? vehicle.r_coast : vehicle.r);
      const point = subsatellite(track, frame.gast);
      const x = point.u * width;
      const y = ((90 - point.lat) / 180) * height;
      if (!penDown || Math.abs(x - previousX) > width * 0.45) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
      penDown = true;
      previousX = x;
    });
    context.globalAlpha = 0.9;
    context.stroke();
    context.globalAlpha = 1;
  });
  const companionNames = Object.keys(episode.frames[0].companions || {});
  companionNames.forEach((name) => {
    let penDown = false;
    let previousX = 0;
    context.beginPath();
    context.setLineDash([5 * ratio, 4 * ratio]);
    context.lineWidth = 1.3 * ratio;
    context.strokeStyle = COMPANION_COLORS[name] || "#f4f7fb";
    episode.frames.forEach((frame) => {
      if (frame.t < from - 1e-6 || frame.t > to + 1e-6) return;
      const body = (frame.companions || {})[name];
      if (!body) return;
      const point = subsatellite(body.r, frame.gast);
      const x = point.u * width;
      const y = ((90 - point.lat) / 180) * height;
      if (!penDown || Math.abs(x - previousX) > width * 0.45) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
      penDown = true;
      previousX = x;
    });
    context.stroke();
    context.setLineDash([]);
  });
  const frame = episode.frames[frameIndex];
  frame.stations.forEach((station) => {
    const point = subsatellite(station.r, frame.gast);
    const x = point.u * width;
    const y = ((90 - point.lat) / 180) * height;
    const marked = station.name === markedStation;
    const mark = marked ? 8 * ratio : 5 * ratio;
    context.fillStyle = marked ? "#ffb15a" : "#f6ead0";
    context.fillRect(x - mark / 2, y - mark / 2, mark, mark);
    context.fillStyle = "#f4f7fb";
    context.font = `${11 * ratio}px "Avenir Next", "Segoe UI", sans-serif`;
    context.fillText(station.name, x + 6 * ratio, y - 4 * ratio);
  });
  names.forEach((name, index) => {
    const vehicle = frame.vehicles[name];
    if (!vehicle) return;
    const liveMarker = liveSample(name, frame.t);
    const marker = liveMarker ? liveMarker.r : (arcMode === "coast" && vehicle.r_coast ? vehicle.r_coast : vehicle.r);
    const point = subsatellite(marker, frame.gast);
    const x = point.u * width;
    const y = ((90 - point.lat) / 180) * height;
    context.beginPath();
    context.arc(x, y, 4.5 * ratio, 0, Math.PI * 2);
    context.fillStyle = COLORS[index % COLORS.length];
    context.fill();
    context.lineWidth = ratio;
    context.strokeStyle = "#f4f7fb";
    context.stroke();
    if (names.length <= 8) {
      context.fillStyle = "#f4f7fb";
      context.font = `${11 * ratio}px "Avenir Next", "Segoe UI", sans-serif`;
      context.fillText(name, x + 7 * ratio, y + 4 * ratio);
    }
  });
  companionNames.forEach((name) => {
    const body = (frame.companions || {})[name];
    if (!body) return;
    const point = subsatellite(body.r, frame.gast);
    const x = point.u * width;
    const y = ((90 - point.lat) / 180) * height;
    context.beginPath();
    context.arc(x, y, 3.5 * ratio, 0, Math.PI * 2);
    context.fillStyle = COMPANION_COLORS[name] || "#f4f7fb";
    context.fill();
    context.lineWidth = ratio;
    context.strokeStyle = "#1a1208";
    context.stroke();
    const offset = companionOffsets(frame, arcMode).find((item) => item.name === name);
    const label = offset ? `${name} ${offset.label}` : name;
    context.fillStyle = "#f4f7fb";
    context.font = `${11 * ratio}px "Avenir Next", "Segoe UI", sans-serif`;
    context.fillText(label, x + 7 * ratio, y + 4 * ratio);
  });
  context.font = `${13 * ratio}px Palatino, Georgia, serif`;
  const caption = windowed ? "Трасса текущего витка" : "Трасса подспутниковой точки";
  const captionWidth = context.measureText(caption).width + 16 * ratio;
  context.fillStyle = "rgba(8, 12, 18, 0.72)";
  context.fillRect(width - captionWidth - 10 * ratio, 10 * ratio, captionWidth, 28 * ratio);
  context.fillStyle = "#e8eef6";
  context.fillText(caption, width - captionWidth - 2 * ratio, 29 * ratio);
}

function companionOffsets(frame, arcMode) {
  const vehicles = frame.vehicles || {};
  const primaryName = Object.keys(vehicles).find((name) => vehicles[name] && vehicles[name].r_coast) || "sentinel-2a";
  const primary = vehicles[primaryName];
  if (!primary) return [];
  const live = liveSample(primaryName, frame.t);
  const originR = live ? live.r : (arcMode === "coast" && primary.r_coast ? primary.r_coast : primary.r);
  const originV = live ? live.v : (arcMode === "coast" && primary.v_coast ? primary.v_coast : primary.v);
  return Object.entries(frame.companions || {}).filter(([, body]) => body).map(([name, body]) => ({
    name,
    ...relativeOffset(originR, originV, body.r),
  }));
}

function relativeOffset(originR, originV, otherR) {
  const radius = Math.hypot(originR[0], originR[1], originR[2]);
  const radial = originR.map((item) => item / radius);
  const angular = [
    originR[1] * originV[2] - originR[2] * originV[1],
    originR[2] * originV[0] - originR[0] * originV[2],
    originR[0] * originV[1] - originR[1] * originV[0],
  ];
  const angularNorm = Math.hypot(angular[0], angular[1], angular[2]);
  const normal = angular.map((item) => item / angularNorm);
  const along = [
    normal[1] * radial[2] - normal[2] * radial[1],
    normal[2] * radial[0] - normal[0] * radial[2],
    normal[0] * radial[1] - normal[1] * radial[0],
  ];
  const relative = [otherR[0] - originR[0], otherR[1] - originR[1], otherR[2] - originR[2]];
  const dot = (left, right) => left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
  const rangeKm = Math.hypot(relative[0], relative[1], relative[2]) / 1000;
  return {
    radialKm: dot(relative, radial) / 1000,
    alongKm: dot(relative, along) / 1000,
    normalKm: dot(relative, normal) / 1000,
    rangeKm,
    label: `${rangeKm < 10 ? rangeKm.toFixed(2) : rangeKm.toFixed(0)} км`,
  };
}

function formatClock(seconds) {
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return [hours, minutes, secs].map((part) => String(part).padStart(2, "0")).join(":");
}
