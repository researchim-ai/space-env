import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

const COLORS = ["#9fd4ff", "#f0c27a", "#d5e2c5", "#e7b4c3"];
const view = document.querySelector("#view");
const readout = document.querySelector("#readout");
const clock = document.querySelector("#clock");
const scrub = document.querySelector("#scrub");
const playButton = document.querySelector("#play");
const legend = document.querySelector("#legend");

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
view.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color("#05070c");
scene.fog = new THREE.FogExp2("#05070c", 0.012);

const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.01, 80);
camera.position.set(3.1, 1.55, 2.4);
camera.up.set(0, 1, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 1.4;
controls.maxDistance = 12;
controls.target.set(0, 0, 0);

const sunLight = new THREE.DirectionalLight("#fff4e0", 3.1);
scene.add(sunLight);
scene.add(new THREE.AmbientLight("#8aa4c8", 0.22));

const stars = new THREE.BufferGeometry();
const starPositions = new Float32Array(1800 * 3);
for (let i = 0; i < 1800; i += 1) {
  const radius = 28 + Math.random() * 18;
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  starPositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
  starPositions[i * 3 + 1] = radius * Math.cos(phi);
  starPositions[i * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);
}
stars.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
scene.add(new THREE.Points(stars, new THREE.PointsMaterial({ color: "#d5deea", size: 0.035, sizeAttenuation: true })));

const earthPivot = new THREE.Group();
scene.add(earthPivot);

const loader = new THREE.TextureLoader();
const dayMap = loader.load("./textures/earth_day.jpg");
const specMap = loader.load("./textures/earth_spec.jpg");
const normalMap = loader.load("./textures/earth_normal.jpg");
dayMap.colorSpace = THREE.SRGBColorSpace;
specMap.colorSpace = THREE.NoColorSpace;

const earth = new THREE.Mesh(
  new THREE.SphereGeometry(1, 128, 96),
  new THREE.MeshPhongMaterial({
    map: dayMap,
    specularMap: specMap,
    normalMap: normalMap,
    specular: new THREE.Color("#9ec3db"),
    shininess: 16,
  }),
);
earthPivot.add(earth);

const atmosphere = new THREE.Mesh(
  new THREE.SphereGeometry(1.045, 96, 64),
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
        float rim = pow(1.0 - abs(dot(normalize(vNormal), viewDir)), 2.6);
        gl_FragColor = vec4(glowColor, rim * 0.9);
      }
    `,
  }),
);
scene.add(atmosphere);

function eciToWorld(values, scale) {
  return new THREE.Vector3(values[0], values[2], values[1]).multiplyScalar(scale);
}

function makeSatellite(color) {
  const group = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(0.028, 0.02, 0.03),
    new THREE.MeshStandardMaterial({
      color,
      metalness: 0.55,
      roughness: 0.32,
      emissive: color,
      emissiveIntensity: 0.25,
    }),
  );
  const panel = new THREE.MeshStandardMaterial({
    color: "#16365f",
    metalness: 0.7,
    roughness: 0.22,
    emissive: "#0b1c33",
    emissiveIntensity: 0.35,
  });
  const left = new THREE.Mesh(new THREE.BoxGeometry(0.075, 0.0018, 0.024), panel);
  const right = left.clone();
  left.position.x = -0.055;
  right.position.x = 0.055;
  group.add(body, left, right);
  group.userData.body = body;
  return group;
}

let episode = null;
let scale = 1;
const satellites = new Map();
const stationMeshes = [];
let playing = true;
let frameIndex = 0;
let accumulator = 0;

function formatClock(seconds) {
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return [hours, minutes, secs].map((part) => String(part).padStart(2, "0")).join(":");
}

function buildOrbits() {
  const names = Object.keys(episode.frames[0].vehicles);
  names.forEach((name, index) => {
    const color = COLORS[index % COLORS.length];
    const positions = [];
    episode.frames.forEach((frame) => positions.push(eciToWorld(frame.vehicles[name].r, scale)));
    const geometry = new THREE.BufferGeometry().setFromPoints(positions);
    scene.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 })));
    const sat = makeSatellite(color);
    scene.add(sat);
    satellites.set(name, sat);
    const chip = document.createElement("span");
    chip.innerHTML = `<i class="swatch" style="background:${color}"></i>${name}`;
    legend.appendChild(chip);
  });
}

function ensureStations(count) {
  while (stationMeshes.length < count) {
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.012, 16, 12),
      new THREE.MeshBasicMaterial({ color: "#f3e2bc" }),
    );
    earthPivot.add(mesh);
    stationMeshes.push(mesh);
  }
}

function showFrame(index) {
  const frame = episode.frames[index];
  const sun = eciToWorld(frame.sun, scale).normalize();
  sunLight.position.copy(sun.clone().multiplyScalar(8));
  earthPivot.rotation.y = -frame.gast;
  earthPivot.updateMatrixWorld(true);

  ensureStations(frame.stations.length);
  frame.stations.forEach((station, stationIndex) => {
    const world = eciToWorld(station.r, scale);
    stationMeshes[stationIndex].position.copy(earthPivot.worldToLocal(world));
    stationMeshes[stationIndex].visible = true;
  });
  for (let extra = frame.stations.length; extra < stationMeshes.length; extra += 1) {
    stationMeshes[extra].visible = false;
  }

  const lines = [`<b>${formatClock(frame.t)}</b>`];
  Object.entries(frame.vehicles).forEach(([name, vehicle]) => {
    const sat = satellites.get(name);
    const position = eciToWorld(vehicle.r, scale);
    const velocity = eciToWorld(vehicle.v, 1);
    sat.position.copy(position);
    const forward = velocity.normalize();
    if (forward.lengthSq() > 0) {
      sat.up.copy(sun);
      sat.lookAt(position.clone().add(forward));
    }
    sat.userData.body.material.emissiveIntensity = vehicle.eclipse ? 0.05 : 0.28;
    const link = vehicle.contact ? vehicle.station || "связь" : "нет линии";
    lines.push(`${name}  ${vehicle.altitude_km.toFixed(0)} км  SoC ${(vehicle.soc * 100).toFixed(0)}%  ${vehicle.eclipse ? "тень" : "свет"}  ${link}`);
  });
  readout.innerHTML = lines.join("<br>");
  clock.textContent = formatClock(frame.t);
  scrub.value = String(index);
}

async function main() {
  const response = await fetch("./sample.json");
  episode = await response.json();
  scale = 1 / episode.earth_radius_m;
  scrub.max = String(episode.frames.length - 1);
  buildOrbits();
  showFrame(0);
  renderer.setAnimationLoop(tick);
}

function tick() {
  if (playing && episode) {
    accumulator += 1;
    const framesPerTick = 0.35;
    if (accumulator >= 1 / framesPerTick) {
      accumulator = 0;
      frameIndex = (frameIndex + 1) % episode.frames.length;
      showFrame(frameIndex);
    }
  }
  controls.update();
  renderer.render(scene, camera);
}

playButton.addEventListener("click", () => {
  playing = !playing;
  playButton.textContent = playing ? "Пауза" : "Пуск";
});
scrub.addEventListener("input", () => {
  frameIndex = Number(scrub.value);
  showFrame(frameIndex);
});
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

main();
