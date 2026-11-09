// SPDX-License-Identifier: Apache-2.0
// Day page: a 72x40 m care facility, 12 residents over 06:30-21:30, driven by the
// demo01 trajectory. Agents that are moving show as ground arrows; agents inside a
// generated motion clip will be drawn as full characters (added in a later step).

import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/controls/OrbitControls.js';
import { GLTFLoader } from '/static/vendor/loaders/GLTFLoader.js';
import { clone as cloneSkeleton } from '/static/vendor/utils/SkeletonUtils.js';

const DEG = Math.PI / 180;
const T_START = 0;
const T_END = 54000;
const WALL_HEIGHT = 1.35;

const ui = {
  badge: document.getElementById('badge'),
  subtitle: document.getElementById('subtitle'),
  clock: document.getElementById('clock'),
  scrub: document.getElementById('scrub'),
  play: document.getElementById('play'),
  speed: document.getElementById('speed'),
  ticks: document.getElementById('ticks'),
  events: document.getElementById('events'),
  legend: document.getElementById('legend'),
  agentList: document.getElementById('agent-list'),
  inspectorBody: document.getElementById('inspector-body'),
  inspectorEmpty: document.getElementById('inspector-empty'),
  inspectorChip: document.getElementById('inspector-chip'),
  inspectorName: document.getElementById('inspector-name'),
  inspectorTime: document.getElementById('inspector-time'),
  inspectorZone: document.getElementById('inspector-zone'),
  inspectorActivity: document.getElementById('inspector-activity'),
  inspectorState: document.getElementById('inspector-state'),
  inspectorNext: document.getElementById('inspector-next'),
  toggleLabels: document.getElementById('toggle-labels'),
  toggleFurniture: document.getElementById('toggle-furniture'),
  toggleArrows: document.getElementById('toggle-arrows'),
  toggleClips: document.getElementById('toggle-clips'),
  eventCard: document.getElementById('event-card'),
  eventTitle: document.getElementById('event-title'),
  eventDetail: document.getElementById('event-detail'),
  eventBeats: document.getElementById('event-beats'),
  toggleFollow: document.getElementById('toggle-follow'),
};

const state = {
  facility: null,
  day: null,
  agents: [],
  zones: new Map(),
  pois: new Map(),
  t: 0,
  playing: false,
  speed: 60,
  selected: null,
  labels: [],
  furniture: null,
  arrows: null,
  joints: [],
  clips: [],
  clipBuffer: new Float32Array(0),
  clipsByAgent: new Map(),
  characters: new Map(),
  showClips: true,
  scenario: null,
};

// ---------------------------------------------------------------- renderer

const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070a0f);
scene.fog = new THREE.Fog(0x070a0f, 90, 190);

const camera = new THREE.PerspectiveCamera(40, 1, 0.5, 400);
camera.up.set(0, 0, 1);
camera.position.set(32, -8, 84);

const orbit = new OrbitControls(camera, canvas);
orbit.target.set(30, 19, 0);
orbit.enableDamping = true;
orbit.dampingFactor = 0.08;
orbit.maxPolarAngle = Math.PI * 0.42;
orbit.minDistance = 6;
orbit.maxDistance = 180;

scene.add(new THREE.HemisphereLight(0xd6e4ff, 0x1a1f2b, 1.7));
scene.add(new THREE.AmbientLight(0xffffff, 0.18));

const sun = new THREE.DirectionalLight(0xfff4e6, 2.4);
sun.position.set(28, -34, 60);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -55;
sun.shadow.camera.right = 55;
sun.shadow.camera.top = 45;
sun.shadow.camera.bottom = -45;
sun.shadow.camera.near = 1;
sun.shadow.camera.far = 160;
sun.shadow.bias = -0.0012;
scene.add(sun);

const fill = new THREE.DirectionalLight(0x8fb4ff, 0.55);
fill.position.set(-30, 30, 40);
scene.add(fill);

// ---------------------------------------------------------------- palette

const MATERIALS = {
  wall: new THREE.MeshStandardMaterial({ color: 0xcfd7e6, roughness: 0.9, metalness: 0.0 }),
  wood: new THREE.MeshStandardMaterial({ color: 0xb08968, roughness: 0.85 }),
  darkWood: new THREE.MeshStandardMaterial({ color: 0x8a6a4f, roughness: 0.85 }),
  fabric: new THREE.MeshStandardMaterial({ color: 0x8fa3c7, roughness: 0.95 }),
  sheet: new THREE.MeshStandardMaterial({ color: 0xeef2fa, roughness: 0.95 }),
  metal: new THREE.MeshStandardMaterial({ color: 0xb8c2d0, roughness: 0.55, metalness: 0.35 }),
  dark: new THREE.MeshStandardMaterial({ color: 0x2a3242, roughness: 0.7 }),
  green: new THREE.MeshStandardMaterial({ color: 0x6fae74, roughness: 0.95 }),
  screen: new THREE.MeshStandardMaterial({ color: 0x1b2330, roughness: 0.4, metalness: 0.2 }),
  accent: new THREE.MeshStandardMaterial({ color: 0xe0a3a3, roughness: 0.9 }),
};

function box(w, d, h, material, x = 0, y = 0, z = 0) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, d, h), material);
  mesh.position.set(x, y, z + h / 2);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function cylinder(radius, height, material, x = 0, y = 0, z = 0, radial = 20) {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, height, radial), material);
  mesh.position.set(x, y, z + height / 2);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

// ---------------------------------------------------------------- building

function buildFloors(facility) {
  const group = new THREE.Group();
  const base = new THREE.Mesh(
    new THREE.PlaneGeometry(90, 60),
    new THREE.MeshStandardMaterial({ color: 0x0d1219, roughness: 1.0 }),
  );
  base.position.set(36, 20, -0.05);
  base.receiveShadow = true;
  group.add(base);

  for (const zone of facility.zones) {
    const [x0, y0, x1, y1] = zone.bbox;
    const width = x1 - x0;
    const depth = y1 - y0;
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(width - 0.06, depth - 0.06),
      new THREE.MeshStandardMaterial({ color: new THREE.Color(zone.color), roughness: 0.96, metalness: 0.0 }),
    );
    floor.position.set((x0 + x1) / 2, (y0 + y1) / 2, 0.01);
    floor.receiveShadow = true;
    group.add(floor);
  }
  scene.add(group);
  return group;
}

function buildWalls(facility) {
  const group = new THREE.Group();
  for (const [x1, y1, x2, y2] of facility.walls) {
    const horizontal = Math.abs(y1 - y2) < 1e-6;
    const length = Math.hypot(x2 - x1, y2 - y1);
    if (length < 0.05) continue;
    const geometry = horizontal
      ? new THREE.BoxGeometry(length, 0.14, WALL_HEIGHT)
      : new THREE.BoxGeometry(0.14, length, WALL_HEIGHT);
    const mesh = new THREE.Mesh(geometry, MATERIALS.wall);
    mesh.position.set((x1 + x2) / 2, (y1 + y2) / 2, WALL_HEIGHT / 2);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);
  }
  scene.add(group);
  return group;
}

function buildFurniture(facility) {
  const group = new THREE.Group();
  for (const poi of facility.pois) {
    const item = new THREE.Group();
    const [w, d, h] = poi.size;
    switch (poi.type) {
      case 'bed':
        item.add(box(w, d, h * 0.55, MATERIALS.wood));
        item.add(box(w - 0.06, d - 0.12, 0.16, MATERIALS.sheet, 0, 0, h * 0.55));
        item.add(box(0.55, 0.35, 0.1, MATERIALS.sheet, 0, -d / 2 + 0.32, h * 0.55 + 0.16));
        item.add(box(w, 0.08, 0.75, MATERIALS.wood, 0, -d / 2, 0));
        break;
      case 'chair':
        item.add(box(0.46, 0.46, 0.06, MATERIALS.fabric, 0, 0, 0.42));
        item.add(box(0.46, 0.07, 0.5, MATERIALS.fabric, 0, 0.19, 0.46));
        for (const [lx, ly] of [[-0.18, -0.18], [0.18, -0.18], [-0.18, 0.18], [0.18, 0.18]]) {
          item.add(box(0.04, 0.04, 0.42, MATERIALS.darkWood, lx, ly, 0));
        }
        break;
      case 'dining_table':
        item.add(cylinder(0.66, 0.05, MATERIALS.wood, 0, 0, 0.72, 24));
        item.add(cylinder(0.08, 0.7, MATERIALS.metal, 0, 0, 0, 12));
        item.add(cylinder(0.34, 0.04, MATERIALS.metal, 0, 0, 0, 16));
        break;
      case 'sofa':
        item.add(box(w, d, 0.38, MATERIALS.fabric, 0, 0, 0.1));
        item.add(box(w, 0.18, 0.5, MATERIALS.fabric, 0, d / 2 - 0.09, 0.42));
        item.add(box(0.14, d, 0.42, MATERIALS.fabric, -w / 2 + 0.07, 0, 0.18));
        item.add(box(0.14, d, 0.42, MATERIALS.fabric, w / 2 - 0.07, 0, 0.18));
        break;
      case 'bench':
        item.add(box(w, d, 0.08, MATERIALS.wood, 0, 0, 0.42));
        item.add(box(0.1, d, 0.42, MATERIALS.metal, -w / 2 + 0.12, 0, 0));
        item.add(box(0.1, d, 0.42, MATERIALS.metal, w / 2 - 0.12, 0, 0));
        break;
      case 'tv':
        item.add(box(w, 0.08, h, MATERIALS.screen, 0, 0, 0.55));
        item.add(box(0.4, 0.3, 0.55, MATERIALS.dark, 0, 0, 0));
        break;
      case 'piano':
        item.add(box(w, d, 0.32, MATERIALS.darkWood, 0, 0, 0.7));
        item.add(box(w - 0.1, 0.16, 0.05, MATERIALS.sheet, 0, -d / 2 + 0.1, 0.7));
        item.add(box(0.1, 0.1, 0.7, MATERIALS.darkWood, -w / 2 + 0.1, 0.1, 0));
        item.add(box(0.1, 0.1, 0.7, MATERIALS.darkWood, w / 2 - 0.1, 0.1, 0));
        break;
      case 'planter': {
        item.add(cylinder(0.3, 0.45, MATERIALS.accent));
        const foliage = new THREE.Mesh(new THREE.SphereGeometry(0.36, 12, 10), MATERIALS.green);
        foliage.position.set(0, 0, 0.8);
        foliage.castShadow = true;
        item.add(foliage);
        break;
      }
      case 'exercise_bike':
        item.add(box(0.2, 0.9, 0.35, MATERIALS.dark));
        item.add(box(0.5, 0.14, 0.65, MATERIALS.metal, 0, -0.25, 0.3));
        item.add(cylinder(0.24, 0.06, MATERIALS.dark, 0, 0.25, 0.45, 18));
        break;
      case 'parallel_bars':
        item.add(box(w, 0.06, 0.06, MATERIALS.metal, 0, -d / 2, 1.0));
        item.add(box(w, 0.06, 0.06, MATERIALS.metal, 0, d / 2, 1.0));
        for (const sx of [-1, 1]) {
          item.add(box(0.06, 0.06, 1.0, MATERIALS.metal, (sx * w) / 2, -d / 2, 0));
          item.add(box(0.06, 0.06, 1.0, MATERIALS.metal, (sx * w) / 2, d / 2, 0));
        }
        break;
      case 'treatment_table':
        item.add(box(w, d, 0.62, MATERIALS.metal));
        item.add(box(w - 0.06, d - 0.06, 0.1, MATERIALS.sheet, 0, 0, 0.62));
        break;
      case 'reception':
        item.add(box(w, d, 1.0, MATERIALS.wood));
        item.add(box(w + 0.1, d + 0.1, 0.05, MATERIALS.sheet, 0, 0, 1.0));
        break;
      case 'desk':
        item.add(box(w, d, 0.06, MATERIALS.wood, 0, 0, 0.72));
        item.add(box(0.06, d - 0.1, 0.72, MATERIALS.darkWood, -w / 2 + 0.1, 0, 0));
        item.add(box(0.06, d - 0.1, 0.72, MATERIALS.darkWood, w / 2 - 0.1, 0, 0));
        break;
      case 'bookshelf':
        item.add(box(w, d, h, MATERIALS.darkWood));
        for (const z of [0.4, 0.9, 1.4]) {
          item.add(box(w - 0.08, d + 0.02, 0.04, MATERIALS.wood, 0, 0, z));
        }
        break;
      case 'notice_board':
        item.add(box(w, d, h, MATERIALS.wood, 0, 0, 1.1));
        item.add(box(w - 0.12, d + 0.02, h - 0.12, MATERIALS.sheet, 0, 0, 1.1));
        break;
      case 'craft_table':
        item.add(box(w, d, 0.06, MATERIALS.wood, 0, 0, 0.72));
        for (const [lx, ly] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
          item.add(box(0.06, 0.06, 0.72, MATERIALS.darkWood, (lx * (w - 0.2)) / 2, (ly * (d - 0.2)) / 2, 0));
        }
        break;
      default:
        item.add(box(w, d, h, MATERIALS.wood));
    }
    item.position.set(poi.xy[0], poi.xy[1], 0);
    item.rotation.z = poi.yaw;
    group.add(item);
  }
  scene.add(group);
  return group;
}

// ---------------------------------------------------------------- labels

function makeLabel(text, color = '#dfe6f2') {
  const canvasEl = document.createElement('canvas');
  canvasEl.width = 512;
  canvasEl.height = 128;
  const ctx = canvasEl.getContext('2d');
  ctx.font = '600 56px "Microsoft YaHei", "Segoe UI", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(6, 10, 16, 0.55)';
  const width = ctx.measureText(text).width + 48;
  ctx.fillRect((512 - width) / 2, 26, width, 76);
  ctx.fillStyle = color;
  ctx.fillText(text, 256, 66);

  const texture = new THREE.CanvasTexture(canvasEl);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }));
  sprite.scale.set(4.6, 1.15, 1);
  return sprite;
}

function buildLabels(facility) {
  const labels = [];
  for (const zone of facility.zones) {
    const isRoom = zone.type === 'resident_room';
    const text = isRoom ? zone.id : zone.name;
    const sprite = makeLabel(text, isRoom ? '#9fb0cc' : '#e8eef8');
    sprite.scale.set(isRoom ? 2.2 : 5.0, isRoom ? 0.55 : 1.25, 1);
    sprite.position.set(zone.centroid[0], zone.centroid[1], isRoom ? 1.6 : 1.5);
    sprite.userData.zone = zone.id;
    scene.add(sprite);
    labels.push(sprite);
  }
  return labels;
}

// ---------------------------------------------------------------- agents

const AGENT_COLORS = [
  0x6ea8ff, 0x7dd3fc, 0x93c5fd, 0xa5b4fc, 0xc4b5fd, 0xf0abfc,
  0xf9a8d4, 0xfca5a5, 0xfdba74, 0xfcd34d, 0xa3e635, 0x5eead4,
];

function arrowGeometry() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([
    0.78, 0, 0.02, -0.42, 0.44, 0.02, -0.42, -0.44, 0.02,
  ], 3));
  geometry.setIndex([0, 1, 2]);
  geometry.computeVertexNormals();
  return geometry;
}

function buildAgents(day) {
  const group = new THREE.Group();
  const agents = [];
  day.agents.forEach((meta, index) => {
    const paletteColor = AGENT_COLORS[index % AGENT_COLORS.length];
    const color = new THREE.Color(meta.role && meta.role !== 'resident' ? (meta.color || paletteColor) : paletteColor);
    const holder = new THREE.Group();

    // Markers draw on top of walls and furniture so an agent inside a room stays visible.
    const markerMaterial = (opacity) => new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
    });

    const arrow = new THREE.Mesh(arrowGeometry(), markerMaterial(1.0));
    arrow.renderOrder = 20;
    holder.add(arrow);

    const ring = new THREE.Mesh(new THREE.RingGeometry(0.34, 0.46, 28), markerMaterial(0.55));
    ring.position.z = 0.01;
    ring.renderOrder = 19;
    holder.add(ring);

    const dot = new THREE.Mesh(new THREE.CircleGeometry(0.16, 16), markerMaterial(0.9));
    dot.position.z = 0.02;
    dot.renderOrder = 20;
    holder.add(dot);

    const label = makeLabel(meta.id, '#f2f6ff');
    label.scale.set(2.3, 0.58, 1);
    label.position.z = 1.5;
    label.material.depthTest = false;
    label.renderOrder = 21;
    holder.add(label);

    group.add(holder);
    agents.push({
      meta,
      holder,
      arrow,
      ring,
      dot,
      label,
      color,
      x: Float32Array.from(meta.x),
      y: Float32Array.from(meta.y),
      h: Float32Array.from(meta.heading),
      zone: Uint8Array.from(meta.zone),
      activity: Uint8Array.from(meta.activity),
      state: Uint8Array.from(meta.state),
      dt: meta.dt,
      t0: meta.t0,
      count: meta.x.length,
    });
  });
  scene.add(group);
  return { group, agents };
}

function sampleAgent(agent, t) {
  const f = (t - agent.t0) / agent.dt;
  const i0 = Math.max(0, Math.min(agent.count - 1, Math.floor(f)));
  const i1 = Math.min(agent.count - 1, i0 + 1);
  const a = Math.max(0, Math.min(1, f - i0));
  const x = agent.x[i0] + (agent.x[i1] - agent.x[i0]) * a;
  const y = agent.y[i0] + (agent.y[i1] - agent.y[i0]) * a;
  let dh = agent.h[i1] - agent.h[i0];
  while (dh > Math.PI) dh -= Math.PI * 2;
  while (dh < -Math.PI) dh += Math.PI * 2;
  return {
    x,
    y,
    heading: agent.h[i0] + dh * a,
    index: i0,
    zone: agent.zone[i0],
    activity: agent.activity[i0],
    state: agent.state[i0],
  };
}

// ---------------------------------------------------------------- clips

const HOLD_KINDS = new Set(['sit', 'meet_start', 'chest_collapse', 'chest_sink', 'chest_kneel', 'chest_recover']);

async function loadClips() {
  const index = await fetch('/api/day/clips').then((r) => r.json());
  const clips = (index.clips || []).filter((clip) => clip.frames > 0);
  if (!clips.length) return { clips: [], buffer: new Float32Array(0) };
  const raw = await fetch('/api/day/clips.bin').then((r) => r.arrayBuffer());
  return { clips, buffer: new Float32Array(raw) };
}

function buildClipIndex(clips) {
  const byAgent = new Map();
  for (const clip of clips) {
    if (!byAgent.has(clip.agent)) byAgent.set(clip.agent, []);
    byAgent.get(clip.agent).push(clip);
  }
  for (const list of byAgent.values()) list.sort((a, b) => a.t0 - b.t0);
  return byAgent;
}

function clipEnd(clip) {
  return clip.t0 + (clip.frames - 1) / clip.fps;
}

function findActiveClip(list, t) {
  if (!list || !list.length) return null;
  let lo = 0;
  let hi = list.length - 1;
  let candidate = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (list[mid].t0 <= t) {
      candidate = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  if (candidate < 0) return null;
  const clip = list[candidate];
  return t <= clipEnd(clip) ? clip : null;
}

function findPreviousClip(list, t) {
  if (!list || !list.length) return null;
  let lo = 0;
  let hi = list.length - 1;
  let candidate = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (list[mid].t0 <= t) {
      candidate = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return candidate >= 0 ? list[candidate] : null;
}

function resolveAgentVisual(agent, t) {
  const list = state.clipsByAgent.get(agent.meta.id);
  const active = findActiveClip(list, t);
  if (active) return { mode: 'clip', clip: active };
  const previous = findPreviousClip(list, t);
  if (previous && HOLD_KINDS.has(previous.kind)) {
    // Scenario beats (the chest-pain incident) override the recorded trajectory, so the
    // recorded position is expected to differ; normal clips must still match their anchor.
    const sample = agent.lastSample;
    const anchor = previous.end || previous.start;
    const atAnchor = sample && Math.hypot(sample.x - anchor.x, sample.y - anchor.y) < 0.8;
    const overridden = previous.scenario === 'chest_pain';
    if ((atAnchor || overridden) && t - clipEnd(previous) < 6 * 3600) return { mode: 'hold', clip: previous };
  }
  return { mode: 'arrow', clip: null };
}

function applyClipFrame(character, clip, t) {
  const local = Math.max(0, Math.min((clip.frames - 1) / clip.fps, t - clip.t0));
  const position = local * clip.fps;
  const i0 = Math.max(0, Math.min(clip.frames - 1, Math.floor(position)));
  const i1 = Math.min(clip.frames - 1, i0 + 1);
  const blend = position - i0;
  const base = clip.offset;
  const joints = clip.joints;
  for (let j = 0; j < joints; j += 1) {
    const bone = character.bones[j];
    if (!bone) continue;
    const a = base + (i0 * joints + j) * 7;
    const b = base + (i1 * joints + j) * 7;
    bone.position.set(
      state.clipBuffer[a] + (state.clipBuffer[b] - state.clipBuffer[a]) * blend,
      state.clipBuffer[a + 1] + (state.clipBuffer[b + 1] - state.clipBuffer[a + 1]) * blend,
      state.clipBuffer[a + 2] + (state.clipBuffer[b + 2] - state.clipBuffer[a + 2]) * blend,
    );
    const qa = new THREE.Quaternion(
      state.clipBuffer[a + 3], state.clipBuffer[a + 4], state.clipBuffer[a + 5], state.clipBuffer[a + 6],
    );
    const qb = new THREE.Quaternion(
      state.clipBuffer[b + 3], state.clipBuffer[b + 4], state.clipBuffer[b + 5], state.clipBuffer[b + 6],
    );
    bone.quaternion.copy(qa).slerp(qb, blend);
  }
  character.root.updateMatrixWorld(true);
}

function prepareCharacter(root, color) {
  const bones = [];
  root.traverse((object) => {
    if (object.isBone) bones.push(object);
    if (object.isSkinnedMesh) {
      object.frustumCulled = false;
      // Fifteen skinned characters casting shadows is the most expensive thing in the
      // scene and adds little at plan-view distance.
      object.castShadow = false;
      object.material = object.material.clone();
      object.material.color.lerp(color, 0.45);
      object.material.roughness = 0.75;
    }
  });
  const byName = new Map(bones.map((bone) => [bone.name, bone]));
  const ordered = state.joints.map((name) => byName.get(name) || null);
  scene.add(root);
  return { root, bones: ordered, byName };
}

async function buildCharacters(gltf) {
  const characters = new Map();
  for (const agent of state.agents) {
    const root = cloneSkeleton(gltf.scene);
    const character = prepareCharacter(root, agent.color);
    character.root.visible = false;
    characters.set(agent.meta.id, character);
  }
  return characters;
}

function updateAgents(t) {
  for (const agent of state.agents) {
    const sample = sampleAgent(agent, t);
    agent.holder.position.set(sample.x, sample.y, 0.04);
    agent.holder.rotation.z = sample.heading;
    agent.lastSample = sample;

    const visual = resolveAgentVisual(agent, t);
    agent.mode = visual.mode;
    agent.arrow.visible = visual.mode === 'arrow';
    agent.ring.visible = visual.mode === 'arrow';

    const character = state.characters.get(agent.meta.id);
    if (!character) continue;
    const show = state.showClips && visual.mode !== 'arrow';
    character.root.visible = show;
    if (show) {
      applyClipFrame(character, visual.clip, t);
    }
  }
}

// ---------------------------------------------------------------- time

function formatClock(seconds) {
  const total = 6.5 * 3600 + seconds;
  const hh = Math.floor(total / 3600) % 24;
  const mm = Math.floor((total % 3600) / 60);
  const ss = Math.floor(total % 60);
  return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}

function setTime(seconds, fromUser = false) {
  state.t = Math.max(T_START, Math.min(T_END, seconds));
  if (fromUser) {
    state.playing = false;
    ui.play.textContent = '▶';
  }
  ui.clock.textContent = formatClock(state.t);
  ui.scrub.value = String(Math.round(state.t));
  updateAgents(state.t);
  updateInspector();
  updateEventCard(state.t);
}

const clock = new THREE.Clock();

function tick() {
  requestAnimationFrame(tick);
  const delta = Math.min(clock.getDelta(), 0.1);
  if (state.playing) {
    let next = state.t + delta * state.speed;
    if (next >= T_END) {
      next = T_END;
      state.playing = false;
      ui.play.textContent = '▶';
    }
    setTime(next);
  }
  if (state.followAgent) {
    const agent = state.agents.find((a) => a.meta.id === state.followAgent);
    if (agent) {
      orbit.target.lerp(new THREE.Vector3(agent.holder.position.x, agent.holder.position.y, 0.6), 0.08);
    }
  }
  orbit.update();
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------- panels

function buildEventMarks() {
  const bins = 45;
  const counts = new Array(bins).fill(0);
  for (const clip of state.clips) {
    const index = Math.max(0, Math.min(bins - 1, Math.floor((clip.t0 / T_END) * bins)));
    counts[index] += 1;
  }
  const max = Math.max(1, ...counts);
  ui.events.innerHTML = counts
    .map((count, index) => {
      if (!count) return '';
      const height = 4 + Math.round((count / max) * 10);
      const left = ((index + 0.5) / bins) * 100;
      return `<span style="left:${left}%;top:${16 - height}px;width:3px;height:${height}px;border-radius:2px;background:rgba(110,231,183,0.55)" data-bin="${index}"></span>`;
    })
    .join('');
  ui.events.querySelectorAll('span').forEach((mark) => {
    mark.addEventListener('click', () => {
      const index = Number(mark.dataset.bin);
      setTime((index / bins) * T_END, true);
    });
  });

  const chest = state.scenario;
  if (chest) {
    const left = (chest.t / T_END) * 100;
    const mark = document.createElement('span');
    mark.className = 'chest';
    mark.style.left = `${left}%`;
    mark.title = chest.title;
    mark.setAttribute('data-chest', '1');
    mark.addEventListener('click', () => setTime(chest.t, true));
    ui.events.appendChild(mark);
  }
}

function updateEventCard(t) {
  const chest = state.scenario;
  if (!chest) return;
  const visible = t >= chest.t - 60 && t <= chest.end + 180;
  ui.eventCard.hidden = !visible;
  if (!visible) return;
  ui.eventTitle.textContent = chest.title;
  ui.eventDetail.textContent = chest.detail;
  ui.eventBeats.innerHTML = chest.beats
    .map((beat) => `<li><b>${formatClock(beat.t)}</b> ${beat.agent} · ${beat.label}</li>`)
    .join('');
}

function buildLegend(facility) {
  const seen = new Map();
  for (const zone of facility.zones) {
    if (!seen.has(zone.type)) seen.set(zone.type, zone);
  }
  ui.legend.innerHTML = [...seen.values()]
    .map((zone) => `<li><i style="background:${zone.color}"></i>${zone.label}</li>`)
    .join('');
}

function buildAgentList() {
  ui.agentList.innerHTML = state.agents
    .map(
      (agent) =>
        `<li data-id="${agent.meta.id}"><span class="chip" style="background:#${agent.color.getHexString()}"></span>` +
        `<span>${agent.meta.name}</span><span class="where">${agent.meta.room}</span></li>`,
    )
    .join('');
  ui.agentList.querySelectorAll('li').forEach((item) => {
    item.addEventListener('click', () => selectAgent(item.dataset.id));
  });
}

function selectAgent(id) {
  state.selected = id;
  ui.agentList.querySelectorAll('li').forEach((item) => {
    item.classList.toggle('active', item.dataset.id === id);
  });
  updateInspector();
}

function nextEpisode(agentId, t) {
  const episodes = state.day.episodes.filter((e) => e.agent === agentId && e.t0 > t);
  episodes.sort((a, b) => a.t0 - b.t0);
  return episodes[0] || null;
}

function updateInspector() {
  if (!state.selected) return;
  const agent = state.agents.find((a) => a.meta.id === state.selected);
  if (!agent || !agent.lastSample) return;
  const sample = agent.lastSample;
  const zoneId = state.day.zones[sample.zone];
  const zone = state.zones.get(zoneId);
  const activity = state.day.activities[sample.activity] || '—';
  const pending = nextEpisode(agent.meta.id, state.t);

  ui.inspectorEmpty.hidden = true;
  ui.inspectorBody.hidden = false;
  ui.inspectorChip.style.background = `#${agent.color.getHexString()}`;
  ui.inspectorName.textContent = `${agent.meta.name}（${agent.meta.id}）`;
  ui.inspectorTime.textContent = formatClock(state.t);
  ui.inspectorZone.textContent = zone ? zone.name : zoneId;
  ui.inspectorActivity.textContent = activity;
  ui.inspectorState.textContent = state.day.states[sample.state] || '—';
  ui.inspectorNext.textContent = pending
    ? `${formatClock(pending.t0)} ${state.zones.get(pending.zone)?.name || pending.zone} · ${pending.activity}`
    : '—';
}

function buildTicks() {
  const marks = [];
  for (let hour = 7; hour <= 21; hour += 2) {
    const seconds = (hour - 6.5) * 3600;
    if (seconds < T_START || seconds > T_END) continue;
    marks.push(`<span style="left:${(seconds / T_END) * 100}%">${String(hour).padStart(2, '0')}:00</span>`);
  }
  ui.ticks.innerHTML = marks.join('');
}

// ---------------------------------------------------------------- boot

function mergeStaff(day, staffAgents) {
  if (!staffAgents || !staffAgents.length) return day;
  const activities = day.activities.slice();
  const zones = day.zones.slice();
  const states = day.states.slice();
  let working = states.indexOf('在岗');
  if (working < 0) {
    states.push('在岗');
    working = states.length - 1;
  }
  const agents = staffAgents.map((agent) => {
    const activity = agent.activity.map((name) => {
      let index = activities.indexOf(name);
      if (index < 0) {
        activities.push(name);
        index = activities.length - 1;
      }
      return index;
    });
    const zone = agent.zone.map((id) => {
      let index = zones.indexOf(id);
      if (index < 0) {
        zones.push(id);
        index = zones.length - 1;
      }
      return index;
    });
    return { ...agent, activity, zone, state: agent.x.map(() => working) };
  });
  return { ...day, activities, zones, states, agents: [...day.agents, ...agents] };
}

async function boot() {
  const [facility, dayRaw, staffData, scenario, clipData, gltf] = await Promise.all([
    fetch('/api/day/facility').then((r) => r.json()),
    fetch('/api/day/day').then((r) => r.json()),
    fetch('/api/day/staff').then((r) => r.json()).catch(() => ({ agents: [] })),
    fetch('/api/day/scenario').then((r) => r.json()).catch(() => ({ chestPain: null })),
    loadClips(),
    new GLTFLoader().loadAsync('/static/soma.glb'),
  ]);
  const day = mergeStaff(dayRaw, staffData.agents || []);
  state.scenario = scenario.chestPain || null;
  state.facility = facility;
  state.day = day;
  state.zones = new Map(facility.zones.map((z) => [z.id, z]));
  state.pois = new Map(facility.pois.map((p) => [p.id, p]));

  buildFloors(facility);
  buildWalls(facility);
  state.furniture = buildFurniture(facility);
  state.labels = buildLabels(facility);

  const built = buildAgents(day);
  state.agents = built.agents;
  state.arrows = built.group;

  let skinned = null;
  gltf.scene.traverse((object) => {
    if (object.isSkinnedMesh && !skinned) skinned = object;
  });
  state.joints = skinned ? skinned.skeleton.bones.map((bone) => bone.name) : [];
  state.clips = clipData.clips;
  state.clipBuffer = clipData.buffer;
  state.clipsByAgent = buildClipIndex(state.clips);
  state.characters = await buildCharacters(gltf);

  buildLegend(facility);
  buildAgentList();
  buildTicks();
  buildEventMarks();
  selectAgent(day.agents[0].id);

  const clipCount = state.clips.length;
  ui.subtitle.textContent = `06:30 – 21:30 · ${day.agents.length} 位居民 · ${clipCount} 段动作 · 72 × 40 m`;
  ui.badge.textContent = '就绪';
  ui.badge.className = 'badge';
  setTime(0);
  window.__day = state;
  window.__three = THREE;
  window.__camera = camera;
  window.__orbit = orbit;
  window.__setTime = (seconds) => setTime(seconds, true);
}

function bindUi() {
  ui.scrub.addEventListener('input', () => {
    setTime(Number.parseFloat(ui.scrub.value), true);
  });
  ui.play.addEventListener('click', () => {
    if (state.t >= T_END) setTime(0);
    state.playing = !state.playing;
    ui.play.textContent = state.playing ? '❚❚' : '▶';
  });
  ui.speed.addEventListener('change', () => {
    state.speed = Number.parseFloat(ui.speed.value);
  });
  ui.toggleLabels.addEventListener('change', () => {
    for (const label of state.labels) label.visible = ui.toggleLabels.checked;
  });
  ui.toggleFurniture.addEventListener('change', () => {
    state.furniture.visible = ui.toggleFurniture.checked;
  });
  ui.toggleArrows.addEventListener('change', () => {
    state.arrows.visible = ui.toggleArrows.checked;
  });
  ui.toggleFollow.addEventListener('change', () => {
    state.followAgent = ui.toggleFollow.checked ? state.selected : null;
  });
  if (ui.toggleClips) {
    ui.toggleClips.addEventListener('change', () => {
      state.showClips = ui.toggleClips.checked;
      updateAgents(state.t);
    });
  }
}

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}

window.addEventListener('resize', resize);
resize();
bindUi();
tick();
boot().catch((error) => {
  ui.badge.textContent = `加载失败：${error.message}`;
  ui.badge.className = 'badge error';
  console.error(error);
});
