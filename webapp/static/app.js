// SPDX-License-Identifier: Apache-2.0
// Browser front end: place a start state and an end state in a 3D scene, ask the
// MotionBricks service for the motion in between, and play it back.

import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/controls/OrbitControls.js';
import { TransformControls } from '/static/vendor/controls/TransformControls.js';
import { GLTFLoader } from '/static/vendor/loaders/GLTFLoader.js';
import { clone as cloneSkeleton } from '/static/vendor/utils/SkeletonUtils.js';
import { GLTFExporter } from '/static/vendor/exporters/GLTFExporter.js';

const DEG = Math.PI / 180;
const FPS = 30;
const UP = new THREE.Vector3(0, 0, 1);

const ui = {
  statusText: document.getElementById('status-text'),
  statusBadge: document.getElementById('status-badge'),
  statusBar: document.getElementById('status-bar'),
  generate: document.getElementById('generate'),
  hint: document.getElementById('hint'),
  play: document.getElementById('play'),
  scrub: document.getElementById('scrub'),
  timeLabel: document.getElementById('time-label'),
  loop: document.getElementById('loop'),
  exportBtn: document.getElementById('export'),
  pace: document.getElementById('pace'),
  step: document.getElementById('step'),
  stepLabel: document.getElementById('step-label'),
  startPose: document.getElementById('start-pose'),
  endPose: document.getElementById('end-pose'),
  startX: document.getElementById('start-x'),
  startY: document.getElementById('start-y'),
  endX: document.getElementById('end-x'),
  endY: document.getElementById('end-y'),
  startHeading: document.getElementById('start-heading'),
  endHeading: document.getElementById('end-heading'),
  startHeadingLabel: document.getElementById('start-heading-label'),
  endHeadingLabel: document.getElementById('end-heading-label'),
  statSegments: document.getElementById('stat-segments'),
  statDuration: document.getElementById('stat-duration'),
  statFrames: document.getElementById('stat-frames'),
  statInference: document.getElementById('stat-inference'),
  statError: document.getElementById('stat-error'),
};

const state = {
  ready: false,
  joints: [],
  numJoints: 0,
  poseSpecs: [],
  poseCache: new Map(),
  modelUrl: null,
  gltf: null,
  rigs: { start: null, end: null, actor: null },
  markers: {
    start: { x: 0, y: 0, heading: 0, pose: 'idle' },
    end: { x: 2.4, y: 0, heading: 0, pose: 'idle' },
  },
  activeTarget: 'start',
  frames: null,
  numFrames: 0,
  duration: 0,
  time: 0,
  playing: false,
  lastResult: null,
};

// ---------------------------------------------------------------- renderer

const canvas = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0c11);
scene.fog = new THREE.Fog(0x0a0c11, 16, 46);

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
camera.up.copy(UP);
camera.position.set(3.6, -4.6, 2.7);

const orbit = new OrbitControls(camera, canvas);
orbit.target.set(1.3, 0, 0.9);
orbit.enableDamping = true;
orbit.dampingFactor = 0.08;
orbit.maxPolarAngle = Math.PI * 0.495;
orbit.minDistance = 1.5;
orbit.maxDistance = 26;

const hemi = new THREE.HemisphereLight(0xa9c0ff, 0x1b2130, 1.5);
scene.add(hemi);

const key = new THREE.DirectionalLight(0xffffff, 2.6);
key.position.set(5, -6, 9);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.near = 0.5;
key.shadow.camera.far = 40;
key.shadow.camera.left = -10;
key.shadow.camera.right = 10;
key.shadow.camera.top = 10;
key.shadow.camera.bottom = -10;
key.shadow.bias = -0.0008;
scene.add(key);

const rim = new THREE.DirectionalLight(0x6ea8ff, 0.9);
rim.position.set(-6, 4, 3);
scene.add(rim);

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(70, 70),
  new THREE.MeshStandardMaterial({ color: 0x1a2130, roughness: 0.94, metalness: 0.0 }),
);
ground.receiveShadow = true;
scene.add(ground);

const grid = new THREE.GridHelper(40, 40, 0x4c5f86, 0x2b3648);
grid.rotation.x = Math.PI / 2;
grid.material.transparent = true;
grid.material.opacity = 0.85;
grid.position.z = 0.002;
scene.add(grid);

// ---------------------------------------------------------------- markers

function makeMarker(color) {
  const group = new THREE.Group();
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.26, 0.34, 56),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, side: THREE.DoubleSide }),
  );
  ring.position.z = 0.01;
  group.add(ring);

  const disc = new THREE.Mesh(
    new THREE.CircleGeometry(0.26, 48),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.13, side: THREE.DoubleSide }),
  );
  disc.position.z = 0.008;
  group.add(disc);

  const arrow = new THREE.Mesh(
    new THREE.ConeGeometry(0.075, 0.34, 16),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.95 }),
  );
  arrow.rotation.z = -Math.PI / 2;
  arrow.position.set(0.5, 0, 0.06);
  group.add(arrow);

  const shaft = new THREE.Mesh(
    new THREE.BoxGeometry(0.26, 0.022, 0.022),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6 }),
  );
  shaft.position.set(0.33, 0, 0.06);
  group.add(shaft);

  group.userData.arrow = arrow;
  group.userData.shaft = shaft;
  return group;
}

const markers = {
  start: makeMarker(0x4ade80),
  end: makeMarker(0x60a5fa),
};
scene.add(markers.start, markers.end);

const gizmo = new TransformControls(camera, renderer.domElement);
gizmo.setMode('translate');
gizmo.setSpace('world');
gizmo.showZ = false;
gizmo.size = 0.8;
gizmo.translationSnap = 0.05;
scene.add(gizmo);

gizmo.addEventListener('dragging-changed', (event) => {
  orbit.enabled = !event.value;
});

gizmo.addEventListener('objectChange', () => {
  const target = state.activeTarget;
  const marker = markers[target];
  marker.position.z = 0;
  state.markers[target].x = marker.position.x;
  state.markers[target].y = marker.position.y;
  if (target === 'start') {
    ui.startX.value = marker.position.x.toFixed(2);
    ui.startY.value = marker.position.y.toFixed(2);
  } else {
    ui.endX.value = marker.position.x.toFixed(2);
    ui.endY.value = marker.position.y.toFixed(2);
  }
  updateGhost(target);
});

// ---------------------------------------------------------------- rigs

function prepareRig(root) {
  const bones = [];
  let skinned = null;
  root.traverse((object) => {
    if (object.isBone) bones.push(object);
    if (object.isSkinnedMesh && !skinned) skinned = object;
  });
  const byName = new Map();
  for (const bone of bones) byName.set(bone.name, bone);
  return { root, bones, byName, skinned };
}

function cloneRig(source, { ghostColor = null, opacity = 1 } = {}) {
  const root = cloneSkeleton(source);
  root.traverse((object) => {
    if (object.isMesh || object.isSkinnedMesh) {
      object.castShadow = opacity === 1;
      object.receiveShadow = false;
      object.frustumCulled = false;
      if (ghostColor !== null) {
        object.material = new THREE.MeshStandardMaterial({
          color: ghostColor,
          emissive: ghostColor,
          emissiveIntensity: 0.35,
          transparent: true,
          opacity,
          roughness: 0.55,
          metalness: 0.0,
          depthWrite: false,
        });
      }
    }
  });
  scene.add(root);
  return prepareRig(root);
}

function applyCanonicalPose(rig, poseData, marker) {
  const spec = state.poseCache.get(marker.pose);
  if (!spec) return;
  const joints = state.joints;
  for (let j = 0; j < joints.length; j += 1) {
    const bone = rig.byName.get(joints[j]);
    if (!bone) continue;
    const o = j * 7;
    bone.position.set(poseData[o], poseData[o + 1], poseData[o + 2]);
    bone.quaternion.set(poseData[o + 3], poseData[o + 4], poseData[o + 5], poseData[o + 6]);
  }
  const rootBone = rig.byName.get(joints[0]);
  if (rootBone) {
    const yaw = marker.heading * DEG - spec.forwardYaw;
    const placement = new THREE.Quaternion().setFromAxisAngle(UP, yaw);
    rootBone.quaternion.premultiply(placement);
    rootBone.position.set(marker.x, marker.y, spec.rootHeight);
  }
  rig.root.updateMatrixWorld(true);
}

function updateGhost(which) {
  const rig = state.rigs[which];
  const spec = state.poseCache.get(state.markers[which].pose);
  if (!rig || !spec) return;
  applyCanonicalPose(rig, spec.data, state.markers[which]);
  // Before anything is generated, the character itself shows the start state.
  if (which === 'start' && !state.frames && state.rigs.actor) {
    applyCanonicalPose(state.rigs.actor, spec.data, state.markers.start);
  }
}

function updateMarkerVisual(which) {
  const marker = markers[which];
  const data = state.markers[which];
  marker.position.set(data.x, data.y, 0);
  const yaw = data.heading * DEG;
  const arrow = marker.userData.arrow;
  const shaft = marker.userData.shaft;
  if (arrow) {
    arrow.position.set(Math.cos(yaw) * 0.5, Math.sin(yaw) * 0.5, 0.06);
    arrow.rotation.z = yaw - Math.PI / 2;
  }
  if (shaft) {
    shaft.position.set(Math.cos(yaw) * 0.33, Math.sin(yaw) * 0.33, 0.06);
    shaft.rotation.z = yaw;
  }
}

function updateAllMarkers() {
  updateMarkerVisual('start');
  updateMarkerVisual('end');
}

// ---------------------------------------------------------------- loading

async function loadScene() {
  const info = await fetch('/api/scene').then((r) => r.json());
  state.joints = info.joints;
  state.numJoints = info.numJoints;
  state.poseSpecs = info.poses;
  state.modelUrl = info.modelUrl;

  const options = state.poseSpecs
    .map((spec) => `<option value="${spec.key}">${spec.label}</option>`)
    .join('');
  ui.startPose.innerHTML = options;
  ui.endPose.innerHTML = options;
  ui.startPose.value = state.markers.start.pose;
  ui.endPose.value = state.markers.end.pose;

  const gltf = await new GLTFLoader().loadAsync(state.modelUrl);
  state.gltf = gltf.scene;
  state.rigs.start = cloneRig(gltf.scene, { ghostColor: 0x4ade80, opacity: 0.22 });
  state.rigs.end = cloneRig(gltf.scene, { ghostColor: 0x60a5fa, opacity: 0.22 });
  state.rigs.actor = cloneRig(gltf.scene);

  await Promise.all(
    [...new Set([state.markers.start.pose, state.markers.end.pose])].map(fetchPose),
  );

  updateAllMarkers();
  updateGhost('start');
  updateGhost('end');

  gizmo.attach(markers.start);
  setActiveTarget('start');
  state.ready = true;
  ui.generate.disabled = false;
  ui.hint.textContent = '拖动彩色圆环或输入坐标布置起止状态，然后点击生成。';
}

async function fetchPose(poseKey) {
  if (state.poseCache.has(poseKey)) return state.poseCache.get(poseKey);
  const raw = await fetch(`/api/pose/${poseKey}`).then((r) => r.json());
  const bytes = Uint8Array.from(atob(raw.data), (c) => c.charCodeAt(0));
  const spec = {
    data: new Float32Array(bytes.buffer),
    forwardYaw: raw.forwardYaw,
    rootHeight: raw.rootHeight,
    numJoints: raw.numJoints,
  };
  state.poseCache.set(poseKey, spec);
  return spec;
}

// ---------------------------------------------------------------- status

function setStatus(text, badge, progress, tone = '') {
  ui.statusText.textContent = text;
  ui.statusBadge.textContent = badge;
  ui.statusBadge.className = `badge ${tone}`;
  ui.statusBar.style.width = `${Math.round(progress * 100)}%`;
}

async function pollStatus() {
  try {
    const status = await fetch('/api/status').then((r) => r.json());
    if (status.state === 'ready') {
      setStatus('引擎就绪', '就绪', 1);
      if (!state.ready) {
        try {
          await loadScene();
          setStatus('引擎就绪', '就绪', 1);
        } catch (error) {
          setStatus(`场景加载失败：${error.message}`, '错误', 0, 'error');
        }
      }
      return;
    }
    if (status.state === 'error') {
      setStatus(status.error || status.message, '错误', 0, 'error');
      return;
    }
    setStatus(status.message, status.state === 'building' ? '构建引擎' : '加载中', status.progress || 0, 'warn');
  } catch (error) {
    setStatus('无法连接后端服务', '离线', 0, 'error');
  }
  setTimeout(pollStatus, 800);
}

// ---------------------------------------------------------------- playback

function applyFrames(frames, numFrames, numJoints, timeSeconds) {
  const rig = state.rigs.actor;
  if (!rig || !frames) return;
  const maxFrame = numFrames - 1;
  const position = THREE.MathUtils.clamp(timeSeconds * FPS, 0, maxFrame);
  const i0 = Math.floor(position);
  const i1 = Math.min(i0 + 1, maxFrame);
  const alpha = position - i0;
  const joints = state.joints;

  for (let j = 0; j < numJoints; j += 1) {
    const bone = rig.byName.get(joints[j]);
    if (!bone) continue;
    const a = (i0 * numJoints + j) * 7;
    const b = (i1 * numJoints + j) * 7;
    bone.position.set(
      THREE.MathUtils.lerp(frames[a], frames[b], alpha),
      THREE.MathUtils.lerp(frames[a + 1], frames[b + 1], alpha),
      THREE.MathUtils.lerp(frames[a + 2], frames[b + 2], alpha),
    );
    const qa = new THREE.Quaternion(frames[a + 3], frames[a + 4], frames[a + 5], frames[a + 6]);
    const qb = new THREE.Quaternion(frames[b + 3], frames[b + 4], frames[b + 5], frames[b + 6]);
    bone.quaternion.copy(qa).slerp(qb, alpha);
  }
  rig.root.updateMatrixWorld(true);
}

function setTime(seconds) {
  state.time = THREE.MathUtils.clamp(seconds, 0, Math.max(0, state.duration));
  if (state.frames) {
    applyFrames(state.frames, state.numFrames, state.numJoints, state.time);
  }
  ui.scrub.value = state.duration > 0 ? state.time / state.duration : 0;
  ui.timeLabel.textContent = `${state.time.toFixed(2)} s / ${state.duration.toFixed(2)} s`;
}

const clock = new THREE.Clock();

function tick() {
  requestAnimationFrame(tick);
  const delta = Math.min(clock.getDelta(), 0.1);
  if (state.playing && state.duration > 0) {
    let next = state.time + delta;
    if (next >= state.duration) {
      if (ui.loop.checked) {
        next = 0;
      } else {
        next = state.duration;
        state.playing = false;
        ui.play.textContent = '▶';
      }
    }
    setTime(next);
  }
  orbit.update();
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------- actions

function setActiveTarget(which) {
  state.activeTarget = which;
  gizmo.attach(markers[which]);
  document.querySelectorAll('.card[data-target]').forEach((card) => {
    card.classList.toggle('selected', card.dataset.target === which);
  });
  document.querySelectorAll('.edit-btn').forEach((button) => {
    button.classList.toggle('active', button.dataset.target === which);
  });
}

async function onPoseChange(which, poseKey) {
  state.markers[which].pose = poseKey;
  await fetchPose(poseKey);
  updateGhost(which);
}

function bindInputs() {
  ui.startPose.addEventListener('change', (e) => onPoseChange('start', e.target.value));
  ui.endPose.addEventListener('change', (e) => onPoseChange('end', e.target.value));

  document.querySelectorAll('.edit-btn').forEach((button) => {
    button.addEventListener('click', () => setActiveTarget(button.dataset.target));
  });

  const bindPosition = (which, inputX, inputY) => {
    const handler = () => {
      const x = Number.parseFloat(inputX.value);
      const y = Number.parseFloat(inputY.value);
      if (Number.isFinite(x)) state.markers[which].x = x;
      if (Number.isFinite(y)) state.markers[which].y = y;
      updateMarkerVisual(which);
      updateGhost(which);
    };
    inputX.addEventListener('input', handler);
    inputY.addEventListener('input', handler);
  };
  bindPosition('start', ui.startX, ui.startY);
  bindPosition('end', ui.endX, ui.endY);

  const bindHeading = (which, input, label) => {
    input.addEventListener('input', () => {
      const value = Number.parseFloat(input.value);
      state.markers[which].heading = value;
      label.textContent = `${value.toFixed(0)}°`;
      updateMarkerVisual(which);
      updateGhost(which);
    });
  };
  bindHeading('start', ui.startHeading, ui.startHeadingLabel);
  bindHeading('end', ui.endHeading, ui.endHeadingLabel);

  ui.step.addEventListener('input', () => {
    ui.stepLabel.textContent = `${Number.parseFloat(ui.step.value).toFixed(1)} m`;
  });

  ui.scrub.addEventListener('input', () => {
    state.playing = false;
    ui.play.textContent = '▶';
    setTime(Number.parseFloat(ui.scrub.value) * state.duration);
  });

  ui.play.addEventListener('click', () => {
    if (!state.frames) return;
    if (!state.playing && state.time >= state.duration - 1e-3) setTime(0);
    state.playing = !state.playing;
    ui.play.textContent = state.playing ? '❚❚' : '▶';
  });

  ui.generate.addEventListener('click', generate);
  ui.exportBtn.addEventListener('click', exportGlb);
}

async function generate() {
  if (!state.ready) return;
  ui.generate.disabled = true;
  ui.generate.textContent = '生成中……';
  ui.hint.classList.remove('error');
  ui.hint.textContent = '正在调用 MotionBricks 生成动作……';

  const body = {
    startPose: state.markers.start.pose,
    endPose: state.markers.end.pose,
    start: [state.markers.start.x, state.markers.start.y],
    end: [state.markers.end.x, state.markers.end.y],
    startHeading: state.markers.start.heading,
    endHeading: state.markers.end.heading,
    pace: ui.pace.value,
    step: Number.parseFloat(ui.step.value),
  };

  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail?.message || detail.detail || `HTTP ${response.status}`);
    }
    const result = await response.json();
    const bytes = Uint8Array.from(atob(result.frames), (c) => c.charCodeAt(0));
    state.frames = new Float32Array(bytes.buffer);
    state.numFrames = result.numFrames;
    state.numJoints = result.numJoints;
    state.duration = result.duration;
    state.lastResult = result;

    ui.play.disabled = false;
    ui.scrub.disabled = false;
    ui.exportBtn.disabled = false;

    const totalInference = result.segments.reduce((sum, s) => sum + s.inferenceMs, 0);
    ui.statSegments.textContent = `${result.segments.length} 段`;
    ui.statDuration.textContent = `${result.duration.toFixed(2)} s`;
    ui.statFrames.textContent = `${result.numFrames} 帧`;
    ui.statInference.textContent = `${totalInference.toFixed(1)} ms`;

    const last = result.rootPath[result.rootPath.length - 1];
    const error = Math.hypot(last[0] - state.markers.end.x, last[1] - state.markers.end.y);
    ui.statError.textContent = `${(error * 100).toFixed(1)} cm`;

    ui.hint.textContent = `生成完成：${result.segments.length} 段 / ${result.duration.toFixed(2)} s，端到端 ${Math.round(result.elapsedMs)} ms。`;
    setTime(0);
    state.playing = true;
    ui.play.textContent = '❚❚';
  } catch (error) {
    ui.hint.classList.add('error');
    ui.hint.textContent = `生成失败：${error.message}`;
  } finally {
    ui.generate.disabled = false;
    ui.generate.textContent = '生成中间动作';
  }
}

function downloadBuffer(buffer, filename) {
  const blob = new Blob([buffer], { type: 'model/gltf-binary' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function exportGlb() {
  if (!state.frames || !state.gltf) return;
  const rig = cloneSkeleton(state.gltf);
  rig.traverse((object) => {
    if (object.isMesh || object.isSkinnedMesh) {
      object.castShadow = false;
      object.frustumCulled = false;
    }
  });

  const joints = state.joints;
  const numFrames = state.numFrames;
  const times = new Float32Array(numFrames);
  for (let f = 0; f < numFrames; f += 1) times[f] = f / FPS;

  const tracks = [];
  for (let j = 0; j < joints.length; j += 1) {
    const positions = new Float32Array(numFrames * 3);
    const quaternions = new Float32Array(numFrames * 4);
    for (let f = 0; f < numFrames; f += 1) {
      const o = (f * joints.length + j) * 7;
      positions[f * 3] = state.frames[o];
      positions[f * 3 + 1] = state.frames[o + 1];
      positions[f * 3 + 2] = state.frames[o + 2];
      quaternions[f * 4] = state.frames[o + 3];
      quaternions[f * 4 + 1] = state.frames[o + 4];
      quaternions[f * 4 + 2] = state.frames[o + 5];
      quaternions[f * 4 + 3] = state.frames[o + 6];
    }
    tracks.push(new THREE.VectorKeyframeTrack(`${joints[j]}.position`, times, positions));
    tracks.push(new THREE.QuaternionKeyframeTrack(`${joints[j]}.quaternion`, times, quaternions));
  }

  const clip = new THREE.AnimationClip('motionbricks', state.duration, tracks);
  new GLTFExporter().parse(
    rig,
    (buffer) => downloadBuffer(buffer, 'motionbricks.web.glb'),
    (error) => {
      ui.hint.classList.add('error');
      ui.hint.textContent = `导出失败：${error}`;
    },
    { binary: true, animations: [clip], trs: true },
  );
}

// ---------------------------------------------------------------- boot

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

window.addEventListener('resize', resize);

function connectSocket() {
  try {
    const socket = new WebSocket(`ws://${window.location.host}/ws`);
    socket.addEventListener('message', (event) => {
      const data = JSON.parse(event.data);
      if (data.state === 'ready') {
        setStatus('引擎就绪', '就绪', 1);
      } else if (data.state === 'error') {
        setStatus(data.message, '错误', 0, 'error');
      } else {
        setStatus(data.message, '构建引擎', data.progress || 0, 'warn');
      }
    });
    socket.addEventListener('error', () => socket.close());
  } catch (error) {
    /* polling still covers the status */
  }
}

resize();
updateAllMarkers();
bindInputs();
tick();
connectSocket();
pollStatus();

// Handy for debugging from the browser console.
window.__scene = scene;
window.__state = state;
window.__three = THREE;
window.__camera = camera;
window.__renderer = renderer;
