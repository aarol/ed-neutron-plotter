import { float, Fn, uniform, vec4, color, instancedArray } from 'three/tsl';
import './style.css'
import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/examples/jsm/Addons.js';
let camera = new THREE.PerspectiveCamera(25, window.innerWidth / window.innerHeight, 0.1, 100);
let scene = new THREE.Scene();
let renderer = new THREE.WebGPURenderer({
  antialias: true,
});

let controls = new OrbitControls(camera, renderer.domElement);

async function main() {
  camera.position.set(3, 5, 8);
  // camera.near = 0.05;
  camera.far = 200

  // ambient light

  const ambientLight = new THREE.AmbientLight('#ffffff', 0.5);
  scene.add(ambientLight);

  // renderer


  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor('#000000');
  document.body.appendChild(renderer.domElement);

  await renderer.init();

  controls.enableDamping = true;
  controls.minDistance = 0.1;
  controls.maxDistance = 150;
  controls.update()

  controls.addEventListener('change', requestRenderIfNotRequested)

  window.addEventListener('resize', onWindowResize);

  await loadStars()

  render()
}

function onWindowResize() {

  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
  requestRenderIfNotRequested()
}

function requestRenderIfNotRequested() {
  if (!renderRequested) {
    renderRequested = true
    requestAnimationFrame(render)
  }
}

let renderRequested = false
function render() {
  renderRequested = false

  controls.update()
  renderer.render(scene, camera)
}

async function loadStars() {
  const starPositionArrays = await Promise.all([0, 1, 2, 3]
    .map(i => fetch(`/neutron_coords_${i}.bin`)
      .then(res => res.arrayBuffer())
      .then(arr => new Float32Array(arr)))
  )

  const count = starPositionArrays.reduce((acc, arr) => acc + arr.length / 3, 0)
  console.log(`Loaded star data with ${count} stars`)

  for (const arr of starPositionArrays) {
    const positionBuffer = instancedArray(arr, 'vec3');

    // nodes
    const material = new THREE.SpriteNodeMaterial({ blending: THREE.AdditiveBlending, depthWrite: false });
    const colorA = uniform(color('#5900ff'));

    material.positionNode = positionBuffer.toAttribute().div(float(1000));

    material.colorNode = Fn(() => {
      return vec4(colorA, 1);

    })();

    material.scaleNode = float(0.05);

    // mesh

    const geometry = new THREE.PlaneGeometry(0.5, 0.5);
    const mesh = new THREE.InstancedMesh(geometry, material, arr.length / 3);
    mesh.frustumCulled = false
    scene.add(mesh);
  }
}

main()
