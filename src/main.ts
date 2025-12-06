import { float, Fn, fract, mx_noise_float, length, positionLocal, time, uniform, vec3, vec4, instancedBufferAttribute, color, instancedArray } from 'three/tsl';
import './style.css'
import * as THREE from 'three/webgpu';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js'
import { OrbitControls } from 'three/examples/jsm/Addons.js';

async function main() {

  const positionsArray = await fetch("/out.bin")
    .then(response => response.arrayBuffer())
    .then(data => {
      return new Float32Array(data)
    })
  const positionsCount = positionsArray.length / 3
  console.log({ len: positionsArray.length, positionsCount })
  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(25, window.innerWidth / window.innerHeight, 0.1, 10)
  camera.position.z = 1.5
  camera.far = 3000
  camera.near = 0.01

  const renderer = new THREE.WebGPURenderer()
  renderer.setSize(window.innerWidth, window.innerHeight)
  document.body.appendChild(renderer.domElement)
  renderer.setAnimationLoop(animate)

  const controls = new OrbitControls(camera, renderer.domElement)

  window.addEventListener('resize', function () {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
  })

  const options = {
    zoom: 5,
  }

  const uZoom = uniform(options.zoom)

  const material = new THREE.SpriteNodeMaterial({
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  })

  const positionsBuffer = instancedArray(positionsArray, 'vec3')

  material.positionNode = positionsBuffer.toAttribute()

  const geometry = new THREE.PlaneGeometry(0.05, 0.05)
  const mesh = new THREE.InstancedMesh(geometry, material, positionsCount)
  scene.add(mesh)

  const gui = new GUI()
  gui.add(options, 'zoom', 1, 100).onChange((value) => {
    uZoom.value = value
  })

  function animate() {
    renderer.render(scene, camera)
  }
}

main()
