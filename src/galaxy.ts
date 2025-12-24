import { OrbitControls } from 'three/examples/jsm/Addons.js';
import Stats from 'three/examples/jsm/libs/stats.module.js';
import { color, float, uniform, vec4 } from 'three/tsl';
import { AdditiveBlending, BufferAttribute, BufferGeometry, CubeTextureLoader, Mesh, PerspectiveCamera, Points, PointsNodeMaterial, Scene, SphereGeometry, SpriteNodeMaterial, Vector3, WebGPURenderer } from 'three/webgpu';
import { RouteLine } from './route-line';


export class Galaxy {
  camera = new PerspectiveCamera(25, window.innerWidth / window.innerHeight, 0.1, 200)
  scene = new Scene();
  renderer = new WebGPURenderer({
    antialias: true,
    depth: false,
  })

  controls = new OrbitControls(this.camera, this.renderer.domElement)
  targetPosition = new Vector3(0, 0, 0)
  currentPosition = this.targetPosition.clone()
  routeLine = new RouteLine(this.scene);

  focusSphere!: Mesh

  stats = new Stats()

  async init() {
    this.camera.position.set(34.65699659876029, 21.90527423256544, -24.079356892645272);

    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setClearColor('#000000');
    document.body.appendChild(this.renderer.domElement);

    document.body.appendChild(this.stats.dom)

    await this.renderer.init();

    this.controls.enableDamping = true;
    this.controls.minDistance = 0.1;
    this.controls.maxDistance = 150;
    this.controls.zoomSpeed = 3.0
    this.controls.update()

    // this is for you javascript (middle finger)
    const localThis = this
    this.controls.addEventListener('change', this.requestRenderIfNotRequested.bind(localThis))

    window.addEventListener('resize', this.onWindowResize.bind(localThis));

    const cubeTextureLoader = new CubeTextureLoader();
    const texture = await cubeTextureLoader.loadAsync([
      '/skybox/front.png',
      '/skybox/back.png',
      '/skybox/top.png',
      '/skybox/bottom.png',
      '/skybox/left.png',
      '/skybox/right.png',
    ])

    this.scene.background = texture;

    this.focusSphere = this.createFocusSphere()
    this.scene.add(this.focusSphere)
  }

  createFocusSphere() {
    const geometry = new SphereGeometry(0.001, 16, 16);
    const material = new SpriteNodeMaterial({
      colorNode: vec4(uniform(color('#ffffff')), float(1.0)),
      sizeAttenuation: false,
    })
    const sphere = new Mesh(geometry, material)
    sphere.position.copy(this.targetPosition)
    return sphere
  }

  setTarget(target: Vector3) {
    this.targetPosition = target
    this.focusSphere.position.copy(this.targetPosition)
    this.requestRenderIfNotRequested()
  }

  setRoutePoints(points: Vector3[]) {
    this.routeLine.updatePoints(points);
    this.requestRenderIfNotRequested()
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.requestRenderIfNotRequested()
  }

  renderRequested = false
  render() {
    this.renderRequested = false

    this.controls.update()
    this.stats.update()
    this.renderer.render(this.scene, this.camera)

    if (this.currentPosition.distanceToSquared(this.targetPosition) > 0.0001) {
      this.currentPosition.lerp(this.targetPosition, 0.08)
      this.controls.target.copy(this.currentPosition)
      this.requestRenderIfNotRequested()
    }
  }

  requestRenderIfNotRequested() {
    if (!this.renderRequested) {
      this.renderRequested = true

      requestAnimationFrame(this.render.bind(this));
    }
  }

  async loadStars(starPositionArrays: DataView[]) {
    console.log('Loading star data...')
    

    const count = starPositionArrays.reduce((acc, arr) => acc + arr.byteLength / 12 - 1, 0)
    console.log(`Loaded ${count} stars`)

    for (let i = 0; i < starPositionArrays.length; i++) {
      const arr = starPositionArrays[i];

      // let aabb_min_x = arr.getFloat32(0, true)
      // let aabb_min_y = arr.getFloat32(4, true)
      // let aabb_min_z = arr.getFloat32(8, true)
      // let aabb_min = new Vector3(aabb_min_x, aabb_min_y, aabb_min_z).divideScalar(1000)
      // let aabb_max_x = arr.getFloat32(12, true)
      // let aabb_max_y = arr.getFloat32(16, true)
      // let aabb_max_z = arr.getFloat32(20, true)
      // let aabb_max = new Vector3(aabb_max_x, aabb_max_y, aabb_max_z).divideScalar(1000)

      // console.log(`Star array ${i}: AABB min(${aabb_min_x}, ${aabb_min_y}, ${aabb_min_z}) max(${aabb_max_x}, ${aabb_max_y}, ${aabb_max_z})`)
      let starArr = new Float32Array(arr.buffer)

      const geometry = new BufferGeometry()
      geometry.setAttribute('position', new BufferAttribute(starArr, 3))

      const colorA = uniform(color('#246acb'));

      const material = new PointsNodeMaterial({
        blending: AdditiveBlending,
        depthWrite: false, depthTest: false,
        colorNode: vec4(colorA, 1),
      });

      // mesh
      const mesh = new Points(geometry, material);
      mesh.frustumCulled = false
      this.scene.add(mesh);
      this.requestRenderIfNotRequested()
    }
  }
}
