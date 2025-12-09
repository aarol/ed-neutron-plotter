import { OrbitControls } from 'three/examples/jsm/Addons.js';
import { color, float, Fn, instancedArray, uniform, vec4 } from 'three/tsl';
import { AdditiveBlending, CubeTextureLoader, InstancedMesh, PerspectiveCamera, PlaneGeometry, Scene, SpriteNodeMaterial, Vector3, WebGPURenderer } from 'three/webgpu';


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

  async init() {
    this.camera.position.set(34.65699659876029, 21.90527423256544, -24.079356892645272);

    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setClearColor('#000000');
    document.body.appendChild(this.renderer.domElement);

    await this.renderer.init();

    this.controls.enableDamping = true;
    this.controls.minDistance = 0.1;
    this.controls.maxDistance = 150;
    this.controls.update()

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

    // ambient light
    // const ambientLight = new AmbientLight('#ffffff', 0.5);
    // this.scene.add(ambientLight);

    this.loadStars()
  }

  setTarget(target: Vector3) {
    this.targetPosition = target
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
    this.renderer.render(this.scene, this.camera)

    if (this.currentPosition.distanceToSquared(this.targetPosition) > 0.01) {
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

  async loadStars() {
    console.log('Loading star data...')
    const starPositionArrays = await Promise.all([0, 1, 2, 3]
      .map(i => fetch(`/neutron_coords_${i}.bin`)
        .then(res => res.arrayBuffer())
        .then(arr => new Float32Array(arr)))
    )

    const count = starPositionArrays.reduce((acc, arr) => acc + arr.length / 3, 0)
    console.log(`Loaded ${count} stars`)

    for (let i = 0; i < starPositionArrays.length; i++) {
      const arr = starPositionArrays[i];
      console.log(arr.length)
      const positionBuffer = instancedArray(arr, 'vec3');
      console.log(positionBuffer)

      // nodes
      const material = new SpriteNodeMaterial({ blending: AdditiveBlending, depthWrite: false, depthTest: false });
      const colorA = uniform(color('#5900ff'));

      new Vector3()
      material.positionNode = positionBuffer.toAttribute().div(float(1000));

      // const colors = [vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1), vec3(1, 1, 1)]
      material.colorNode = Fn(() => {
        return vec4(colorA, 1);
      })();

      material.scaleNode = float(0.05);

      // mesh
      const geometry = new PlaneGeometry(0.5, 0.5);
      const mesh = new InstancedMesh(geometry, material, arr.length / 3);
      mesh.frustumCulled = false
      this.scene.add(mesh);
      this.requestRenderIfNotRequested()
    }
  }
}
